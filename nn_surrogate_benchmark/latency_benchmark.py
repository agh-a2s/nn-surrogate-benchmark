import torch
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from .surrogate import MLP, prepare_dataloaders


def measure_inference_time(
    model: torch.nn.Module,
    test_loader: DataLoader,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> dict:
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup_runs):
            for batch in test_loader:
                x, _ = batch
                x = x.to(device)
                _ = model(x)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            run_latencies = []
            for batch in test_loader:
                x, _ = batch
                x = x.to(device)

                start_time = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.perf_counter()

                run_latencies.append((end_time - start_time) * 1000)
            latencies.append(np.mean(run_latencies))

    return {
        "mean_latency": np.mean(latencies),
        "std_latency": np.std(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "p95_latency": np.percentile(latencies, 95),
    }


def run_latency_benchmark(
    file_path: str,
    column_names: list[str],
    hidden_dims: list[int] = [512],
    activation: str = "relu",
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> pd.DataFrame:
    train_loader, val_loader, test_loader, _, _ = prepare_dataloaders(
        file_path=file_path,
        column_names=column_names,
        batch_size=32,
    )

    model = MLP(
        input_dim=len(column_names),
        hidden_dims=hidden_dims,
        activation=activation,
    )

    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu" if device in ["cuda", "mps"] else "cpu",
        devices=1,
        enable_progress_bar=True,
        callbacks=[ModelCheckpoint(monitor="val_loss")],
    )

    trainer.fit(model, train_loader, val_loader)
    model.eval()
    print("Measuring baseline model performance...")
    baseline_stats = measure_inference_time(
        model, test_loader, num_runs, warmup_runs, device
    )

    print("Measuring compiled model performance...")
    compiled_model = torch.compile(model)
    compiled_stats = measure_inference_time(
        compiled_model, test_loader, num_runs, warmup_runs, device
    )

    results = pd.DataFrame(
        {
            "Metric": [
                "Mean Latency (ms)",
                "Std Latency (ms)",
                "Min Latency (ms)",
                "Max Latency (ms)",
                "P95 Latency (ms)",
            ],
            "Baseline": [
                baseline_stats["mean_latency"],
                baseline_stats["std_latency"],
                baseline_stats["min_latency"],
                baseline_stats["max_latency"],
                baseline_stats["p95_latency"],
            ],
            "Compiled": [
                compiled_stats["mean_latency"],
                compiled_stats["std_latency"],
                compiled_stats["min_latency"],
                compiled_stats["max_latency"],
                compiled_stats["p95_latency"],
            ],
        }
    )

    results["Speedup"] = results["Baseline"] / results["Compiled"]
    return results


if __name__ == "__main__":
    file_path = "data/bbob_f024_i01_d02_samples.csv"
    column_names = ["x1", "x2"]

    results = run_latency_benchmark(
        file_path=file_path,
        column_names=column_names,
        hidden_dims=[512],
        activation="relu",
        num_runs=100,
        warmup_runs=10,
    )

    print("\nLatency Benchmark Results:")
    print(results.to_string(index=False))
    results.to_csv("latency_benchmark_results.csv", index=False)
