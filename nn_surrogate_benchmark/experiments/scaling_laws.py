from ..surrogate import MLP, prepare_dataloaders
from ..model_evaluator import compute_rank_correlation_metrics
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.cuda import is_available as is_cuda_available
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
import os
from ..tensorboard import ensure_tensorboard_running


class ModelPerformanceTracker:
    def __init__(self, output_dir: str = "results"):
        self.metrics_df = pd.DataFrame()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        scaler_y: None = None,
        hidden_dim_size: int = None,
        n_layers: int = None,
        device: str = "cpu",
        config_params: dict = None,
    ) -> dict:

        model.to(device)
        model.eval()

        # Collect all predictions and true values
        y_true_scaled = []
        y_pred_scaled = []

        with torch.no_grad():
            for batch in test_dataloader:
                X, y = batch
                X = X.to(device)
                y_pred = model(X)

                y_true_scaled.append(y.numpy())
                y_pred_scaled.append(y_pred.cpu().numpy())

        y_true_scaled = np.vstack(y_true_scaled).flatten()
        y_pred_scaled = np.vstack(y_pred_scaled).flatten()

        # Apply inverse transform if scaler is provided
        if scaler_y is not None:
            y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            y_true = y_true_scaled
            y_pred = y_pred_scaled

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        rank_metrics = compute_rank_correlation_metrics(y_true, y_pred)

        # Create metrics dictionary
        metrics = {
            "hidden_dim_size": hidden_dim_size,
            "n_layers": n_layers,
            "rmse": rmse,
            "r2": r2,
            "spearman_rho": rank_metrics["spearman_rho"],
            "kendall_tau": rank_metrics["kendall_tau"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add any additional config parameters
        if config_params:
            metrics.update(config_params)

        # Add row to dataframe
        self.metrics_df = pd.concat(
            [self.metrics_df, pd.DataFrame([metrics])], ignore_index=True
        )

        return metrics

    def save_results(self, filename: str = "model_performance_metrics.csv"):
        filepath = os.path.join(self.output_dir, filename)
        self.metrics_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

        return filepath


N_LAYERS = 3

if __name__ == "__main__":
    input_column_names = ["x1", "x2"]
    output_column_names = ["y"]
    fid = 23
    file_path = f"data/bbob_f0{fid}_i01_d02_samples.csv"
    total_epochs = 100
    tensorboard_dir = "lightning_logs"
    results_dir = "results"
    activation = "relu"

    device = "cuda" if is_cuda_available() else "cpu"
    accelerator = "gpu" if is_cuda_available() else "cpu"

    ensure_tensorboard_running(tensorboard_dir)

    # Initialize performance tracker
    performance_tracker = ModelPerformanceTracker(output_dir=results_dir)

    train_dataloder, val_dataloader, test_dataloader, scaler_x, scaler_y = (
        prepare_dataloaders(
            file_path=file_path,
            column_names=input_column_names,
            batch_size=256,
            scaler_type=None,
        )
    )

    config_params = {
        "function_id": fid,
        "activation": "relu",
        "epochs": total_epochs,
        "input_dims": len(input_column_names),
        "lr": 1e-4,
    }

    for hidden_dim_size in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        print(f"Training model with hidden_dim_size={hidden_dim_size}")
        experiment_name = f"bbob_f{fid}_hidden_{hidden_dim_size}"
        model = MLP(
            input_dim=len(input_column_names),
            lr=1e-4,
            hidden_dims=[hidden_dim_size] * N_LAYERS,
            activation=activation,
        )
        tb_logger = TensorBoardLogger(save_dir=tensorboard_dir, name=experiment_name)
        trainer = Trainer(
            max_epochs=total_epochs,
            logger=tb_logger,
            accelerator=accelerator,
            devices=1,
        )

        trainer.fit(model, train_dataloder, val_dataloader)
        trainer.test(model, test_dataloader)

        performance_tracker.evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
            scaler_y=scaler_y,
            hidden_dim_size=hidden_dim_size,
            n_layers=N_LAYERS,
            device=device,
            config_params=config_params,
        )

    performance_tracker.save_results(f"bbob_f{fid}_performance_metrics.csv")

    # Print summary of results
    print("\nPerformance Summary:")
    print(
        performance_tracker.metrics_df[
            ["hidden_dim_size", "rmse", "r2", "spearman_rho"]
        ]
    )
