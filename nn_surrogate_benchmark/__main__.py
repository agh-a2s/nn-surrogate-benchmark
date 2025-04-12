from .surrogate import MLP, prepare_dataloaders
from .ela_comparator import SurrogateELAComparator
from .model_evaluator import ModelEvaluator
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import socket
import subprocess
import time
import webbrowser
from pytorch_lightning.callbacks import EarlyStopping
import torch


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def ensure_tensorboard_running(logdir: str, port: int = 6006) -> None:
    if not is_port_in_use(port):
        print(f"Starting TensorBoard on port {port}...")
        subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)
        webbrowser.open(f"http://localhost:{port}")
    else:
        print(f"TensorBoard already running on port {port}")


# if __name__ == "__main__":
#     column_names = ["x1", "x2"]
#     experiment_name = "bbob_tanh"
#     tensorboard_dir = "lightning_logs"
#     total_epochs = 150
#     ensure_tensorboard_running(tensorboard_dir)
#     for function_id in range(23, 25):
#         file_path = f"data/bbob_f{function_id:03d}_i01_d02_samples.csv"
#         current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
#         model = MLP(
#             input_dim=len(column_names),
#             lr=1e-3,
#             hidden_dims=[1024] * 3,
#             activation="tanh",
#             layer_norm=True,
#         )
#         train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y = (
#             prepare_dataloaders(
#                 file_path=file_path,
#                 scaler_type="robust",
#                 column_names=column_names,
#             )
#         )
#         tb_logger = TensorBoardLogger(save_dir=tensorboard_dir, name=experiment_name)
#         trainer = Trainer(
#             max_epochs=total_epochs,
#             logger=tb_logger,
#             log_every_n_steps=10,
#             accelerator="gpu",
#             devices=1,
#         )

#         trainer.fit(model, train_dataloader, val_dataloader)
#         trainer.test(model, test_dataloader)
#         trainer = Trainer(
#             max_epochs=300,
#             logger=tb_logger,
#             callbacks=[],
#             accelerator="gpu",
#             devices=1,
#         )

#         trainer.fit(model, train_dataloader, val_dataloader)
#         trainer.test(model, test_dataloader)

#         # ela_comparator = SurrogateELAComparator(model, device="cpu")
#         # ela_results = ela_comparator.compare_multiple_sets(
#         #     train_loader=train_dataloder,
#         #     val_loader=val_dataloader,
#         #     test_loader=test_dataloader,
#         #     n_points=1000,
#         # )
#         # ela_comparator.log_metrics_to_tensorboard(ela_results, total_epochs)

#         evaluator = ModelEvaluator(model, scaler_y=scaler_y, tb_logger=tb_logger)
#         results = evaluator.evaluate_multiple_sets(
#             train_loader=train_dataloader,
#             val_loader=val_dataloader,
#             test_loader=test_dataloader,
#         )
#         evaluator.log_metrics_to_tensorboard(results, total_epochs)

from .learning_to_rank import RankingMLP, prepare_ranking_dataloaders

if __name__ == "__main__":
    loss_type = "listnet"
    list_size = 1000
    batch_size = 64
    total_epochs = 50
    start_function = 21
    end_function = 24

    column_names = ["x1", "x2"]
    experiment_name = f"bbob_ranking_{loss_type}_2048"
    tensorboard_dir = "lightning_logs"

    num_train_lists = 10000
    num_val_lists = 2000
    num_test_lists = 2000

    print(f"Starting experiment with loss type: {loss_type}")
    print(f"List size: {list_size}, Batch size: {batch_size}, Epochs: {total_epochs}")
    print(f"Processing functions {start_function} to {end_function}")

    ensure_tensorboard_running(tensorboard_dir)

    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"
    print(f"Using device: {device}")

    for function_id in range(start_function, end_function + 1):
        file_path = f"data/bbob_f{function_id:03d}_i01_d02_samples.csv"
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M")

        print(f"Processing function {function_id}...")

        # Prepare data
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            regular_train_dataloader,
            regular_val_dataloader,
            regular_test_dataloader,
            scaler_x,
            scaler_y,
        ) = prepare_ranking_dataloaders(
            file_path=file_path,
            column_names=column_names,
            list_size=list_size,
            num_train_lists=num_train_lists,
            num_val_lists=num_val_lists,
            num_test_lists=num_test_lists,
            batch_size=batch_size,
            scaler_type="robust",
            device=device,
        )

        # Create model
        model = RankingMLP(
            input_dim=len(column_names),
            lr=1e-3,
            hidden_dims=[2048, 2048],
            activation="relu",
            layer_norm=True,
            loss_type=loss_type,
        )

        # Set up logger
        tb_logger = TensorBoardLogger(
            save_dir=tensorboard_dir,
            name=experiment_name,
            version=f"f{function_id:03d}",
        )

        trainer = Trainer(
            max_epochs=total_epochs,
            logger=tb_logger,
            log_every_n_steps=10,
            accelerator="gpu" if gpu_available else "cpu",
            devices=1,
        )

        print(f"Training model for function {function_id}...")
        trainer.fit(model, train_dataloader, val_dataloader)

        # Test the model
        print(f"Testing model for function {function_id}...")
        trainer.test(model, test_dataloader)

        # Evaluate the model
        print(f"Evaluating model for function {function_id}...")
        evaluator = ModelEvaluator(model, scaler_y=scaler_y, tb_logger=tb_logger)
        results = evaluator.evaluate_multiple_sets(
            train_loader=regular_train_dataloader,
            val_loader=regular_val_dataloader,
            test_loader=regular_test_dataloader,
        )
        evaluator.log_metrics_to_tensorboard(results, total_epochs)

        print(f"Completed function {function_id}")
