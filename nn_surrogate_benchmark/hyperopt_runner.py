#!/usr/bin/env python
import argparse
import os
import subprocess
import time
import webbrowser
import socket
import logging
from datetime import datetime

from .hyperopt import OptunaHyperOptimizer
from .model_evaluator import ModelEvaluator
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def start_optuna_dashboard(storage: str, port: int = 8080) -> subprocess.Popen:
    if storage.startswith("sqlite:///"):
        db_path = storage[len("sqlite:///") :]
        if not os.path.exists(db_path):
            logger.info(
                f"Storage file {db_path} does not exist. Skipping Optuna Dashboard."
            )
            return None

    if not is_port_in_use(port):
        logger.info(f"Starting Optuna Dashboard on port {port}...")
        dashboard_process = subprocess.Popen(
            ["optuna-dashboard", storage, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)  # Give it time to start
        webbrowser.open(f"http://localhost:{port}")
        return dashboard_process
    else:
        logger.info(
            f"Port {port} is already in use. Optuna Dashboard may already be running."
        )
        return None


def start_tensorboard(logdir: str, port: int = 6006) -> subprocess.Popen:
    if not is_port_in_use(port):
        logger.info(f"Starting TensorBoard on port {port}...")
        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port)],
        )
        time.sleep(2)  # Give it time to start
        webbrowser.open(f"http://localhost:{port}")
        return tensorboard_process
    else:
        logger.info(
            f"Port {port} is already in use. TensorBoard may already be running."
        )
        return None


def run_hyperopt(args):
    os.makedirs(args.results_dir, exist_ok=True)
    db_path = os.path.join(args.results_dir, f"{args.study_name}.db")
    storage = f"sqlite:///{db_path}"

    dashboard_process = start_optuna_dashboard(storage, args.dashboard_port)

    tensorboard_process = start_tensorboard(args.tensorboard_dir, args.tensorboard_port)

    try:
        optimizer = OptunaHyperOptimizer(
            data_file_path=args.data_file,
            column_names=args.column_names.split(","),
            model_type=args.model_type,
            study_name=args.study_name,
            storage=storage,
            n_trials=args.n_trials,
            timeout=args.timeout,
            tensorboard_dir=args.tensorboard_dir,
            results_dir=args.results_dir,
        )

        study = optimizer.optimize()

        if args.train_best:
            logger.info("Training final model with best hyperparameters...")
            best_model = optimizer.load_best_model()

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=20,
                    verbose=True,
                    mode="min",
                )
            ]

            tb_logger = TensorBoardLogger(
                save_dir=args.tensorboard_dir, name=f"{args.study_name}_best_model"
            )

            trainer = Trainer(
                max_epochs=300,
                logger=tb_logger,
                callbacks=callbacks,
                gradient_clip_val=1.0,
                accumulate_grad_batches=4,
                accelerator="gpu" if args.gpu else "cpu",
                devices=1,
            )

            trainer.fit(
                best_model, optimizer.train_dataloader, optimizer.val_dataloader
            )

            trainer.test(best_model, optimizer.test_dataloader)

            evaluator = ModelEvaluator(
                best_model, scaler_y=optimizer.scaler_y, tb_logger=tb_logger
            )
            results = evaluator.evaluate_multiple_sets(
                train_loader=optimizer.train_dataloader,
                val_loader=optimizer.val_dataloader,
                test_loader=optimizer.test_dataloader,
            )

            for dataset, metrics in results.items():
                logger.info(f"Final {dataset} metrics:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value}")

        logger.info(
            f"Hyperparameter optimization completed. Best parameters: {study.best_params}"
        )
        logger.info(f"Best validation loss: {study.best_value}")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    finally:
        # Keep the dashboard and TensorBoard running if requested
        if not args.keep_dashboard_running and dashboard_process:
            logger.info("Shutting down Optuna Dashboard...")
            dashboard_process.terminate()

        if not args.keep_tensorboard_running and tensorboard_process:
            logger.info("Shutting down TensorBoard...")
            tensorboard_process.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for surrogate models"
    )

    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to the CSV data file"
    )
    parser.add_argument(
        "--column-names",
        type=str,
        required=True,
        help="Comma-separated list of column names for input features",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp",
        choices=["mlp"],
        help="Type of model to optimize",
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=f"surrogate_opt_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the optimization",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="lightning_logs",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="optuna_results",
        help="Directory to save optimization results",
    )

    # Dashboard parameters
    parser.add_argument(
        "--dashboard-port", type=int, default=8080, help="Port for Optuna Dashboard"
    )
    parser.add_argument(
        "--tensorboard-port", type=int, default=6006, help="Port for TensorBoard"
    )
    parser.add_argument(
        "--keep-dashboard-running",
        action="store_true",
        help="Keep Optuna Dashboard running after optimization",
    )
    parser.add_argument(
        "--keep-tensorboard-running",
        action="store_true",
        help="Keep TensorBoard running after optimization",
    )

    # Final model training
    parser.add_argument(
        "--train-best",
        action="store_true",
        help="Train a final model with the best hyperparameters",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU for training if available"
    )

    args = parser.parse_args()
    run_hyperopt(args)


if __name__ == "__main__":
    main()
