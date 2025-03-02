import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Literal
import torch
import os
import logging
from datetime import datetime
import pytorch_lightning as pl
from .surrogate import MLP, prepare_dataloaders


logger = logging.getLogger(__name__)


class OptunaHyperOptimizer:
    def __init__(
        self,
        data_file_path: str,
        column_names: list[str],
        model_type: Literal["mlp"] = "mlp",
        study_name: str | None = None,
        storage: str | None = None,
        n_trials: int = 100,
        timeout: int | None = None,
        tensorboard_dir: str = "lightning_logs",
        results_dir: str = "optuna_results",
    ):
        self.data_file_path = data_file_path
        self.column_names = column_names
        self.model_type = model_type
        self.study_name = (
            study_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        self.storage = storage
        self.n_trials = n_trials
        self.timeout = timeout
        self.tensorboard_dir = tensorboard_dir
        self.results_dir = results_dir

        os.makedirs(self.results_dir, exist_ok=True)

        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.scaler_x,
            self.scaler_y,
        ) = prepare_dataloaders(
            file_path=data_file_path,
            column_names=column_names,
            scaler_type="robust",
            batch_size=256,
        )

        self.input_dim = len(column_names)

    def _create_model(self, trial: optuna.Trial) -> pl.LightningModule:
        if self.model_type == "mlp":
            n_layers = trial.suggest_int("n_layers", 1, 5)
            hidden_dim = trial.suggest_categorical(
                "hidden_dim", [64, 128, 256, 512, 1024]
            )
            hidden_dims = [hidden_dim] * n_layers

            activation = trial.suggest_categorical(
                "activation", ["tanh", "relu", "gelu", "leaky_relu"]
            )
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            layer_norm = trial.suggest_categorical("layer_norm", [True, False])

            return MLP(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                lr=lr,
                layer_norm=layer_norm,
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _objective(self, trial: optuna.Trial) -> float:
        model = self._create_model(trial)

        callbacks = []
        tb_logger = TensorBoardLogger(
            save_dir=self.tensorboard_dir,
            name=f"{self.study_name}_trial_{trial.number}",
        )

        max_epochs = 100
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.0, 3.0)
        accumulate_grad_batches = trial.suggest_categorical(
            "accumulate_grad_batches", [1, 2, 4, 8]
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            logger=tb_logger,
            callbacks=callbacks,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            accelerator=(
                "gpu"
                if torch.cuda.is_available() or torch.backends.mps.is_available()
                else "cpu"
            ),
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model, self.train_dataloader, self.val_dataloader)

        best_val_loss = trainer.callback_metrics.get("val_loss").item()

        test_results = trainer.test(model, self.test_dataloader)
        test_loss = test_results[0]["test_loss"]

        trial.set_user_attr("test_loss", test_loss)
        trial.set_user_attr("best_epoch", trainer.current_epoch)
        return best_val_loss

    def optimize(self) -> optuna.Study:
        logger.info(f"Starting hyperparameter optimization for {self.model_type}")
        logger.info(f"Study name: {self.study_name}")
        logger.info(f"Number of trials: {self.n_trials}")

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best hyperparameters: {study.best_params}")

        return study
