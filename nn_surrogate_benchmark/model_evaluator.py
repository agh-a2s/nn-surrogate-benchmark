import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
)
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger


class ModelEvaluator:
    """
    A class to evaluate model predictions and compare them with original values,
    taking into account the scaling that was applied during training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scaler_y: (
            MinMaxScaler | StandardScaler | RobustScaler | MaxAbsScaler | None
        ) = None,
        device: str = "cpu",
        tb_logger: TensorBoardLogger | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.scaler_y = scaler_y
        self.tb_logger = tb_logger

    def _prepare_data(
        self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if y is not None and isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        return X, y

    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            y_pred = self.model(X_tensor).cpu().numpy()
        return y_pred

    def _extract_from_dataloader(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []

        for batch in dataloader:
            batch_X, batch_y = self._prepare_data(batch[0], batch[1])
            X_list.append(batch_X)
            y_list.append(batch_y)

        return np.vstack(X_list), np.vstack(y_list)

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str,
    ) -> pd.DataFrame:
        X, y_true_scaled = self._extract_from_dataloader(dataloader)
        y_true_scaled = y_true_scaled.flatten()
        y_pred_scaled = self._get_predictions(X)
        y_pred_scaled = y_pred_scaled.flatten()
        results = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])

        results["y_true_scaled"] = y_true_scaled
        results["y_pred_scaled"] = y_pred_scaled

        if self.scaler_y is not None:
            y_true = self.scaler_y.inverse_transform(y_true_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

            results["y_true"] = y_true
            results["y_pred"] = y_pred
        else:
            results["y_true"] = y_true_scaled
            results["y_pred"] = y_pred_scaled

        results["abs_error"] = np.abs(results["y_true"] - results["y_pred"])
        results["rel_error"] = (
            np.abs(results["y_true"] - results["y_pred"])
            / np.abs(results["y_true"])
            * 100
        )
        results["dataset"] = dataset_name
        return results

    def evaluate_multiple_sets(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        grads = False,
    ) -> dict[str, pd.DataFrame]:
        results = {}

        if train_loader is not None:
            results["train"] = self.evaluate_dataset(train_loader, "train")

        if val_loader is not None:
            results["validation"] = self.evaluate_dataset(val_loader, "validation")

        if test_loader is not None:
            results["test"] = self.evaluate_dataset(test_loader, "test")

        return results

    def log_metrics_to_tensorboard(
        self,
        results: dict[str, pd.DataFrame],
        current_epoch: int,
    ) -> None:
        if self.tb_logger is None:
            return

        for dataset_name, df in results.items():
            self.tb_logger.experiment.add_histogram(
                f"{dataset_name}/absolute_errors", df["abs_error"].values, current_epoch
            )
            self.tb_logger.experiment.add_histogram(
                f"{dataset_name}/relative_errors", df["rel_error"].values, current_epoch
            )
            self._log_prediction_plot(df, dataset_name, current_epoch)
            self._log_contour_plots(df, dataset_name, current_epoch)

    def _log_prediction_plot(
        self, df: pd.DataFrame, dataset_name: str, current_epoch: int
    ) -> None:
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(df["y_true"], df["y_pred"], alpha=0.5)
        plt.plot(
            [df["y_true"].min(), df["y_true"].max()],
            [df["y_true"].min(), df["y_true"].max()],
            "r--",
            label="Perfect prediction",
        )
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"{dataset_name.capitalize()} Set: Predictions vs True Values")
        plt.legend()
        self.tb_logger.experiment.add_figure(
            f"{dataset_name}/predictions_vs_true",
            fig,
            current_epoch,
        )
        plt.close(fig)

    def _log_contour_plots(
        self, df: pd.DataFrame, dataset_name: str, current_epoch: int
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x1_vals = np.linspace(df["x1"].min(), df["x1"].max(), 100)
        x2_vals = np.linspace(df["x2"].min(), df["x2"].max(), 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        true_contour = ax1.tricontourf(df["x1"], df["x2"], df["y_true"])
        ax1.set_title(f"{dataset_name.capitalize()}: True Values")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        plt.colorbar(true_contour, ax=ax1)

        pred_contour = ax2.tricontourf(df["x1"], df["x2"], df["y_pred"])
        ax2.set_title(f"{dataset_name.capitalize()}: Predicted Values")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        plt.colorbar(pred_contour, ax=ax2)

        plt.tight_layout()
        self.tb_logger.experiment.add_figure(
            f"{dataset_name}/contour_comparison",
            fig,
            current_epoch,
        )
        plt.close(fig)
