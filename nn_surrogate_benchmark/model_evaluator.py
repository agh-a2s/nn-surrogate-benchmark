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


class ModelEvaluator:
    """
    A class to evaluate model predictions and compare them with original values,
    taking into account the scaling that was applied during training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scaler_y: MinMaxScaler | StandardScaler | RobustScaler | MaxAbsScaler | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.scaler_y = scaler_y

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

        for X_batch, y_batch in dataloader:
            batch_X, batch_y = self._prepare_data(X_batch, y_batch)
            X_list.append(batch_X)
            y_list.append(batch_y)

        return np.vstack(X_list), np.vstack(y_list)

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str,
    ) -> pd.DataFrame:
        X, y_true_scaled = self._extract_from_dataloader(dataloader)
        y_pred_scaled = self._get_predictions(X)
        results = pd.DataFrame(X, columns=["k1", "k2", "k3"])

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
    ) -> dict[str, pd.DataFrame]:
        results = {}

        if train_loader is not None:
            results["train"] = self.evaluate_dataset(train_loader, "train")

        if val_loader is not None:
            results["validation"] = self.evaluate_dataset(val_loader, "validation")

        if test_loader is not None:
            results["test"] = self.evaluate_dataset(test_loader, "test")

        return results
