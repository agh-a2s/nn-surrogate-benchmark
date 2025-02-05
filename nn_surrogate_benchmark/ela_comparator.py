import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pflacco.classical_ela_features import (
    calculate_dispersion,
    calculate_nbc,
    calculate_ela_meta,
    calculate_cm_angle,
    calculate_information_content,
)


class SurrogateELAComparator:
    """
    A class to compare ELA features between original data and MLP predictions.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

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
        self, dataloader: DataLoader, n_points: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract X and y from a dataloader.

        Args:
            dataloader: DataLoader to extract data from
            n_points: Maximum number of points to extract. If None, extract all points.
        """
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        total_points = 0

        for X_batch, y_batch in dataloader:
            batch_X = self._prepare_data(X_batch)[0]
            batch_y = self._prepare_data(y_batch)[0]

            if n_points is not None:
                remaining = n_points - total_points
                if remaining <= 0:
                    break
                if remaining < len(batch_X):
                    batch_X = batch_X[:remaining]
                    batch_y = batch_y[:remaining]

            X_list.append(batch_X)
            y_list.append(batch_y)
            total_points += len(batch_X)

        return np.vstack(X_list), np.vstack(y_list)

    def _calculate_ela_features(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Calculate ELA features for given X and y."""
        features: dict[str, float] = {}

        disp = calculate_dispersion(X, y)
        features.update(disp)

        nbc = calculate_nbc(X, y)
        features.update(nbc)

        meta = calculate_ela_meta(X, y)
        features.update(meta)

        ic = calculate_information_content(X, y, seed=100)
        features.update(ic)

        cm = calculate_cm_angle(X, y)
        features.update(cm)
        return features

    def compare_features(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        dataloader: DataLoader | None = None,
        n_points: int | None = None,
    ) -> pd.DataFrame:
        if dataloader is not None:
            X, y = self._extract_from_dataloader(dataloader, n_points)
        else:
            X, y = self._prepare_data(X, y)

        y_pred = self._get_predictions(X)
        original_features = self._calculate_ela_features(X, y)
        predicted_features = self._calculate_ela_features(X, y_pred)

        differences: dict[str, float] = {}
        for feature in original_features:
            differences[feature] = abs(
                original_features[feature] - predicted_features[feature]
            )

        diff_df = pd.DataFrame(
            {
                "feature": list(differences.keys()),
                "absolute_difference": list(differences.values()),
            }
        )

        diff_df["relative_difference_percent"] = diff_df["absolute_difference"].apply(
            lambda x: x * 100 if abs(x) <= 1 else x
        )

        return diff_df.sort_values("absolute_difference", ascending=False)

    def compare_multiple_sets(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        n_points: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}

        if train_loader is not None:
            results["train"] = self.compare_features(
                dataloader=train_loader, n_points=n_points
            )

        if val_loader is not None:
            results["validation"] = self.compare_features(
                dataloader=val_loader, n_points=n_points
            )

        if test_loader is not None:
            results["test"] = self.compare_features(
                dataloader=test_loader, n_points=n_points
            )

        return results
