import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Literal, Tuple, Optional
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)

from nn_surrogate_benchmark.loss import ListNetLossFunction, RankCosineLossFunction


class RankingMLP(pl.LightningModule):
    """
    MLP model for learning to rank.
    This model is trained on batches where each batch contains multiple examples that are ranked together.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512],
        activation: Literal["tanh", "relu", "gelu", "leaky_relu"] = "tanh",
        lr: float = 1e-3,
        layer_norm: bool = False,
        loss_type: Literal["listnet", "ranknet", "rankcosine"] = "listnet",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build network architecture
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.3))
            else:
                layers.append(nn.GELU())
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

        # Set up loss function
        if loss_type == "listnet":
            self.criterion = ListNetLossFunction()
        elif loss_type == "rankcosine":
            self.criterion = RankCosineLossFunction()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Initialize weights
        self.net.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.hparams.activation in ["relu", "gelu", "leaky_relu"]:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(
                    module.weight, gain=nn.init.calculate_gain(self.hparams.activation)
                )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # Each batch is a list of examples that are ranked together
        x, y = batch
        batch_size, list_size, feature_dim = x.shape

        # Reshape for processing through network
        x_flat = x.reshape(-1, feature_dim)

        # Forward pass
        preds_flat = self.forward(x_flat)

        # Reshape predictions back to [batch_size, list_size]
        preds = preds_flat.reshape(batch_size, list_size)

        # Compute loss
        loss = self.criterion(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size, list_size, feature_dim = x.shape

        x_flat = x.reshape(-1, feature_dim)
        preds_flat = self.forward(x_flat)
        preds = preds_flat.reshape(batch_size, list_size)

        loss = self.criterion(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        batch_size, list_size, feature_dim = x.shape

        x_flat = x.reshape(-1, feature_dim)
        preds_flat = self.forward(x_flat)
        preds = preds_flat.reshape(batch_size, list_size)

        loss = self.criterion(preds, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def create_ranking_dataset(
    x: np.ndarray,
    y: np.ndarray,
    list_size: int = 100,
    num_lists: int = 1000,
    device: str = "cpu",
    indices_pool: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dataset for learning to rank where each entry is a list of examples.

    Args:
        x: Input features of shape (num_samples, feature_dim)
        y: Target values of shape (num_samples,)
        list_size: Number of examples per list
        num_lists: Number of lists to create
        device: Device to create tensors on
        indices_pool: Optional tensor of indices to sample from. If None, all indices are used.

    Returns:
        X_lists: Tensor of shape (num_lists, list_size, feature_dim)
        y_lists: Tensor of shape (num_lists, list_size)
    """
    assert len(x.shape) == 2, "X should be of shape (num_samples, feature_dim)"
    assert len(y.shape) == 1 or (
        len(y.shape) == 2 and y.shape[1] == 1
    ), "y should be of shape (num_samples,) or (num_samples, 1)"

    if len(y.shape) == 2:
        y = y.squeeze()

    # Convert to torch tensors if they're not already
    if not isinstance(x, torch.Tensor):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x_tensor = x.to(device=device)

    if not isinstance(y, torch.Tensor):
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    else:
        y_tensor = y.to(device=device)

    # Determine the pool of indices to sample from
    if indices_pool is None:
        num_samples = x_tensor.shape[0]
        indices_pool = torch.arange(num_samples, device=device)

    # Ensure list_size doesn't exceed the number of available indices
    actual_list_size = min(list_size, len(indices_pool))
    if actual_list_size < list_size:
        print(
            f"Warning: Requested list_size {list_size} exceeds available indices {len(indices_pool)}. Using list_size={actual_list_size}"
        )

    # Create random indices for each list
    lists_indices = []
    for _ in tqdm(range(num_lists), desc="Creating ranking lists"):
        # For each list, we'll need to ensure we have enough samples
        if actual_list_size == len(indices_pool):
            # Just use all indices if we need exactly all of them
            selected_indices = indices_pool
        else:
            # Otherwise take a random subset
            perm = torch.randperm(len(indices_pool), device=device)
            selected_indices = indices_pool[perm[:actual_list_size]]
        lists_indices.append(selected_indices)

    # Stack indices
    indices_tensor = torch.stack(lists_indices)

    # Create batches by indexing
    X_lists = torch.stack([x_tensor[indices] for indices in indices_tensor])
    y_lists = torch.stack([y_tensor[indices] for indices in indices_tensor])

    return X_lists.detach(), y_lists.detach()


def prepare_ranking_dataloaders(
    file_path: str,
    column_names: list[str],
    list_size: int = 100,
    num_train_lists: int = 70000,
    num_val_lists: int = 15000,
    num_test_lists: int = 15000,
    batch_size: int = 32,
    standard_batch_size: int = 256,
    scaler_type: str = "minmax",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    device: str = "cpu",
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    DataLoader,
    DataLoader,
    DataLoader,
    object | None,
    object | None,
]:
    """
    Prepare train, validation and test dataloaders for learning to rank with no overlap between datasets.
    Returns both ranking dataloaders (with lists of examples) and standard dataloaders (with individual examples).

    Args:
        file_path: Path to CSV data file
        column_names: Column names for the input features
        list_size: Number of examples in each ranking list
        num_train_lists: Number of lists for training
        num_val_lists: Number of lists for validation
        num_test_lists: Number of lists for testing
        batch_size: Batch size for ranking dataloaders
        standard_batch_size: Batch size for standard dataloaders
        scaler_type: Type of scaling to use ('minmax', 'standard', 'robust', 'maxabs', or None)
        train_ratio: Ratio of data to use for training (default: 0.7)
        val_ratio: Ratio of data to use for validation (default: 0.15)
        device: Device to use for tensor operations during dataset creation

    Returns:
        Tuple containing:
        - train_dataloader: Ranking DataLoader for training
        - val_dataloader: Ranking DataLoader for validation
        - test_dataloader: Ranking DataLoader for testing
        - std_train_dataloader: Standard DataLoader for training (single examples)
        - std_val_dataloader: Standard DataLoader for validation (single examples)
        - std_test_dataloader: Standard DataLoader for testing (single examples)
        - scaler_x: Feature scaler (or None if scaler_type is None)
        - scaler_y: Target scaler (or None if scaler_type is None)
    """
    # Load data
    data_df = pd.read_csv(file_path)
    X = data_df[column_names].values
    y = data_df["y"].values

    # Apply scaling if requested
    scaler_x, scaler_y = None, None

    if scaler_type is not None:
        if scaler_type.lower() == "minmax":
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()
        elif scaler_type.lower() == "standard":
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
        elif scaler_type.lower() == "robust":
            scaler_x = RobustScaler()
            scaler_y = RobustScaler()
        elif scaler_type.lower() == "maxabs":
            scaler_x = MaxAbsScaler()
            scaler_y = MaxAbsScaler()
        else:
            raise ValueError(
                "scaler_type must be one of: 'minmax', 'standard', 'robust', 'maxabs', or None"
            )

        X = scaler_x.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).squeeze()

    # Create non-overlapping train/val/test splits
    num_samples = len(X)
    perm = torch.randperm(num_samples, device=device)

    # Calculate split points
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)

    # Split indices into non-overlapping sets
    train_indices = perm[:train_end]
    val_indices = perm[train_end:val_end]
    test_indices = perm[val_end:]

    print(
        f"Split data into {len(train_indices)} training samples, {len(val_indices)} validation samples, and {len(test_indices)} test samples"
    )

    # Convert NumPy arrays to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)

    # Create standard dataloaders with individual examples
    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]
    X_test = X_tensor[test_indices]
    y_test = y_tensor[test_indices]
    regular_train_dataset = TensorDataset(X_train, y_train)
    regular_val_dataset = TensorDataset(X_val, y_val)
    regular_test_dataset = TensorDataset(X_test, y_test)

    regular_train_dataloader = DataLoader(
        regular_train_dataset, batch_size=standard_batch_size, shuffle=True
    )
    regular_val_dataloader = DataLoader(
        regular_val_dataset, batch_size=standard_batch_size
    )
    regular_test_dataloader = DataLoader(
        regular_test_dataset, batch_size=standard_batch_size
    )

    # Create ranking datasets using non-overlapping indices
    X_train_lists, y_train_lists = create_ranking_dataset(
        X,
        y,
        list_size=list_size,
        num_lists=num_train_lists,
        device=device,
        indices_pool=train_indices,
    )

    X_val_lists, y_val_lists = create_ranking_dataset(
        X,
        y,
        list_size=list_size,
        num_lists=num_val_lists,
        device=device,
        indices_pool=val_indices,
    )

    X_test_lists, y_test_lists = create_ranking_dataset(
        X,
        y,
        list_size=list_size,
        num_lists=num_test_lists,
        device=device,
        indices_pool=test_indices,
    )

    train_dataset = TensorDataset(X_train_lists, y_train_lists)
    val_dataset = TensorDataset(X_val_lists, y_val_lists)
    test_dataset = TensorDataset(X_test_lists, y_test_lists)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        regular_train_dataloader,
        regular_val_dataloader,
        regular_test_dataloader,
        scaler_x,
        scaler_y,
    )
