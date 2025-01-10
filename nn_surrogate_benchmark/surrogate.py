import torch
from torch import nn
from torch.optim import AdamW
from typing import Literal
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512],
        activation: Literal["tanh", "relu"] = "tanh",
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)


def prepare_dataloaders(
    file_path: str,
    batch_size: int = 128,
    train_perc: float = 0.6,
    val_perc: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    assert train_perc > 0 and val_perc > 0, "train_perc and val_perc must be positive"
    assert train_perc + val_perc < 1, "train_perc + val_perc must be less than 1"

    data_df = pd.read_csv(file_path)
    X = data_df.drop(columns=["y"]).values
    y = data_df["y"].values.reshape(-1, 1)

    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    train_size = int(train_perc * num_samples)
    val_size = int(val_perc * num_samples)
    train_idx, val_idx, test_idx = (
        indices[:train_size],
        indices[train_size : (train_size + val_size)],
        indices[(train_size + val_size) :],
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":

    model = MLP(input_dim=3)
    train_dataloder, val_dataloader, test_dataloader = prepare_dataloaders(
        file_path="heat_inversion.csv"
    )
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs", name="surrogate_experiment"
    )
    trainer = Trainer(
        max_epochs=1000,
        logger=tb_logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloder, val_dataloader)
    trainer.test(model, test_dataloader)
