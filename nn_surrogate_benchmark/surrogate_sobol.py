import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.optim import AdamW
from typing import Literal
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import (
	StandardScaler,
	MinMaxScaler,
	RobustScaler,
	MaxAbsScaler,
)

class Sobolev(pl.LightningModule):
	def __init__(
		self,
		input_dim: int,
		hidden_dims: list[int] = [512],
		activation: Literal["tanh", "relu", "gelu"] = "tanh",
		lr: float = 1e-3,
	) -> None:
		super().__init__()
		self.save_hyperparameters()

		layers = []
		for hidden_dim in hidden_dims:
			layers.append(nn.Linear(input_dim, hidden_dim))
			if activation == "tanh":
				layers.append(nn.Tanh())
			elif activation == "relu":
				layers.append(nn.ReLU())
			else:
				layers.append(nn.GELU())
			input_dim = hidden_dim
		layers.append(nn.Linear(input_dim, 1))
		self.net = nn.Sequential(*layers)
		self.criterion = nn.MSELoss()
		self.net.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			if self.hparams.activation in ["relu", "gelu"]:
				nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
			else:
				nn.init.xavier_normal_(
					module.weight, gain=nn.init.calculate_gain(self.hparams.activation)
				)

			if module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(self, x):
		print(f"Input x requires_grad: {x.requires_grad}")  # Debug: Check input
		for i, layer in enumerate(self.net):
			x = layer(x)
			print(f"Layer {i} output requires_grad: {x.requires_grad}")  # Debug: Check layer output
		return x

	def _shared_step(self, batch, stage: str, create_graph: bool = False):
		x: torch.Tensor
		x, y, dy_dx = batch
		x.requires_grad_(True)
		preds = self(x)
		
		grads = torch.autograd.grad(
			outputs=preds,
			inputs=x,
			grad_outputs=torch.ones_like(preds, requires_grad=True),
			create_graph=create_graph,
			retain_graph=create_graph,
			allow_unused=True
		)[0]

		loss_value = self.criterion(preds, y)
		loss_grad = self.criterion(grads, dy_dx)
		total_loss = loss_value + loss_grad

		self.log(f"{stage}_loss", total_loss, prog_bar=True, on_epoch=True)
		self.log(f"{stage}_value_loss", loss_value, on_epoch=True)
		self.log(f"{stage}_grad_loss", loss_grad, on_epoch=True)
		return total_loss

	def training_step(self, batch, batch_idx):
		return self._shared_step(batch, "train", create_graph=True)

	def validation_step(self, batch, batch_idx):
		return self._shared_step(batch, "val")

	def test_step(self, batch, batch_idx):
		return self._shared_step(batch, "test")

	def configure_optimizers(self):
		optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode="min", factor=0.1, patience=5, verbose=True
		)
		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"monitor": "val_loss",
			},
		}


def prepare_sobol_dataloaders(
	file_path: str,
	input_column_names: list[str],
	output_column_names: list[str],
	batch_size: int = 256,
	train_perc: float = 0.6,
	val_perc: float = 0.2,
	scaler_type: str = "minmax",
) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""
	Prepare train, validation, and test dataloaders with specified scaling.

	Args:
		file_path: Path to CSV data file
		input_column_names: Column names for the input features
		output_column_names: Column names for the output values and gradients
		batch_size: Batch size for dataloaders
		train_perc: Percentage of data for training
		val_perc: Percentage of data for validation
		scaler_type: Type of scaling to use ('minmax', 'standard', 'robust', or 'maxabs')
	"""
	assert train_perc > 0 and val_perc > 0, "train_perc and val_perc must be positive"
	assert train_perc + val_perc < 1, "train_perc + val_perc must be less than 1"

	# Load data
	data_df = pd.read_csv(file_path)
	X = data_df[input_column_names].values
	y = data_df[output_column_names].values

	# Split data into train, validation, and test sets
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

	# Apply scaling
	if scaler_type is None:
		scaler_x = None
		scaler_y = None
	elif scaler_type.lower() == "minmax":
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
			"scaler_type must be one of: 'minmax', 'standard', 'robust', 'maxabs'"
		)

	if scaler_x is not None:
		X_train = scaler_x.fit_transform(X_train)
		X_val = scaler_x.transform(X_val)
		X_test = scaler_x.transform(X_test)

	if scaler_y is not None:
		y_train = scaler_y.fit_transform(y_train)
		y_val = scaler_y.transform(y_val)
		y_test = scaler_y.transform(y_test)

	# Convert to PyTorch tensors
	X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
	X_val = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
	X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)

	# Split y into function values and gradients
	y_train_values = torch.tensor(y_train[:, 0], dtype=torch.float32)
	y_train_grads = torch.tensor(y_train[:, 1:], dtype=torch.float32)
	y_val_values = torch.tensor(y_val[:, 0], dtype=torch.float32)
	y_val_grads = torch.tensor(y_val[:, 1:], dtype=torch.float32)
	y_test_values = torch.tensor(y_test[:, 0], dtype=torch.float32)
	y_test_grads = torch.tensor(y_test[:, 1:], dtype=torch.float32)

	# Create datasets
	train_dataset = TensorDataset(X_train, y_train_values, y_train_grads)
	val_dataset = TensorDataset(X_val, y_val_values, y_val_grads)
	test_dataset = TensorDataset(X_test, y_test_values, y_test_grads)

	# Create dataloaders
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

	return train_dataloader, val_dataloader, test_dataloader, scaler_x, scaler_y