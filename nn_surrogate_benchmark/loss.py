import torch
import torch.nn as nn
from typing import Callable


def default_weight_fn(x: torch.Tensor) -> torch.Tensor:
    return 1/(1+x)

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_fn: Callable[[torch.Tensor], torch.Tensor] = default_weight_fn):
        super().__init__()
        self.weight_fn = weight_fn

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.weight_fn(y_true) * torch.mean((y_pred - y_true) ** 2)
        return loss
