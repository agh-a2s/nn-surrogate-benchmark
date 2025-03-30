import torch
import torch.nn as nn
from typing import Callable

def default_weight_fn(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + x)


class WeightedMSELoss(nn.Module):
    def __init__(
        self, weight_fn: Callable[[torch.Tensor], torch.Tensor] = default_weight_fn
    ):
        super().__init__()
        self.weight_fn = weight_fn

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.weight_fn(y_true) * torch.mean((y_pred - y_true) ** 2)
        return loss

class ListNetLossFunction(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        preds_smax = torch.softmax(y_pred, dim=1)
        true_smax = torch.softmax(y_true, dim=1)

        preds_smax = preds_smax + 1e-10
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


class RankCosineLossFunction(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.float()
        y_true = y_true.float()

        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True)

        y_pred_centered = y_pred - y_pred_mean
        y_true_centered = y_true - y_true_mean

        numerator = torch.sum(y_pred_centered * y_true_centered, dim=1)
        denominator = torch.sqrt(torch.sum(y_pred_centered**2, dim=1)) * torch.sqrt(
            torch.sum(y_true_centered**2, dim=1)
        )

        cosine_similarities = numerator / (denominator + 1e-8)

        cosine_distances = 1 - cosine_similarities

        return torch.mean(cosine_distances)