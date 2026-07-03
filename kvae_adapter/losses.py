from __future__ import annotations

import torch
from torch.nn import functional as F


def temporal_delta_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] < 2 or target.shape[2] < 2:
        return pred.new_zeros(())
    return F.l1_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def highpass2d(x: torch.Tensor) -> torch.Tensor:
    blurred = F.avg_pool3d(x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    return x - blurred


def highpass_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(highpass2d(pred), highpass2d(target))
