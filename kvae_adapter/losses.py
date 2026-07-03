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


def time_loss_weights(
    frames: int,
    *,
    key_weight: float = 0.25,
    inbetween_weight: float = 1.0,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized per-frame weights for t4 latent key/in-between frames."""
    frame_idx = torch.arange(frames, device=device)
    weights = torch.full((frames,), float(inbetween_weight), device=device, dtype=dtype)
    weights[frame_idx.remainder(4) == 0] = float(key_weight)
    return weights / weights.mean().clamp_min(1e-6)


def charbonnier_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    diff = pred.float() - target.float()
    loss = torch.sqrt(diff.square() + eps * eps) - eps
    if weight is not None:
        loss = loss * weight.to(device=loss.device, dtype=loss.dtype)
    return loss.flatten(1).mean(dim=1)


def laplacian_highpass2d(x: torch.Tensor) -> torch.Tensor:
    """Depthwise 2D Laplacian over each video frame shaped [B,C,T,H,W]."""
    b, c, t, h, w = x.shape
    x2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    kernel = x.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    y = F.conv2d(x2d, kernel, padding=1, groups=c)
    return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def highpass_charbonnier_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    return charbonnier_per_sample(
        laplacian_highpass2d(pred),
        laplacian_highpass2d(target),
        weight=weight,
        eps=eps,
    )


def flicker_error_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """High-frequency temporal error change, useful for shimmer suppression."""
    if pred.shape[2] < 2:
        return pred.new_zeros((pred.shape[0],), dtype=torch.float32)
    err_hp = laplacian_highpass2d(pred.float() - target.float())
    diff = err_hp[:, :, 1:] - err_hp[:, :, :-1]
    return diff.abs().flatten(1).mean(dim=1)


def jitter_error_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Second temporal difference of reconstruction error."""
    if pred.shape[2] < 3:
        return pred.new_zeros((pred.shape[0],), dtype=torch.float32)
    err = pred.float() - target.float()
    diff2 = err[:, :, 2:] - 2.0 * err[:, :, 1:-1] + err[:, :, :-2]
    return diff2.abs().flatten(1).mean(dim=1)
