from __future__ import annotations

import math
from typing import Iterable

import torch

from .losses import laplacian_highpass2d


DEFAULT_METRIC_LOG_KEYS = (
    "phase_shimmer_score",
    "boundary_3_to_0_error",
    "fade_strobe_score",
    "temporal_jitter_score",
    "detail_loss_score",
    "detail_retention_ratio",
    "grid_artifact_score",
    "color_mean_l1",
)


def _zeros_like_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((x.shape[0],), device=x.device, dtype=torch.float32)


def _per_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).mean(dim=1)


def _masked_per_sample_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0 or not bool(mask.any().item()):
        return _zeros_like_batch(x)
    return x[:, :, mask].float().flatten(1).mean(dim=1)


def _psnr_from_mse(mse: torch.Tensor, *, max_value: float = 2.0) -> torch.Tensor:
    mse = mse.float().clamp_min(1e-12)
    return (20.0 * math.log10(max_value)) - (10.0 * torch.log10(mse))


def _masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0 or not bool(mask.any().item()):
        return _zeros_like_batch(pred)
    diff = pred[:, :, mask].float() - target[:, :, mask].float()
    return _psnr_from_mse(diff.square().flatten(1).mean(dim=1))


def _temporal_delta_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] < 2:
        return _zeros_like_batch(pred)
    diff = (pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])
    return _per_sample_mean(diff.abs())


def _temporal_second_difference_l1(x: torch.Tensor) -> torch.Tensor:
    if x.shape[2] < 3:
        return _zeros_like_batch(x)
    diff2 = x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]
    return _per_sample_mean(diff2.abs())


def _boundary_delta_l1(pred: torch.Tensor, target: torch.Tensor, *, phase_period: int) -> torch.Tensor:
    if pred.shape[2] < 2:
        return _zeros_like_batch(pred)
    frame_idx = torch.arange(1, pred.shape[2], device=pred.device)
    mask = frame_idx.remainder(phase_period) == 0
    if not bool(mask.any().item()):
        return _zeros_like_batch(pred)
    pred_delta = pred[:, :, 1:] - pred[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    return _masked_per_sample_mean((pred_delta - target_delta).abs(), mask)


def _phase_means(x_abs: torch.Tensor, *, phase_period: int) -> torch.Tensor:
    frame_idx = torch.arange(x_abs.shape[2], device=x_abs.device)
    values = []
    for phase in range(phase_period):
        mask = frame_idx.remainder(phase_period) == phase
        values.append(_masked_per_sample_mean(x_abs, mask))
    return torch.stack(values, dim=1)


def _periodic_axis_imbalance(x_abs: torch.Tensor, *, period: int, axis: str) -> torch.Tensor:
    values = []
    for offset in range(period):
        if axis == "y":
            part = x_abs[..., offset::period, :]
        elif axis == "x":
            part = x_abs[..., :, offset::period]
        else:
            raise ValueError(f"axis must be x or y, got {axis!r}")
        values.append(_per_sample_mean(part))
    stacked = torch.stack(values, dim=1)
    mean = stacked.mean(dim=1).clamp_min(1e-6)
    return stacked.std(dim=1, unbiased=False) / mean


def _grid_artifact_score(err_hp_abs: torch.Tensor, periods: Iterable[int]) -> torch.Tensor:
    scores = []
    _, _, _, h, w = err_hp_abs.shape
    for period in periods:
        if period <= 1 or period > h or period > w:
            continue
        scores.append(_periodic_axis_imbalance(err_hp_abs, period=period, axis="y"))
        scores.append(_periodic_axis_imbalance(err_hp_abs, period=period, axis="x"))
    if not scores:
        return _zeros_like_batch(err_hp_abs)
    return torch.stack(scores, dim=1).amax(dim=1)


def decoded_metric_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    phase_period: int = 4,
    grid_periods: tuple[int, ...] = (2, 4, 8),
) -> dict[str, torch.Tensor]:
    """Return no-flow decoded video metrics as per-sample tensors.

    Inputs are expected to be decoded RGB videos shaped [B,C,T,H,W] in [-1,1].
    The metrics are detached by convention at call sites when used for logging
    or hard replay, but this function itself does not alter autograd state.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {tuple(pred.shape)} != {tuple(target.shape)}")
    if pred.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(pred.shape)}")
    if phase_period <= 0:
        raise ValueError("phase_period must be positive")

    pred_f = pred.float()
    target_f = target.float()
    err = pred_f - target_f
    err_abs = err.abs()
    mse = err.square().flatten(1).mean(dim=1)
    frame_idx = torch.arange(pred.shape[2], device=pred.device)
    key_mask = frame_idx.remainder(phase_period) == 0
    inbetween_mask = ~key_mask

    out: dict[str, torch.Tensor] = {
        "l1_m11": _per_sample_mean(err_abs),
        "psnr_m11": _psnr_from_mse(mse),
        "l1_key_m11": _masked_per_sample_mean(err_abs, key_mask),
        "l1_inbetween_m11": _masked_per_sample_mean(err_abs, inbetween_mask),
        "psnr_key_m11": _masked_psnr(pred_f, target_f, key_mask),
        "psnr_inbetween_m11": _masked_psnr(pred_f, target_f, inbetween_mask),
        "temporal_delta_l1": _temporal_delta_l1(pred_f, target_f),
        "boundary_3_to_0_error": _boundary_delta_l1(pred_f, target_f, phase_period=phase_period),
    }
    out["inbetween_over_anchor"] = out["l1_inbetween_m11"] / out["l1_key_m11"].clamp_min(1e-6)

    for phase in range(phase_period):
        mask = frame_idx.remainder(phase_period) == phase
        out[f"l1_phase{phase}_m11"] = _masked_per_sample_mean(err_abs, mask)
        out[f"psnr_phase{phase}_m11"] = _masked_psnr(pred_f, target_f, mask)

    pred_hp = laplacian_highpass2d(pred_f)
    target_hp = laplacian_highpass2d(target_f)
    err_hp = pred_hp - target_hp
    err_hp_abs = err_hp.abs()
    out["highpass_l1_m11"] = _per_sample_mean(err_hp_abs)

    phase_hp = _phase_means(err_hp_abs, phase_period=phase_period)
    hp_phase_mean = phase_hp.mean(dim=1).clamp_min(1e-6)
    out["phase_highpass_imbalance"] = phase_hp.std(dim=1, unbiased=False) / hp_phase_mean
    for phase in range(phase_period):
        out[f"highpass_l1_phase{phase}_m11"] = phase_hp[:, phase]

    if pred.shape[2] < 2:
        highpass_temporal = _zeros_like_batch(pred_f)
    else:
        highpass_temporal = _per_sample_mean((err_hp[:, :, 1:] - err_hp[:, :, :-1]).abs())
    out["highpass_temporal_delta_l1"] = highpass_temporal

    out["temporal_jitter_score"] = _temporal_second_difference_l1(err)
    color_pred = pred_f.mean(dim=(-1, -2))
    color_target = target_f.mean(dim=(-1, -2))
    color_err = color_pred - color_target
    out["color_mean_l1"] = color_err.abs().flatten(1).mean(dim=1)
    if pred.shape[2] < 2:
        out["color_delta_l1"] = _zeros_like_batch(pred_f)
    else:
        color_delta_err = (color_pred[:, :, 1:] - color_pred[:, :, :-1]) - (
            color_target[:, :, 1:] - color_target[:, :, :-1]
        )
        out["color_delta_l1"] = color_delta_err.abs().flatten(1).mean(dim=1)
    if pred.shape[2] < 3:
        out["fade_strobe_score"] = _zeros_like_batch(pred_f)
    else:
        color_diff2 = color_err[:, :, 2:] - 2.0 * color_err[:, :, 1:-1] + color_err[:, :, :-2]
        out["fade_strobe_score"] = color_diff2.abs().flatten(1).mean(dim=1)

    pred_detail = _per_sample_mean(pred_hp.abs())
    target_detail = _per_sample_mean(target_hp.abs()).clamp_min(1e-6)
    detail_ratio = pred_detail / target_detail
    out["pred_detail_energy"] = pred_detail
    out["target_detail_energy"] = target_detail
    out["detail_retention_ratio"] = detail_ratio
    out["detail_loss_score"] = (1.0 - detail_ratio).clamp_min(0.0)
    out["detail_excess_score"] = (detail_ratio - 1.0).clamp_min(0.0)
    out["grid_artifact_score"] = _grid_artifact_score(err_hp_abs, grid_periods)

    out["phase_shimmer_score"] = (
        out["phase_highpass_imbalance"] + out["highpass_temporal_delta_l1"] + out["boundary_3_to_0_error"]
    )
    return out


def metric_tensors_to_float_dict(metrics: dict[str, torch.Tensor], *, index: int | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        selected = value if index is None else value[index]
        out[key] = float(torch.nan_to_num(selected.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).mean().item())
    return out


def decoded_metrics_for_json(pred: torch.Tensor, target: torch.Tensor, *, phase_period: int = 4) -> dict[str, float]:
    metrics = decoded_metric_tensors(pred, target, phase_period=phase_period)
    return metric_tensors_to_float_dict(metrics)


def decoded_hard_replay_score(
    decoded_loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    *,
    metric_weight: float = 1.0,
    phase_weight: float = 1.0,
    fade_weight: float = 0.75,
    detail_weight: float = 1.0,
    grid_weight: float = 0.5,
    color_weight: float = 0.25,
) -> torch.Tensor:
    score = decoded_loss.detach().float()
    if metric_weight <= 0 or not metrics:
        return score
    parts = [
        phase_weight * metrics.get("phase_shimmer_score", score.new_zeros(score.shape)),
        fade_weight * metrics.get("fade_strobe_score", score.new_zeros(score.shape)),
        detail_weight * metrics.get("detail_loss_score", score.new_zeros(score.shape)),
        grid_weight * metrics.get("grid_artifact_score", score.new_zeros(score.shape)),
        color_weight * metrics.get("color_mean_l1", score.new_zeros(score.shape)),
    ]
    metric_part = torch.stack([torch.nan_to_num(p.detach().float()) for p in parts], dim=0).sum(dim=0)
    return score + float(metric_weight) * metric_part
