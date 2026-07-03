from __future__ import annotations

import math

import torch
from torch import nn

FEATURE_MODE_COORDS = "coords"
FEATURE_MODE_COORDS_PHASE_V1 = "coords_phase_v1"
FEATURE_MODE_CHOICES = (FEATURE_MODE_COORDS, FEATURE_MODE_COORDS_PHASE_V1)
FEATURE_MODE_COORD_CHANNELS = {
    FEATURE_MODE_COORDS: 3,
    FEATURE_MODE_COORDS_PHASE_V1: 10,
}


def resolve_feature_mode(value: str | None) -> str:
    mode = FEATURE_MODE_COORDS if value is None else str(value)
    if mode not in FEATURE_MODE_CHOICES:
        raise ValueError(f"unknown feature mode {mode!r}; expected one of {FEATURE_MODE_CHOICES}")
    return mode


def feature_coord_channels(feature_mode: str | None) -> int:
    return FEATURE_MODE_COORD_CHANNELS[resolve_feature_mode(feature_mode)]


def make_coord_grid(
    *,
    batch: int,
    t: int,
    h: int,
    w: int,
    t0: torch.Tensor,
    y0: torch.Tensor,
    x0: torch.Tensor,
    total_t: int,
    total_h: int,
    total_w: int,
    device: torch.device,
    dtype: torch.dtype,
    feature_mode: str = FEATURE_MODE_COORDS,
) -> torch.Tensor:
    """Return adapter conditioning channels shaped [B, C, T, H, W]."""
    feature_mode = resolve_feature_mode(feature_mode)
    tt = torch.arange(t, device=device, dtype=dtype).view(1, 1, t, 1, 1)
    yy = torch.arange(h, device=device, dtype=dtype).view(1, 1, 1, h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, 1, w)

    t0 = t0.to(device=device, dtype=dtype).view(batch, 1, 1, 1, 1)
    y0 = y0.to(device=device, dtype=dtype).view(batch, 1, 1, 1, 1)
    x0 = x0.to(device=device, dtype=dtype).view(batch, 1, 1, 1, 1)

    denom_t = max(total_t - 1, 1)
    denom_h = max(total_h - 1, 1)
    denom_w = max(total_w - 1, 1)

    coords = torch.cat(
        [
            ((t0 + tt) / denom_t).expand(batch, 1, t, h, w),
            ((y0 + yy) / denom_h).expand(batch, 1, t, h, w),
            ((x0 + xx) / denom_w).expand(batch, 1, t, h, w),
        ],
        dim=1,
    )
    coords = coords.mul_(2).sub_(1)
    if feature_mode == FEATURE_MODE_COORDS:
        return coords

    phase_idx = (
        t0.to(device=device, dtype=torch.long).view(batch, 1, 1, 1, 1)
        + torch.arange(t, device=device, dtype=torch.long).view(1, 1, t, 1, 1)
    ).remainder(4)
    phase_float = phase_idx.to(dtype=dtype)
    phase_angle = phase_float * (2.0 * math.pi / 4.0)
    sin_phase = phase_angle.sin().expand(batch, 1, t, h, w)
    cos_phase = phase_angle.cos().expand(batch, 1, t, h, w)
    one_hot = [(phase_idx == i).to(dtype=dtype).expand(batch, 1, t, h, w) for i in range(4)]
    phase_boundary = (phase_idx == 0).to(dtype=dtype).expand(batch, 1, t, h, w)
    return torch.cat([coords, sin_phase, cos_phase, *one_hot, phase_boundary], dim=1)


class ResBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class K5ToKVAEAdapter(nn.Module):
    """Small residual 3D adapter from K5/HVAE latents to KVAE t4s8 latents."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        coord_channels: int | None = None,
        feature_mode: str = FEATURE_MODE_COORDS,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        residual: bool = True,
    ) -> None:
        super().__init__()
        feature_mode = resolve_feature_mode(feature_mode)
        self.latent_channels = latent_channels
        self.feature_mode = feature_mode
        self.coord_channels = feature_coord_channels(feature_mode) if coord_channels is None else coord_channels
        self.residual = residual
        self.in_proj = nn.Conv3d(latent_channels + self.coord_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock3D(hidden_channels) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.out_act = nn.SiLU(inplace=True)
        self.out_proj = nn.Conv3d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, coords], dim=1)
        h = self.in_proj(h)
        h = self.blocks(h)
        delta = self.out_proj(self.out_act(self.out_norm(h)))
        return z + delta if self.residual else delta
