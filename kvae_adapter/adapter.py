from __future__ import annotations

import torch
from torch import nn


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
) -> torch.Tensor:
    """Return normalized t/y/x coordinate channels shaped [B, 3, T, H, W]."""
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
    return coords.mul_(2).sub_(1)


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
        coord_channels: int = 3,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.coord_channels = coord_channels
        self.residual = residual
        self.in_proj = nn.Conv3d(latent_channels + coord_channels, hidden_channels, kernel_size=3, padding=1)
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
