from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .data import DiTSample


@dataclass(frozen=True)
class CropSpec:
    sample_idx: int
    t0: int
    y0: int
    x0: int


def positions(total: int, crop: int, stride: int) -> list[int]:
    if crop >= total:
        return [0]
    stride = max(1, int(stride))
    vals = list(range(0, total - crop + 1, stride))
    last = total - crop
    if vals[-1] != last:
        vals.append(last)
    return vals


def latent_shape_from_meta(sample: DiTSample) -> tuple[int, int, int]:
    if sample.meta_path is not None and sample.meta_path.exists():
        with sample.meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        shape = meta.get("latent_shape")
        if shape and len(shape) == 5:
            return int(shape[2]), int(shape[3]), int(shape[4])
    raise RuntimeError(f"Missing latent_shape metadata for {sample.latent_path}")


def build_stratified_specs(
    samples: list[DiTSample],
    *,
    count: int,
    t_lat: int,
    crop_lat: int,
    t_stride: int,
    y_stride: int,
    x_stride: int,
    seed: int,
) -> list[CropSpec]:
    rng = random.Random(seed)
    n = len(samples)
    base = count // n
    extra = count % n
    specs: list[CropSpec] = []
    for sample_idx, sample in enumerate(samples):
        total_t, total_h, total_w = latent_shape_from_meta(sample)
        grid = [
            CropSpec(sample_idx=sample_idx, t0=t0, y0=y0, x0=x0)
            for t0 in positions(total_t, t_lat, t_stride)
            for y0 in positions(total_h, crop_lat, y_stride)
            for x0 in positions(total_w, crop_lat, x_stride)
        ]
        rng.shuffle(grid)
        quota = base + (1 if sample_idx < extra else 0)
        if quota <= len(grid):
            specs.extend(grid[:quota])
        else:
            specs.extend(grid)
            specs.extend(rng.choice(grid) for _ in range(quota - len(grid)))
    return specs
