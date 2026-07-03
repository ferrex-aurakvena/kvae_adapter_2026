from __future__ import annotations

import gzip
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import zarr


DEFAULT_SCALING_FACTOR = 0.476986


@dataclass(frozen=True)
class DiTSample:
    latent_path: Path
    teacher_zarr_path: Path
    meta_path: Path | None
    stem: str


@dataclass(frozen=True)
class CropInfo:
    stem: str
    t0: int
    y0: int
    x0: int
    total_t: int
    total_h: int
    total_w: int


def strip_latent_name(path: Path) -> str:
    name = path.name
    for suffix in (".pt.gz", ".pth.gz", ".pt", ".pth", ".safetensors"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def strip_zarr_name(path: Path) -> str:
    name = path.name
    for suffix in (".frames.zarr.zip", ".frames.zarr", ".zarr.zip", ".zarr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_pt_any(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=False)


def maybe_squeeze_batch(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 5 and z.shape[0] == 1:
        return z[0]
    return z


def latent_std(z: torch.Tensor) -> float:
    sample = z.detach().float()
    if sample.numel() > 2_000_000:
        sample = sample.flatten()[:: max(1, sample.numel() // 2_000_000)]
    return float(sample.std(unbiased=False).item())


def unscale_latents_if_needed(
    z: torch.Tensor,
    *,
    latent_space: str,
    scaling_factor: float,
    auto_std_threshold: float = 1.25,
) -> torch.Tensor:
    if latent_space not in {"auto", "scaled", "unscaled"}:
        raise ValueError(f"latent_space must be auto|scaled|unscaled, got {latent_space!r}")
    if latent_space == "unscaled":
        return z
    if latent_space == "scaled":
        return z / float(scaling_factor)
    return z / float(scaling_factor) if latent_std(z) < auto_std_threshold else z


def read_meta(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dit_samples(data_root: Path) -> list[DiTSample]:
    latent_root = data_root
    hvae_root = data_root / "hvae_decode"
    latent_files = sorted(
        p
        for p in latent_root.iterdir()
        if p.is_file() and (p.name.endswith(".pt") or p.name.endswith(".pt.gz") or p.name.endswith(".pth.gz"))
    )
    zarr_files = sorted(
        [p for p in hvae_root.rglob("*.zarr.zip") if p.is_file()]
        + [p for p in hvae_root.rglob("*.zarr") if p.is_dir()]
    )
    zarr_by_stem = {strip_zarr_name(p): p for p in zarr_files}
    samples: list[DiTSample] = []
    for latent_path in latent_files:
        stem = strip_latent_name(latent_path)
        zarr_path = zarr_by_stem.get(stem)
        if zarr_path is None:
            continue
        meta_path = latent_path.with_name(latent_path.name + ".meta.json")
        if not meta_path.exists():
            meta_path = None
        samples.append(DiTSample(latent_path=latent_path, teacher_zarr_path=zarr_path, meta_path=meta_path, stem=stem))
    if not samples:
        raise RuntimeError(f"No matched K5 latent/HVAE zarr samples found under {data_root}")
    return samples


class TensorLRU:
    def __init__(self, max_items: int = 2) -> None:
        self.max_items = max(0, int(max_items))
        self._cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

    def get(self, path: Path) -> torch.Tensor | None:
        if self.max_items <= 0 or path not in self._cache:
            return None
        value = self._cache.pop(path)
        self._cache[path] = value
        return value

    def put(self, path: Path, tensor: torch.Tensor) -> None:
        if self.max_items <= 0:
            return
        self._cache[path] = tensor
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


class ZarrZipLRU:
    def __init__(self, max_items: int = 4) -> None:
        self.max_items = max(1, int(max_items))
        self._cache: OrderedDict[Path, tuple[Any, Any]] = OrderedDict()

    def get_array(self, path: Path) -> Any:
        path = Path(path)
        if path in self._cache:
            store, arr = self._cache.pop(path)
            self._cache[path] = (store, arr)
            return arr
        if path.is_dir():
            store = zarr.DirectoryStore(str(path))
        else:
            store = zarr.ZipStore(str(path), mode="r")
        root = zarr.open(store, mode="r")
        arr = root
        if hasattr(root, "array_keys"):
            keys = list(root.array_keys())
            if keys:
                arr = root[keys[0]]
        self._cache[path] = (store, arr)
        while len(self._cache) > self.max_items:
            _, (old_store, _) = self._cache.popitem(last=False)
            try:
                old_store.close()
            except Exception:
                pass
        return arr

    def close(self) -> None:
        for store, _ in self._cache.values():
            try:
                store.close()
            except Exception:
                pass
        self._cache.clear()


def zarr_read_crop_to_torch_m11(
    arr: Any,
    *,
    t0: int,
    t1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if len(arr.shape) == 5:
        np_crop = np.asarray(arr[0, :, t0:t1, y0:y1, x0:x1])
    elif len(arr.shape) == 4:
        np_crop = np.asarray(arr[:, t0:t1, y0:y1, x0:x1])
    else:
        raise RuntimeError(f"Unexpected zarr shape: {arr.shape}")
    if np_crop.dtype == np.uint16:
        u32 = np_crop.astype(np.uint32) << 16
        np_crop = u32.view(np.float32)
    return torch.from_numpy(np_crop).to(dtype=dtype)


def pad_video_crop(x: torch.Tensor, *, frames: int, height: int, width: int) -> torch.Tensor:
    _, t, h, w = x.shape
    if t < frames:
        x = torch.cat([x, x[:, -1:].repeat(1, frames - t, 1, 1)], dim=1)
    if h < height or w < width:
        pad_w = max(0, width - w)
        pad_h = max(0, height - h)
        x = torch.nn.functional.pad(x.float(), (0, pad_w, 0, pad_h), mode="replicate").to(dtype=x.dtype)
    return x[:, :frames, :height, :width]


class K5CropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Iterable[DiTSample],
        *,
        crop_px: int = 128,
        frames: int = 33,
        latent_space: str = "auto",
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
        auto_std_threshold: float = 1.25,
        latent_cache_items: int = 2,
        zarr_cache_items: int = 4,
        latent_dtype: torch.dtype = torch.float16,
        teacher_dtype: torch.dtype = torch.float16,
    ) -> None:
        if crop_px % 8 != 0:
            raise ValueError("crop_px must be divisible by 8 for t4s8 crops")
        if (frames - 1) % 4 != 0:
            raise ValueError("frames must satisfy frames = 1 + 4 * n, e.g. 33")
        self.samples = list(samples)
        self.crop_px = int(crop_px)
        self.frames = int(frames)
        self.crop_lat = self.crop_px // 8
        self.t_lat = 1 + (self.frames - 1) // 4
        self.latent_space = latent_space
        self.scaling_factor = float(scaling_factor)
        self.auto_std_threshold = float(auto_std_threshold)
        self.latent_dtype = latent_dtype
        self.teacher_dtype = teacher_dtype
        self.latent_cache = TensorLRU(latent_cache_items)
        self.zarr_cache = ZarrZipLRU(zarr_cache_items)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_latent(self, sample: DiTSample) -> torch.Tensor:
        cached = self.latent_cache.get(sample.latent_path)
        if cached is not None:
            return cached
        z = load_pt_any(sample.latent_path, map_location="cpu")
        if isinstance(z, dict):
            for key in ("z_unscaled", "z", "latents"):
                if key in z:
                    z = z[key]
                    break
            if isinstance(z, dict):
                z = next(v for v in z.values() if isinstance(v, torch.Tensor))
        if not isinstance(z, torch.Tensor):
            raise RuntimeError(f"Latent file did not contain a tensor: {sample.latent_path}")
        z = maybe_squeeze_batch(z).contiguous().float()
        z = unscale_latents_if_needed(
            z,
            latent_space=self.latent_space,
            scaling_factor=self.scaling_factor,
            auto_std_threshold=self.auto_std_threshold,
        )
        self.latent_cache.put(sample.latent_path, z)
        return z

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, CropInfo]:
        sample = self.samples[idx]
        z = self._load_latent(sample)
        if z.ndim != 4:
            raise RuntimeError(f"Expected latent [C,T,H,W], got {tuple(z.shape)} from {sample.latent_path}")
        c, total_t, total_h, total_w = z.shape
        if c != 16:
            raise RuntimeError(f"Expected 16 latent channels, got {c} from {sample.latent_path}")

        crop_lat_h = min(self.crop_lat, total_h)
        crop_lat_w = min(self.crop_lat, total_w)
        t_lat = min(self.t_lat, total_t)
        max_t0 = max(0, total_t - t_lat)
        max_y0 = max(0, total_h - crop_lat_h)
        max_x0 = max(0, total_w - crop_lat_w)
        t0 = 0 if max_t0 == 0 else random.randint(0, max_t0)
        y0 = 0 if max_y0 == 0 else random.randint(0, max_y0)
        x0 = 0 if max_x0 == 0 else random.randint(0, max_x0)

        z_crop = z[:, t0 : t0 + t_lat, y0 : y0 + crop_lat_h, x0 : x0 + crop_lat_w].to(dtype=self.latent_dtype)
        arr = self.zarr_cache.get_array(sample.teacher_zarr_path)
        if len(arr.shape) == 5:
            _, _, total_px_t, total_px_h, total_px_w = arr.shape
        else:
            _, total_px_t, total_px_h, total_px_w = arr.shape

        px_t0 = t0 * 4
        px_y0 = y0 * 8
        px_x0 = x0 * 8
        x_crop = zarr_read_crop_to_torch_m11(
            arr,
            t0=px_t0,
            t1=min(total_px_t, px_t0 + self.frames),
            y0=px_y0,
            y1=min(total_px_h, px_y0 + self.crop_px),
            x0=px_x0,
            x1=min(total_px_w, px_x0 + self.crop_px),
            dtype=self.teacher_dtype,
        )
        x_crop = pad_video_crop(x_crop, frames=self.frames, height=self.crop_px, width=self.crop_px)
        info = CropInfo(stem=sample.stem, t0=t0, y0=y0, x0=x0, total_t=total_t, total_h=total_h, total_w=total_w)
        return z_crop, x_crop, info


def collate_crops(batch: list[tuple[torch.Tensor, torch.Tensor, CropInfo]]) -> dict[str, Any]:
    z, x, infos = zip(*batch)
    return {
        "z": torch.stack(list(z), dim=0),
        "x": torch.stack(list(x), dim=0),
        "stem": [i.stem for i in infos],
        "t0": torch.tensor([i.t0 for i in infos], dtype=torch.long),
        "y0": torch.tensor([i.y0 for i in infos], dtype=torch.long),
        "x0": torch.tensor([i.x0 for i in infos], dtype=torch.long),
        "total_t": torch.tensor([i.total_t for i in infos], dtype=torch.long),
        "total_h": torch.tensor([i.total_h for i in infos], dtype=torch.long),
        "total_w": torch.tensor([i.total_w for i in infos], dtype=torch.long),
    }
