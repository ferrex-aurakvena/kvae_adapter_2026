from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file

from .io_utils import fadvise_dontneed


PACKED_FORMAT = "kvae_adapter_packed_latents_v1"


@dataclass
class PackedShard:
    path: str
    count: int
    z_shape: list[int]


@dataclass
class PackedLatentCache:
    z_k5: torch.Tensor
    z_kvae: torch.Tensor
    t0: torch.Tensor
    y0: torch.Tensor
    x0: torch.Tensor
    total_t: torch.Tensor
    total_h: torch.Tensor
    total_w: torch.Tensor
    device: torch.device

    @property
    def count(self) -> int:
        return int(self.z_k5.shape[0])

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        gpu_resident: bool = True,
    ) -> "PackedLatentCache":
        manifest_path = cache_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != PACKED_FORMAT:
            raise RuntimeError(f"{cache_dir} is not a packed cache ({manifest.get('format')!r})")

        tensors: dict[str, list[torch.Tensor]] = {
            "z_k5_unscaled": [],
            "z_kvae_t4s8": [],
            "t0": [],
            "y0": [],
            "x0": [],
            "total_t": [],
            "total_h": [],
            "total_w": [],
        }
        for shard in manifest["shards"]:
            shard_path = cache_dir / shard["path"]
            data = load_file(str(shard_path), device="cpu")
            if not gpu_resident:
                fadvise_dontneed(shard_path)
            for key in tensors:
                tensors[key].append(data[key])

        target_device = device if gpu_resident else torch.device("cpu")
        z_k5 = torch.cat(tensors["z_k5_unscaled"], dim=0).to(device=target_device, dtype=dtype)
        z_kvae = torch.cat(tensors["z_kvae_t4s8"], dim=0).to(device=target_device, dtype=dtype)
        coords = {
            key: torch.cat(tensors[key], dim=0).to(device=target_device, dtype=torch.long)
            for key in ("t0", "y0", "x0", "total_t", "total_h", "total_w")
        }
        if target_device.type == "cpu" and device.type == "cuda":
            z_k5 = z_k5.pin_memory()
            z_kvae = z_kvae.pin_memory()
            coords = {k: v.pin_memory() for k, v in coords.items()}
        return cls(z_k5=z_k5, z_kvae=z_kvae, device=target_device, **coords)

    def batch(self, indices: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        source_device = self.z_k5.device
        if source_device == indices.device:
            gather_idx = indices
        else:
            gather_idx = indices.to(source_device)
        out = {
            "z": self.z_k5.index_select(0, gather_idx),
            "target": self.z_kvae.index_select(0, gather_idx),
            "t0": self.t0.index_select(0, gather_idx),
            "y0": self.y0.index_select(0, gather_idx),
            "x0": self.x0.index_select(0, gather_idx),
            "total_t": self.total_t.index_select(0, gather_idx),
            "total_h": self.total_h.index_select(0, gather_idx),
            "total_w": self.total_w.index_select(0, gather_idx),
        }
        if source_device != device:
            out = {k: v.to(device=device, non_blocking=True) for k, v in out.items()}
        out["z"] = out["z"].to(dtype=dtype)
        out["target"] = out["target"].to(dtype=dtype)
        return out


def save_packed_shard(
    *,
    out_dir: Path,
    shard_idx: int,
    z_k5: torch.Tensor,
    z_kvae: torch.Tensor,
    crop_meta: dict[str, torch.Tensor],
) -> PackedShard:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"shard_{shard_idx:05d}.safetensors"
    tensors = {
        "z_k5_unscaled": z_k5.detach().cpu().to(torch.float16).contiguous(),
        "z_kvae_t4s8": z_kvae.detach().cpu().to(torch.float16).contiguous(),
    }
    for key in ("t0", "y0", "x0", "total_t", "total_h", "total_w"):
        tensors[key] = crop_meta[key].detach().cpu().to(torch.long).contiguous()
    out_path = out_dir / name
    save_file(tensors, str(out_path), metadata={"format": PACKED_FORMAT})
    fadvise_dontneed(out_path, sync=True)
    return PackedShard(path=name, count=int(z_k5.shape[0]), z_shape=list(z_k5.shape[1:]))


def write_manifest(
    *,
    out_dir: Path,
    args: dict[str, Any],
    shards: Iterable[PackedShard],
    item_count: int,
) -> None:
    payload = {
        "format": PACKED_FORMAT,
        "args": {k: str(v) for k, v in args.items()},
        "total_count": int(item_count),
        "shards": [shard.__dict__ for shard in shards],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
