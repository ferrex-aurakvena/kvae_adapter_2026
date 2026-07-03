from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file

from .io_utils import fadvise_dontneed


DECODED_CACHE_FORMAT = "kvae_adapter_decoded_targets_v1"


@dataclass
class DecodedShard:
    path: str
    count: int
    x_shape: list[int]


@dataclass
class DecodedTargetCache:
    x: torch.Tensor
    source_index: torch.Tensor
    source_to_row: dict[int, int]
    device: torch.device

    @property
    def count(self) -> int:
        return int(self.x.shape[0])

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        gpu_resident: bool = True,
    ) -> "DecodedTargetCache":
        manifest_path = cache_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != DECODED_CACHE_FORMAT:
            raise RuntimeError(f"{cache_dir} is not a decoded target cache ({manifest.get('format')!r})")

        x_parts: list[torch.Tensor] = []
        source_parts: list[torch.Tensor] = []
        for shard in manifest["shards"]:
            shard_path = cache_dir / shard["path"]
            data = load_file(str(shard_path), device="cpu")
            x_parts.append(data["decoded_kvae_recon"])
            source_parts.append(data["source_index"].to(dtype=torch.long))
            fadvise_dontneed(shard_path)

        source_cpu = torch.cat(source_parts, dim=0).contiguous()
        source_to_row = {int(src): row for row, src in enumerate(source_cpu.tolist())}
        target_device = device if gpu_resident else torch.device("cpu")
        x = torch.cat(x_parts, dim=0).to(device=target_device, dtype=dtype).contiguous()
        source_index = source_cpu.to(device=target_device, dtype=torch.long)
        return cls(x=x, source_index=source_index, source_to_row=source_to_row, device=target_device)

    def batch_rows(self, rows: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        source_device = self.x.device
        gather_rows = rows.to(device=source_device, dtype=torch.long) if rows.device != source_device else rows
        out = {
            "x": self.x.index_select(0, gather_rows),
            "source_index": self.source_index.index_select(0, gather_rows),
        }
        if source_device != device:
            out = {k: v.to(device=device, non_blocking=True) for k, v in out.items()}
        out["x"] = out["x"].to(dtype=dtype)
        return out

    def batch_sources(
        self,
        source_indices: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        rows = []
        for src in source_indices.detach().cpu().tolist():
            try:
                rows.append(self.source_to_row[int(src)])
            except KeyError as exc:
                raise KeyError(f"source index {int(src)} is not present in decoded cache") from exc
        row_tensor = torch.tensor(rows, device=device, dtype=torch.long)
        return self.batch_rows(row_tensor, device=device, dtype=dtype)


def save_decoded_shard(
    *,
    out_dir: Path,
    shard_idx: int,
    decoded: torch.Tensor,
    source_index: torch.Tensor,
) -> DecodedShard:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"shard_{shard_idx:05d}.safetensors"
    tensors = {
        "decoded_kvae_recon": decoded.detach().cpu().to(torch.float16).contiguous(),
        "source_index": source_index.detach().cpu().to(torch.long).contiguous(),
    }
    out_path = out_dir / name
    save_file(tensors, str(out_path), metadata={"format": DECODED_CACHE_FORMAT})
    fadvise_dontneed(out_path, sync=True)
    return DecodedShard(path=name, count=int(decoded.shape[0]), x_shape=list(decoded.shape[1:]))


def write_decoded_manifest(
    *,
    out_dir: Path,
    args: dict[str, Any],
    shards: Iterable[DecodedShard],
    item_count: int,
    source_cache_count: int,
) -> None:
    payload = {
        "format": DECODED_CACHE_FORMAT,
        "args": {k: str(v) for k, v in args.items()},
        "total_count": int(item_count),
        "source_cache_count": int(source_cache_count),
        "shards": [shard.__dict__ for shard in shards],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
