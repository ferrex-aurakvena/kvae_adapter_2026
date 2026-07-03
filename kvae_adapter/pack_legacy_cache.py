from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

from .io_utils import fadvise_dontneed
from .packed_cache import save_packed_shard, write_manifest


def read_legacy_item(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta_raw = f.metadata() or {}
        z_k5 = f.get_tensor("z_k5_unscaled")
        z_kvae = f.get_tensor("z_kvae_t4s8")
    fadvise_dontneed(path)
    meta = {
        "t0": int(meta_raw.get("t0", 0)),
        "y0": int(meta_raw.get("y0", 0)),
        "x0": int(meta_raw.get("x0", 0)),
        "total_t": int(meta_raw.get("total_t", z_k5.shape[1])),
        "total_h": int(meta_raw.get("total_h", z_k5.shape[2])),
        "total_w": int(meta_raw.get("total_w", z_k5.shape[3])),
    }
    return z_k5, z_kvae, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert per-crop safetensors cache into packed shards.")
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard-size", type=int, default=1024)
    args = ap.parse_args()

    files = sorted(p for p in args.in_dir.glob("*.safetensors") if p.name.startswith("crop_"))
    if not files:
        raise RuntimeError(f"No legacy crop_*.safetensors files found under {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    item_count = 0
    shard_idx = 0
    for start in tqdm(range(0, len(files), args.shard_size), desc="pack legacy", dynamic_ncols=True):
        chunk = files[start : start + args.shard_size]
        z_k5_list = []
        z_kvae_list = []
        meta_lists = {k: [] for k in ("t0", "y0", "x0", "total_t", "total_h", "total_w")}
        for path in chunk:
            z_k5, z_kvae, meta = read_legacy_item(path)
            z_k5_list.append(z_k5)
            z_kvae_list.append(z_kvae)
            for key in meta_lists:
                meta_lists[key].append(meta[key])
        meta_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in meta_lists.items()}
        shards.append(
            save_packed_shard(
                out_dir=args.out_dir,
                shard_idx=shard_idx,
                z_k5=torch.stack(z_k5_list, dim=0),
                z_kvae=torch.stack(z_kvae_list, dim=0),
                crop_meta=meta_tensors,
            )
        )
        shard_idx += 1
        item_count += len(chunk)
    write_manifest(out_dir=args.out_dir, args=vars(args), shards=shards, item_count=item_count)
    with (args.out_dir / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"in_dir": str(args.in_dir), "out_dir": str(args.out_dir), "count": item_count}, f, indent=2)
    print(f"packed {item_count} legacy crops into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
