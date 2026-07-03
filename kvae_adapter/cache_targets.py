from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from .data import K5CropDataset, build_dit_samples
from .io_utils import fadvise_dontneed
from .kvae2_loader import load_kvae2_t4s8
from .packed_cache import save_packed_shard, write_manifest
from .sampling import build_stratified_specs


def _stack_meta(infos) -> dict[str, torch.Tensor]:
    return {
        "t0": torch.tensor([i.t0 for i in infos], dtype=torch.long),
        "y0": torch.tensor([i.y0 for i in infos], dtype=torch.long),
        "x0": torch.tensor([i.x0 for i in infos], dtype=torch.long),
        "total_t": torch.tensor([i.total_t for i in infos], dtype=torch.long),
        "total_h": torch.tensor([i.total_h for i in infos], dtype=torch.long),
        "total_w": torch.tensor([i.total_w for i in infos], dtype=torch.long),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache K5 crop and KVAE-3D-2.0 t4s8 target latent pairs.")
    ap.add_argument("--data-root", type=Path, default=Path("datasets/DiT_latents"))
    ap.add_argument("--kvae", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_targets"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--count", type=int, default=256)
    ap.add_argument("--crop-px", type=int, default=128)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--latent-space", default="auto", choices=["auto", "scaled", "unscaled"])
    ap.add_argument("--format", default="packed", choices=["packed", "files"])
    ap.add_argument("--sample-mode", default="stratified", choices=["stratified", "random"])
    ap.add_argument("--cache-batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=1024)
    ap.add_argument("--t-stride-lat", type=int, default=4)
    ap.add_argument("--y-stride-lat", type=int, default=8)
    ap.add_argument("--x-stride-lat", type=int, default=8)
    ap.add_argument("--latent-cache-items", type=int, default=1)
    ap.add_argument("--zarr-cache-items", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    samples = build_dit_samples(args.data_root)
    dataset = K5CropDataset(
        samples,
        crop_px=args.crop_px,
        frames=args.frames,
        latent_space=args.latent_space,
        latent_cache_items=args.latent_cache_items,
        zarr_cache_items=args.zarr_cache_items,
    )
    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=dtype)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    shard_idx = 0
    shard_count = 0
    shard_z_k5: list[torch.Tensor] = []
    shard_z_kvae: list[torch.Tensor] = []
    shard_meta: dict[str, list[torch.Tensor]] = {k: [] for k in ("t0", "y0", "x0", "total_t", "total_h", "total_w")}
    packed_shards = []

    def flush_shard() -> None:
        nonlocal shard_idx, shard_count, shard_z_k5, shard_z_kvae, shard_meta
        if shard_count == 0:
            return
        meta = {k: torch.cat(v, dim=0) for k, v in shard_meta.items()}
        packed_shards.append(
            save_packed_shard(
                out_dir=args.out_dir,
                shard_idx=shard_idx,
                z_k5=torch.cat(shard_z_k5, dim=0),
                z_kvae=torch.cat(shard_z_kvae, dim=0),
                crop_meta=meta,
            )
        )
        shard_idx += 1
        shard_count = 0
        shard_z_k5 = []
        shard_z_kvae = []
        shard_meta = {k: [] for k in ("t0", "y0", "x0", "total_t", "total_h", "total_w")}

    specs = None
    if args.sample_mode == "stratified":
        specs = build_stratified_specs(
            samples,
            count=args.count,
            t_lat=dataset.t_lat,
            crop_lat=dataset.crop_lat,
            t_stride=args.t_stride_lat,
            y_stride=args.y_stride_lat,
            x_stride=args.x_stride_lat,
            seed=args.seed,
        )
    pbar = tqdm(total=args.count, desc="cache targets", dynamic_ncols=True)
    written = 0
    while written < args.count:
        batch_n = min(args.cache_batch_size, args.count - written)
        if specs is None:
            crops = [dataset[random.randrange(len(dataset))] for _ in range(batch_n)]
        else:
            crops = [
                dataset.get_fixed(spec.sample_idx, t0=spec.t0, y0=spec.y0, x0=spec.x0)
                for spec in specs[written : written + batch_n]
            ]
        z_k5_list, x_list, infos = zip(*crops)
        x_in = torch.stack(list(x_list), dim=0).to(device=device, dtype=dtype)
        with torch.no_grad():
            z_kvae_batch = kvae.encode(x_in).detach().cpu().to(torch.float16)
        z_k5_batch = torch.stack(list(z_k5_list), dim=0).detach().cpu().to(torch.float16)
        meta_batch = _stack_meta(infos)

        if args.format == "files":
            for j, info in enumerate(infos):
                item_idx = written + j
                tensors = {
                    "z_k5_unscaled": z_k5_batch[j],
                    "z_kvae_t4s8": z_kvae_batch[j],
                }
                name = f"crop_{item_idx:06d}_t{info.t0:03d}_y{info.y0:03d}_x{info.x0:03d}.safetensors"
                save_file(
                    tensors,
                    str(args.out_dir / name),
                    metadata={
                        "stem": info.stem,
                        "t0": str(info.t0),
                        "y0": str(info.y0),
                        "x0": str(info.x0),
                        "total_t": str(info.total_t),
                        "total_h": str(info.total_h),
                        "total_w": str(info.total_w),
                        "frames": str(args.frames),
                        "crop_px": str(args.crop_px),
                    },
                )
                fadvise_dontneed(args.out_dir / name, sync=True)
                manifest.append(
                    {
                        "path": name,
                        "stem": info.stem,
                        "t0": info.t0,
                        "y0": info.y0,
                        "x0": info.x0,
                        "total_t": info.total_t,
                        "total_h": info.total_h,
                        "total_w": info.total_w,
                        "z_k5_shape": list(z_k5_batch[j].shape),
                        "z_kvae_shape": list(z_kvae_batch[j].shape),
                    }
                )
        else:
            shard_z_k5.append(z_k5_batch)
            shard_z_kvae.append(z_kvae_batch)
            for key in shard_meta:
                shard_meta[key].append(meta_batch[key])
            shard_count += batch_n
            if shard_count >= args.shard_size:
                flush_shard()

        written += batch_n
        pbar.update(batch_n)
    pbar.close()

    if args.format == "packed":
        flush_shard()
        write_manifest(out_dir=args.out_dir, args=vars(args), shards=packed_shards, item_count=written)
    else:
        with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "items": manifest}, f, indent=2)
    print(f"cached {written} crops under {args.out_dir} format={args.format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
