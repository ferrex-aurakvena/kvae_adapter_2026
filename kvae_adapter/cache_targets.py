from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from .data import K5CropDataset, build_dit_samples
from .kvae2_loader import load_kvae2_t4s8


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
        latent_cache_items=2,
        zarr_cache_items=4,
    )
    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=dtype)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for i in tqdm(range(args.count), desc="cache targets", dynamic_ncols=True):
        idx = random.randrange(len(dataset))
        z_k5, x, info = dataset[idx]
        x_in = x.unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            z_kvae = kvae.encode(x_in).squeeze(0).detach().cpu().to(torch.float16)
        tensors = {
            "z_k5_unscaled": z_k5.detach().cpu().to(torch.float16),
            "z_kvae_t4s8": z_kvae,
        }
        name = f"crop_{i:06d}_t{info.t0:03d}_y{info.y0:03d}_x{info.x0:03d}.safetensors"
        out_path = args.out_dir / name
        save_file(
            tensors,
            str(out_path),
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
                "z_k5_shape": list(z_k5.shape),
                "z_kvae_shape": list(z_kvae.shape),
            }
        )
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "items": manifest}, f, indent=2)
    print(f"cached {len(manifest)} crops under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
