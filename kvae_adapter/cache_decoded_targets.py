from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .decoded_cache import save_decoded_shard, write_decoded_manifest
from .io_utils import fadvise_dontneed
from .kvae2_loader import load_kvae2_t4s8
from .packed_cache import PackedLatentCache


def select_indices(*, total: int, count: int, mode: str, seed: int) -> torch.Tensor:
    if total <= 0:
        raise ValueError("source cache is empty")
    if mode == "all" or count >= total:
        return torch.arange(total, dtype=torch.long)
    if count <= 0:
        raise ValueError("count must be positive")
    if mode == "first":
        return torch.arange(count, dtype=torch.long)
    if mode == "random":
        rng = random.Random(seed)
        values = list(range(total))
        rng.shuffle(values)
        return torch.tensor(sorted(values[:count]), dtype=torch.long)
    if mode == "stride":
        if count == 1:
            return torch.tensor([total // 2], dtype=torch.long)
        values = [round(i * (total - 1) / (count - 1)) for i in range(count)]
        return torch.tensor(values, dtype=torch.long)
    raise ValueError(f"unknown index mode: {mode}")


def expected_video_shape(z: torch.Tensor) -> tuple[int, int, int]:
    _, _, t, h, w = z.shape
    return 1 + 4 * (t - 1), h * 8, w * 8


def fadvise_source_shards(cache_dir: Path) -> None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    for shard in manifest.get("shards", []):
        path = cache_dir / shard.get("path", "")
        fadvise_dontneed(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache KVAE-3D-2.0 decoded recon targets from a packed latent cache.")
    ap.add_argument("--source-cache-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_targets_strat8192"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_decoded_recon2048"))
    ap.add_argument("--kvae", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--count", type=int, default=2048)
    ap.add_argument("--index-mode", default="stride", choices=["stride", "random", "first", "all"])
    ap.add_argument("--decode-batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--source-gpu-resident", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if args.decode_batch_size <= 0:
        raise ValueError("--decode-batch-size must be positive")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.set_float32_matmul_precision("high")

    amp_dtype = None if args.amp == "none" else {"fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    cache_dtype = torch.float32 if amp_dtype is None else amp_dtype
    cache = PackedLatentCache.load(
        args.source_cache_dir,
        device=device,
        dtype=cache_dtype,
        gpu_resident=args.source_gpu_resident,
    )
    fadvise_source_shards(args.source_cache_dir)
    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=cache_dtype)

    indices = select_indices(total=cache.count, count=args.count, mode=args.index_mode, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    shard_idx = 0
    shard_decoded: list[torch.Tensor] = []
    shard_source: list[torch.Tensor] = []
    shard_count = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_count, shard_decoded, shard_source
        if shard_count == 0:
            return
        shards.append(
            save_decoded_shard(
                out_dir=args.out_dir,
                shard_idx=shard_idx,
                decoded=torch.cat(shard_decoded, dim=0),
                source_index=torch.cat(shard_source, dim=0),
            )
        )
        shard_idx += 1
        shard_count = 0
        shard_decoded = []
        shard_source = []

    pbar = tqdm(range(0, int(indices.numel()), args.decode_batch_size), desc="cache decoded", dynamic_ncols=True)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for start in pbar:
        source_idx = indices[start : start + args.decode_batch_size].to(device=device)
        batch = cache.batch(source_idx, device=device, dtype=cache_dtype)
        z_target = batch["target"]
        frames, height, width = expected_video_shape(z_target)
        with torch.no_grad(), torch.autocast(
            device_type=autocast_device,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            decoded = kvae.decode(z_target)
        decoded = decoded[:, :, :frames, :height, :width].detach().cpu().to(torch.float16)
        shard_decoded.append(decoded)
        shard_source.append(source_idx.detach().cpu().to(torch.long))
        shard_count += int(decoded.shape[0])
        if shard_count >= args.shard_size:
            flush_shard()

    flush_shard()
    write_decoded_manifest(
        out_dir=args.out_dir,
        args=vars(args),
        shards=shards,
        item_count=int(indices.numel()),
        source_cache_count=cache.count,
    )
    summary = {
        "decoded_count": int(indices.numel()),
        "source_cache_count": cache.count,
        "out_dir": str(args.out_dir),
        "shards": len(shards),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
