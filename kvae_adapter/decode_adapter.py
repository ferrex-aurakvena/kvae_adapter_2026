from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
import torch

from .adapter import make_coord_grid
from .data import (
    DEFAULT_SCALING_FACTOR,
    DiTSample,
    build_dit_samples,
    latent_std,
    load_pt_any,
    maybe_squeeze_batch,
    unscale_latents_if_needed,
)
from .eval_decode import load_adapter, write_mp4
from .io_utils import fadvise_dontneed
from .kvae2_loader import load_kvae2_t4s8


@dataclass(frozen=True)
class SampleSelection:
    index: int
    label: str


@dataclass(frozen=True)
class AdapterSelection:
    label: str
    path: Path


def clean_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    label = label.strip("._-")
    return label or "item"


def parse_sample_spec(spec: str) -> SampleSelection:
    if "=" in spec:
        index_text, label = spec.split("=", 1)
    else:
        index_text, label = spec, ""
    try:
        index = int(index_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"sample must be INDEX or INDEX=LABEL, got {spec!r}") from exc
    if index < 0:
        raise argparse.ArgumentTypeError(f"sample index must be non-negative, got {index}")
    return SampleSelection(index=index, label=clean_label(label or f"sample_{index:06d}"))


def parse_adapter_spec(spec: str) -> AdapterSelection:
    if "=" in spec:
        label, path_text = spec.split("=", 1)
        path = Path(path_text)
    else:
        path = Path(spec)
        label = path.name
        if label.endswith(".safetensors"):
            label = label[: -len(".safetensors")]
    return AdapterSelection(label=clean_label(label), path=path)


def resolve_samples(
    samples: list[DiTSample],
    *,
    sample_specs: list[SampleSelection] | None,
    all_samples: bool,
    limit: int | None,
) -> list[SampleSelection]:
    if all_samples and sample_specs:
        raise ValueError("Use either --sample or --all-samples, not both")
    if all_samples:
        count = len(samples) if limit is None else min(len(samples), max(0, limit))
        return [SampleSelection(index=i, label=f"sample_{i:06d}") for i in range(count)]
    if not sample_specs:
        raise ValueError("Select at least one --sample or use --all-samples")
    resolved: list[SampleSelection] = []
    for spec in sample_specs:
        if spec.index >= len(samples):
            raise IndexError(f"sample index {spec.index} is out of range for {len(samples)} samples")
        resolved.append(spec)
    return resolved


def load_full_latent(
    sample: DiTSample,
    *,
    latent_space: str,
    scaling_factor: float,
    auto_std_threshold: float,
) -> torch.Tensor:
    z: Any = load_pt_any(sample.latent_path, map_location="cpu")
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
    if z.ndim != 4:
        raise RuntimeError(f"Expected full latent [C,T,H,W], got {tuple(z.shape)} from {sample.latent_path}")
    if z.shape[0] != 16:
        raise RuntimeError(f"Expected 16 latent channels, got {z.shape[0]} from {sample.latent_path}")
    before_std = latent_std(z)
    z = unscale_latents_if_needed(
        z,
        latent_space=latent_space,
        scaling_factor=scaling_factor,
        auto_std_threshold=auto_std_threshold,
    )
    after_std = latent_std(z)
    fadvise_dontneed(sample.latent_path)
    return z.to(dtype=torch.float16), before_std, after_std


def expected_video_shape(z_shape: tuple[int, int, int, int]) -> tuple[int, int, int]:
    _, t, h, w = z_shape
    return (1 + 4 * (t - 1), h * 8, w * 8)


def axis_weight(length: int, *, left: int, right: int) -> torch.Tensor:
    weight = torch.ones(length, dtype=torch.float32)
    if left > 0:
        weight[:left] = torch.linspace(0.0, 1.0, left, dtype=torch.float32)
    if right > 0:
        weight[-right:] = torch.linspace(1.0, 0.0, right, dtype=torch.float32)
    return weight.clamp_min_(1e-6)


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    if tile >= size:
        return [0]
    step = max(1, tile - overlap)
    count = max(2, ((size - overlap) + step - 1) // step)
    last_start = size - tile
    if count == 2:
        return [0, last_start]
    return sorted({round(i * last_start / (count - 1)) for i in range(count)})


def decode_full(
    *,
    kvae: torch.nn.Module,
    z_pred: torch.Tensor,
    z_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    with torch.no_grad():
        pred = kvae.decode(z_pred)
    frames, height, width = expected_video_shape(z_shape)
    return pred[:, :, :frames, :height, :width].detach().cpu()


def decode_tiled(
    *,
    kvae: torch.nn.Module,
    z_pred: torch.Tensor,
    z_shape: tuple[int, int, int, int],
    tile_lat_h: int,
    tile_lat_w: int,
    overlap_lat: int,
    device: torch.device,
    blend_device: str,
    empty_cache_each_tile: bool,
    cuda_memory_stats: bool,
    tile_progress_every: int,
    adapter_label: str,
    sample_label: str,
) -> torch.Tensor:
    _, _, _, latent_h, latent_w = z_pred.shape
    frames, height, width = expected_video_shape(z_shape)
    accum_device = device if blend_device == "cuda" and device.type == "cuda" else torch.device("cpu")
    accum = torch.zeros((1, 3, frames, height, width), dtype=torch.float32, device=accum_device)
    weight_sum = torch.zeros((1, 1, 1, height, width), dtype=torch.float32, device=accum_device)
    y_starts = tile_starts(latent_h, tile_lat_h, overlap_lat)
    x_starts = tile_starts(latent_w, tile_lat_w, overlap_lat)

    tile_index = 0
    tile_count = len(y_starts) * len(x_starts)
    for y0 in y_starts:
        y1 = min(latent_h, y0 + tile_lat_h)
        for x0 in x_starts:
            x1 = min(latent_w, x0 + tile_lat_w)
            tile_index += 1
            if cuda_memory_stats and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            py0, py1 = y0 * 8, y1 * 8
            px0, px1 = x0 * 8, x1 * 8
            with torch.no_grad():
                tile = kvae.decode(z_pred[:, :, :, y0:y1, x0:x1])
            tile = tile[:, :, :frames, : py1 - py0, : px1 - px0].detach().to(device=accum_device, dtype=torch.float32)
            wy = axis_weight(py1 - py0, left=0 if y0 == 0 else min(overlap_lat * 8, py1 - py0), right=0 if y1 == latent_h else min(overlap_lat * 8, py1 - py0))
            wx = axis_weight(px1 - px0, left=0 if x0 == 0 else min(overlap_lat * 8, px1 - px0), right=0 if x1 == latent_w else min(overlap_lat * 8, px1 - px0))
            weight = (wy[:, None] * wx[None, :]).view(1, 1, 1, py1 - py0, px1 - px0).to(device=accum_device)
            accum[:, :, :, py0:py1, px0:px1].add_(tile * weight)
            weight_sum[:, :, :, py0:py1, px0:px1].add_(weight)
            del tile
            cuda_stats: dict[str, float] = {}
            if cuda_memory_stats and device.type == "cuda":
                cuda_stats = {
                    "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
                    "cuda_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
                    "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                    "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
                }
            if empty_cache_each_tile and device.type == "cuda":
                torch.cuda.empty_cache()
            should_log = tile_progress_every > 0 and (
                tile_index == 1 or tile_index == tile_count or tile_index % tile_progress_every == 0
            )
            if should_log:
                print(
                    json.dumps(
                        {
                            "event": "tile_done",
                            "adapter": adapter_label,
                            "sample": sample_label,
                            "tile": tile_index,
                            "tiles": tile_count,
                            "latent_window": [y0, y1, x0, x1],
                            **cuda_stats,
                        }
                    ),
                    flush=True,
                )
    accum.div_(weight_sum.clamp_min_(1e-6))
    return accum.detach().cpu()


def predict_adapter_latent(
    *,
    adapter: torch.nn.Module,
    z_cpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = z_cpu.unsqueeze(0).to(device=device, dtype=dtype)
    _, _, t, h, w = z.shape
    coords = make_coord_grid(
        batch=1,
        t=t,
        h=h,
        w=w,
        t0=torch.tensor([0], device=device),
        y0=torch.tensor([0], device=device),
        x0=torch.tensor([0], device=device),
        total_t=t,
        total_h=h,
        total_w=w,
        device=device,
        dtype=dtype,
        feature_mode=adapter.feature_mode,
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda" if device.type == "cuda" else "cpu",
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        return adapter(z, coords)


def decode_one(
    *,
    adapter: torch.nn.Module,
    kvae: torch.nn.Module,
    z_cpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    decode_mode: str,
    tile_lat_h: int,
    tile_lat_w: int,
    tile_overlap_lat: int,
    blend_device: str,
    empty_cache_each_tile: bool,
    cuda_memory_stats: bool,
    tile_progress_every: int,
    adapter_label: str,
    sample_label: str,
) -> torch.Tensor:
    z_shape = tuple(z_cpu.shape)
    z_pred = predict_adapter_latent(adapter=adapter, z_cpu=z_cpu, device=device, dtype=dtype)
    if decode_mode == "full":
        return decode_full(kvae=kvae, z_pred=z_pred, z_shape=z_shape)
    return decode_tiled(
        kvae=kvae,
        z_pred=z_pred,
        z_shape=z_shape,
        tile_lat_h=tile_lat_h,
        tile_lat_w=tile_lat_w,
        overlap_lat=tile_overlap_lat,
        device=device,
        blend_device=blend_device,
        empty_cache_each_tile=empty_cache_each_tile,
        cuda_memory_stats=cuda_memory_stats,
        tile_progress_every=tile_progress_every,
        adapter_label=adapter_label,
        sample_label=sample_label,
    )


def inspect_mp4(path: Path) -> dict[str, Any]:
    container = av.open(str(path), mode="r")
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame_count = 0
        for _ in container.decode(stream):
            frame_count += 1
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        return {
            "width": int(stream.width),
            "height": int(stream.height),
            "frames": frame_count,
            "fps": fps,
        }
    finally:
        container.close()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode K5 Pro latents through adapter checkpoints and KVAE t4s8.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", type=Path, default=Path("datasets/DiT_latents"))
    ap.add_argument("--kvae", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/eval/adapter_decode_full_t4s8"))
    ap.add_argument("--sample", action="append", type=parse_sample_spec, default=[])
    ap.add_argument("--all-samples", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--adapter", action="append", type=parse_adapter_spec, required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--latent-space", choices=("auto", "scaled", "unscaled"), default="auto")
    ap.add_argument("--scaling-factor", type=float, default=DEFAULT_SCALING_FACTOR)
    ap.add_argument("--auto-std-threshold", type=float, default=1.25)
    ap.add_argument("--decode-mode", choices=("tiled", "full"), default="tiled")
    ap.add_argument("--tile-lat-h", type=int, default=24)
    ap.add_argument("--tile-lat-w", type=int, default=32)
    ap.add_argument("--tile-overlap-lat", type=int, default=4)
    ap.add_argument("--blend-device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--empty-cache-each-tile", action="store_true")
    ap.add_argument("--cuda-memory-stats", action="store_true")
    ap.add_argument("--tile-progress-every", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.tile_lat_h <= 0 or args.tile_lat_w <= 0:
        raise ValueError("--tile-lat-h and --tile-lat-w must be positive")
    if args.tile_overlap_lat < 0:
        raise ValueError("--tile-overlap-lat must be non-negative")
    if args.tile_overlap_lat >= min(args.tile_lat_h, args.tile_lat_w):
        raise ValueError("--tile-overlap-lat must be smaller than both tile dimensions")
    if args.tile_progress_every < 0:
        raise ValueError("--tile-progress-every must be non-negative")

    samples = build_dit_samples(args.data_root)
    sample_specs = resolve_samples(samples, sample_specs=args.sample, all_samples=args.all_samples, limit=args.limit)
    adapters: list[AdapterSelection] = list(args.adapter)
    for adapter_spec in adapters:
        if not adapter_spec.path.exists():
            raise FileNotFoundError(adapter_spec.path)

    manifest: dict[str, Any] = {
        "format": "kvae_adapter_decode_manifest_v1",
        "data_root": str(args.data_root),
        "kvae": str(args.kvae),
        "out_dir": str(args.out_dir),
        "fps": args.fps,
        "latent_space": args.latent_space,
        "decode_mode": args.decode_mode,
        "tile_lat_h": args.tile_lat_h,
        "tile_lat_w": args.tile_lat_w,
        "tile_overlap_lat": args.tile_overlap_lat,
        "blend_device": args.blend_device,
        "empty_cache_each_tile": args.empty_cache_each_tile,
        "cuda_memory_stats": args.cuda_memory_stats,
        "tile_progress_every": args.tile_progress_every,
        "samples": [asdict(spec) | {"stem": samples[spec.index].stem} for spec in sample_specs],
        "adapters": [{"label": a.label, "path": str(a.path)} for a in adapters],
        "dry_run": bool(args.dry_run),
        "outputs": [],
    }
    manifest_path = args.out_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=dtype)

    latent_cache: dict[int, tuple[torch.Tensor, float, float]] = {}
    for adapter_spec in adapters:
        adapter = load_adapter(
            adapter_spec.path,
            device=device,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
        )
        for sample_spec in sample_specs:
            sample = samples[sample_spec.index]
            out_path = args.out_dir / adapter_spec.label / sample_spec.label / "adapter_kvae_t4s8.mp4"
            if out_path.exists() and not args.overwrite:
                video_info = inspect_mp4(out_path)
                entry = {
                    "adapter": asdict(adapter_spec) | {"path": str(adapter_spec.path)},
                    "sample": asdict(sample_spec) | {"stem": sample.stem},
                    "output": str(out_path),
                    "skipped": True,
                    "file_size": out_path.stat().st_size,
                    "video": video_info,
                }
                manifest["outputs"].append(entry)
                write_manifest(manifest_path, manifest)
                print(json.dumps(entry, indent=2))
                continue

            if sample_spec.index not in latent_cache:
                latent_cache[sample_spec.index] = load_full_latent(
                    sample,
                    latent_space=args.latent_space,
                    scaling_factor=args.scaling_factor,
                    auto_std_threshold=args.auto_std_threshold,
                )
            z_cpu, latent_std_before, latent_std_after = latent_cache[sample_spec.index]
            started = time.perf_counter()
            video = decode_one(
                adapter=adapter,
                kvae=kvae,
                z_cpu=z_cpu,
                device=device,
                dtype=dtype,
                decode_mode=args.decode_mode,
                tile_lat_h=args.tile_lat_h,
                tile_lat_w=args.tile_lat_w,
                tile_overlap_lat=args.tile_overlap_lat,
                blend_device=args.blend_device,
                empty_cache_each_tile=args.empty_cache_each_tile,
                cuda_memory_stats=args.cuda_memory_stats,
                tile_progress_every=args.tile_progress_every,
                adapter_label=adapter_spec.label,
                sample_label=sample_spec.label,
            )
            decode_elapsed = time.perf_counter() - started
            write_mp4(out_path, video, fps=args.fps)
            video_info = inspect_mp4(out_path)
            entry = {
                "adapter": {"label": adapter_spec.label, "path": str(adapter_spec.path)},
                "sample": {"index": sample_spec.index, "label": sample_spec.label, "stem": sample.stem},
                "latent_shape": list(z_cpu.shape),
                "latent_std_before": latent_std_before,
                "latent_std_after": latent_std_after,
                "expected_video_shape": list(expected_video_shape(tuple(z_cpu.shape))),
                "decode_mode": args.decode_mode,
                "tile_lat_h": args.tile_lat_h,
                "tile_lat_w": args.tile_lat_w,
                "tile_overlap_lat": args.tile_overlap_lat,
                "blend_device": args.blend_device,
                "empty_cache_each_tile": args.empty_cache_each_tile,
                "cuda_memory_stats": args.cuda_memory_stats,
                "tile_progress_every": args.tile_progress_every,
                "output": str(out_path),
                "file_size": out_path.stat().st_size,
                "decode_elapsed_s": decode_elapsed,
                "video": video_info,
                "skipped": False,
            }
            manifest["outputs"].append(entry)
            write_manifest(manifest_path, manifest)
            print(json.dumps(entry, indent=2))
            del video
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
