from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import av
import torch
from safetensors.torch import load_file

from .adapter import FEATURE_MODE_COORDS, K5ToKVAEAdapter, make_coord_grid, resolve_feature_mode
from .data import K5CropDataset, build_dit_samples
from .kvae2_loader import load_kvae2_t4s8
from .metrics import decoded_metrics_for_json


def adapter_config_from_sidecar(path: Path, hidden_channels: int, num_blocks: int) -> tuple[int, int, str]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        return hidden_channels, num_blocks, FEATURE_MODE_COORDS
    with sidecar.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload.get("config", {})
    return (
        int(cfg.get("hidden_channels", hidden_channels)),
        int(cfg.get("num_blocks", num_blocks)),
        resolve_feature_mode(cfg.get("feature_mode", FEATURE_MODE_COORDS)),
    )


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(k.startswith("_orig_mod.") for k in state):
        return {k[len("_orig_mod.") :]: v for k, v in state.items()}
    return state


def load_adapter(path: Path, *, device: torch.device, hidden_channels: int, num_blocks: int) -> K5ToKVAEAdapter:
    hidden_channels, num_blocks, feature_mode = adapter_config_from_sidecar(path, hidden_channels, num_blocks)
    adapter = K5ToKVAEAdapter(hidden_channels=hidden_channels, num_blocks=num_blocks, feature_mode=feature_mode)
    adapter.load_state_dict(normalize_state_dict(load_file(str(path), device="cpu")), strict=True)
    adapter.to(device=device)
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


def video_to_u8_hwc(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x[0].permute(1, 2, 3, 0).contiguous().cpu()


def write_mp4(path: Path, video_bcthw: torch.Tensor, *, fps: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video_to_u8_hwc(video_bcthw)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_np in frames.numpy():
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def psnr_m11(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred.float() - target.float()).pow(2).mean().item()
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(mse)


def masked_l1_m11(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred_m = pred[:, :, mask]
    target_m = target[:, :, mask]
    if pred_m.numel() == 0:
        return float("nan")
    return float((pred_m.float() - target_m.float()).abs().mean().item())


def masked_psnr_m11(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred_m = pred[:, :, mask]
    target_m = target[:, :, mask]
    if pred_m.numel() == 0:
        return float("nan")
    return psnr_m11(pred_m, target_m)


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode a K5 crop through adapter -> KVAE-3D-2.0-t4s8.")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("datasets/DiT_latents"))
    ap.add_argument("--kvae", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/eval/k5_to_kvae_t4s8"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--t0", type=int, default=-1)
    ap.add_argument("--y0", type=int, default=-1)
    ap.add_argument("--x0", type=int, default=-1)
    ap.add_argument("--crop-px", type=int, default=256)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    samples = build_dit_samples(args.data_root)
    sample_idx = max(0, min(args.sample_idx, len(samples) - 1))
    dataset = K5CropDataset(samples, crop_px=args.crop_px, frames=args.frames, latent_cache_items=1, zarr_cache_items=1)
    total_t, total_h, total_w = dataset._load_latent(samples[sample_idx]).shape[1:]
    t0 = (total_t - dataset.t_lat) // 2 if args.t0 < 0 else args.t0
    y0 = (total_h - dataset.crop_lat) // 2 if args.y0 < 0 else args.y0
    x0 = (total_w - dataset.crop_lat) // 2 if args.x0 < 0 else args.x0
    z_crop, teacher, info = dataset.get_fixed(sample_idx, t0=t0, y0=y0, x0=x0)

    adapter = load_adapter(args.adapter, device=device, hidden_channels=args.hidden_channels, num_blocks=args.num_blocks)
    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=dtype)
    z = z_crop.unsqueeze(0).to(device=device, dtype=dtype)
    b, _, t, h, w = z.shape
    coords = make_coord_grid(
        batch=b,
        t=t,
        h=h,
        w=w,
        t0=torch.tensor([info.t0], device=device),
        y0=torch.tensor([info.y0], device=device),
        x0=torch.tensor([info.x0], device=device),
        total_t=info.total_t,
        total_h=info.total_h,
        total_w=info.total_w,
        device=device,
        dtype=dtype,
        feature_mode=adapter.feature_mode,
    )
    with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype, enabled=device.type == "cuda"):
        z_pred = adapter(z, coords)
        pred = kvae.decode(z_pred)
    teacher_b = teacher.unsqueeze(0).to(device=pred.device, dtype=pred.dtype)
    pred = pred[:, :, : teacher_b.shape[2], : teacher_b.shape[3], : teacher_b.shape[4]]
    side_by_side = torch.cat([teacher_b, pred], dim=-1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_mp4(args.out_dir / "teacher.mp4", teacher_b.cpu(), fps=args.fps)
    write_mp4(args.out_dir / "adapter_kvae.mp4", pred.cpu(), fps=args.fps)
    write_mp4(args.out_dir / "side_by_side_teacher_left.mp4", side_by_side.cpu(), fps=args.fps)
    metrics = {
        "sample_idx": sample_idx,
        "stem": info.stem,
        "t0": info.t0,
        "y0": info.y0,
        "x0": info.x0,
        "crop_px": args.crop_px,
        "frames": args.frames,
        "reference_source": "legacy_hvae_decode",
        "quality_target_note": "Use KVAE decoded-cache eval/training metrics for the real t4s8 target; this crop teacher is legacy HVAE decode.",
        **decoded_metrics_for_json(pred, teacher_b),
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
