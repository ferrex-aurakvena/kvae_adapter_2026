from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapter import K5ToKVAEAdapter, make_coord_grid
from .data import K5CropDataset, build_dit_samples, collate_crops
from .kvae2_loader import load_kvae2_t4s8
from .losses import highpass_loss, temporal_delta_loss


@dataclass
class HardExample:
    score: float
    step_added: int
    z: torch.Tensor
    x: torch.Tensor
    t0: torch.Tensor
    y0: torch.Tensor
    x0: torch.Tensor
    total_t: torch.Tensor
    total_h: torch.Tensor
    total_w: torch.Tensor
    uses: int = 0


def add_hard_example(queue: list[HardExample], item: HardExample, max_items: int) -> None:
    if max_items <= 0:
        return
    queue.append(item)
    queue.sort(key=lambda v: v.score, reverse=True)
    del queue[max_items:]


def prune_hard_queue(queue: list[HardExample], *, step: int, max_age: int, max_uses: int) -> None:
    queue[:] = [q for q in queue if (step - q.step_added) <= max_age and q.uses < max_uses]


def hard_to_batch(item: HardExample) -> dict[str, Any]:
    item.uses += 1
    return {
        "z": item.z.clone(),
        "x": item.x.clone(),
        "t0": item.t0.clone(),
        "y0": item.y0.clone(),
        "x0": item.x0.clone(),
        "total_t": item.total_t.clone(),
        "total_h": item.total_h.clone(),
        "total_w": item.total_w.clone(),
        "stem": ["hard"],
    }


def to_hard_example(batch: dict[str, Any], *, score: float, step: int) -> HardExample:
    return HardExample(
        score=float(score),
        step_added=step,
        z=batch["z"].detach().to("cpu", dtype=torch.float16),
        x=batch["x"].detach().to("cpu", dtype=torch.float16),
        t0=batch["t0"].detach().cpu(),
        y0=batch["y0"].detach().cpu(),
        x0=batch["x0"].detach().cpu(),
        total_t=batch["total_t"].detach().cpu(),
        total_h=batch["total_h"].detach().cpu(),
        total_w=batch["total_w"].detach().cpu(),
    )


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out = dict(batch)
    out["z"] = batch["z"].to(device=device, dtype=dtype, non_blocking=True)
    out["x"] = batch["x"].to(device=device, dtype=dtype, non_blocking=True)
    for key in ("t0", "y0", "x0", "total_t", "total_h", "total_w"):
        out[key] = batch[key].to(device=device, non_blocking=True)
    return out


def train_step(
    *,
    adapter: K5ToKVAEAdapter,
    kvae,
    batch: dict[str, Any],
    amp_dtype: torch.dtype | None,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    z = batch["z"]
    x_target = batch["x"]
    b, _, t, h, w = z.shape
    coords = make_coord_grid(
        batch=b,
        t=t,
        h=h,
        w=w,
        t0=batch["t0"],
        y0=batch["y0"],
        x0=batch["x0"],
        total_t=int(batch["total_t"].max().item()),
        total_h=int(batch["total_h"].max().item()),
        total_w=int(batch["total_w"].max().item()),
        device=z.device,
        dtype=z.dtype,
    )
    autocast_device = "cuda" if z.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_dtype is not None):
        z_pred = adapter(z, coords)
        x_pred = kvae.decode(z_pred)
        if x_pred.shape[2:] != x_target.shape[2:]:
            x_pred = x_pred[:, :, : x_target.shape[2], : x_target.shape[3], : x_target.shape[4]]
        pixel = F.l1_loss(x_pred.float(), x_target.float())
        temporal = temporal_delta_loss(x_pred.float(), x_target.float())
        high = highpass_loss(x_pred.float(), x_target.float())
        latent_reg = F.mse_loss(z_pred.float(), z.float())
        loss = (
            weights["pixel"] * pixel
            + weights["temporal"] * temporal
            + weights["highpass"] * high
            + weights["latent_reg"] * latent_reg
        )
    stats = {
        "loss": float(loss.detach().item()),
        "pixel": float(pixel.detach().item()),
        "temporal": float(temporal.detach().item()),
        "highpass": float(high.detach().item()),
        "latent_reg": float(latent_reg.detach().item()),
    }
    return loss, stats


def save_checkpoint(path: Path, adapter: K5ToKVAEAdapter, config: dict[str, Any], step: int, stats: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    save_file(tensors, str(path), metadata={"step": str(step)})
    meta = {"step": step, "stats": stats, "config": config}
    with path.with_suffix(path.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rapid K5 Pro latent -> KVAE-3D-2.0 t4s8 adapter trainer.")
    ap.add_argument("--data-root", type=Path, default=Path("datasets/DiT_latents"))
    ap.add_argument("--kvae", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/runs/k5_to_kvae_t4s8_rapid"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--crop-px", type=int, default=128)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--latent-space", default="auto", choices=["auto", "scaled", "unscaled"])
    ap.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--pixel-weight", type=float, default=1.0)
    ap.add_argument("--temporal-weight", type=float, default=0.35)
    ap.add_argument("--highpass-weight", type=float, default=0.15)
    ap.add_argument("--latent-reg-weight", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--hard-replay", action="store_true")
    ap.add_argument("--hard-warmup", type=int, default=100)
    ap.add_argument("--hard-queue-k", type=int, default=64)
    ap.add_argument("--hard-prob", type=float, default=0.5)
    ap.add_argument("--hard-min-loss", type=float, default=0.12)
    ap.add_argument("--hard-max-age", type=int, default=2000)
    ap.add_argument("--hard-max-uses", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    amp_dtype = None if args.amp == "none" else {"fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    train_dtype = torch.float32 if amp_dtype is None else amp_dtype

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["data_root"] = str(config["data_root"])
    config["kvae"] = str(config["kvae"])
    config["out_dir"] = str(config["out_dir"])
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    samples = build_dit_samples(args.data_root)
    dataset = K5CropDataset(
        samples,
        crop_px=args.crop_px,
        frames=args.frames,
        latent_space=args.latent_space,
        latent_cache_items=2 if args.num_workers == 0 else 0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_crops,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    stream = cycle_loader(loader)

    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    adapter = K5ToKVAEAdapter(hidden_channels=args.hidden_channels, num_blocks=args.num_blocks).to(device=device)
    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    weights = {
        "pixel": args.pixel_weight,
        "temporal": args.temporal_weight,
        "highpass": args.highpass_weight,
        "latent_reg": args.latent_reg_weight,
    }
    hard_queue: list[HardExample] = []
    last_stats: dict[str, float] = {}
    start = time.time()
    pbar = tqdm(range(1, args.max_steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        use_hard = (
            args.hard_replay
            and step >= args.hard_warmup
            and hard_queue
            and random.random() < max(0.0, min(1.0, args.hard_prob))
        )
        raw_batch = hard_to_batch(random.choice(hard_queue)) if use_hard else next(stream)
        batch = move_batch(raw_batch, device, train_dtype)
        optim.zero_grad(set_to_none=True)
        loss, stats = train_step(adapter=adapter, kvae=kvae, batch=batch, amp_dtype=amp_dtype, weights=weights)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optim.step()

        if args.hard_replay and not use_hard and step >= args.hard_warmup and stats["loss"] >= args.hard_min_loss:
            add_hard_example(hard_queue, to_hard_example(raw_batch, score=stats["loss"], step=step), args.hard_queue_k)
        if args.hard_replay and step % 25 == 0:
            prune_hard_queue(hard_queue, step=step, max_age=args.hard_max_age, max_uses=args.hard_max_uses)

        last_stats = stats
        if step % args.log_every == 0 or step == 1:
            elapsed = max(time.time() - start, 1e-9)
            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                pix=f"{stats['pixel']:.4f}",
                temp=f"{stats['temporal']:.4f}",
                hard=len(hard_queue),
                sps=f"{step / elapsed:.2f}",
            )
        if step % args.save_every == 0:
            save_checkpoint(args.out_dir / f"adapter_step_{step:06d}.safetensors", adapter, config, step, stats)

    save_checkpoint(args.out_dir / "adapter_final.safetensors", adapter, config, args.max_steps, last_stats)
    print(f"saved {args.out_dir / 'adapter_final.safetensors'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
