from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from tqdm import tqdm

from .adapter import K5ToKVAEAdapter
from .packed_cache import PackedLatentCache


def make_coords_fast(batch: dict[str, torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
    z = batch["z"]
    b, _, t, h, w = z.shape
    device = z.device
    tt = torch.arange(t, device=device, dtype=dtype).view(1, 1, t, 1, 1)
    yy = torch.arange(h, device=device, dtype=dtype).view(1, 1, 1, h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, 1, w)
    t0 = batch["t0"].to(dtype=dtype).view(b, 1, 1, 1, 1)
    y0 = batch["y0"].to(dtype=dtype).view(b, 1, 1, 1, 1)
    x0 = batch["x0"].to(dtype=dtype).view(b, 1, 1, 1, 1)
    total_t = (batch["total_t"] - 1).clamp_min(1).to(dtype=dtype).view(b, 1, 1, 1, 1)
    total_h = (batch["total_h"] - 1).clamp_min(1).to(dtype=dtype).view(b, 1, 1, 1, 1)
    total_w = (batch["total_w"] - 1).clamp_min(1).to(dtype=dtype).view(b, 1, 1, 1, 1)
    return torch.cat(
        [
            ((t0 + tt) / total_t).expand(b, 1, t, h, w),
            ((y0 + yy) / total_h).expand(b, 1, t, h, w),
            ((x0 + xx) / total_w).expand(b, 1, t, h, w),
        ],
        dim=1,
    ).mul_(2).sub_(1)


def per_sample_temporal(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] < 2:
        return pred.new_zeros((pred.shape[0],))
    diff = (pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])
    return diff.float().abs().flatten(1).mean(dim=1)


def loss_vec(
    adapter: K5ToKVAEAdapter,
    batch: dict[str, torch.Tensor],
    *,
    amp_dtype: torch.dtype | None,
    temporal_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    autocast_device = "cuda" if batch["z"].device.type == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_dtype is not None):
        coords = make_coords_fast(batch, dtype=batch["z"].dtype)
        pred = adapter(batch["z"], coords)
        latent = (pred.float() - batch["target"].float()).pow(2).flatten(1).mean(dim=1)
        temporal = per_sample_temporal(pred, batch["target"])
        total = latent + temporal_weight * temporal
    return total, latent, temporal


def make_optimizer(params, *, lr: float, device: torch.device) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, fused=(device.type == "cuda"))
    except TypeError:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)


def save_checkpoint(path: Path, adapter: K5ToKVAEAdapter, config: dict[str, Any], step: int, stats: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, str(path), metadata={"step": str(step)})
    with path.with_suffix(path.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump({"step": step, "stats": stats, "config": config}, f, indent=2)


def sample_indices(
    *,
    cache_count: int,
    batch_size: int,
    device: torch.device,
    hard_indices: torch.Tensor,
    hard_prob: float,
    hard_replay: bool,
) -> tuple[torch.Tensor, bool]:
    use_hard = hard_replay and hard_indices.numel() > 0 and torch.rand((), device=device) < hard_prob
    if use_hard:
        picks = torch.randint(0, hard_indices.numel(), (batch_size,), device=device)
        return hard_indices.index_select(0, picks), True
    return torch.randint(0, cache_count, (batch_size,), device=device), False


def update_hard_queue(
    *,
    hard_indices: torch.Tensor,
    hard_scores: torch.Tensor,
    batch_indices: torch.Tensor,
    scores: torch.Tensor,
    max_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_items <= 0:
        return hard_indices, hard_scores
    all_indices = torch.cat([hard_indices, batch_indices.detach()])
    all_scores = torch.cat([hard_scores, scores.detach()])
    k = min(max_items, all_scores.numel())
    top_scores, top_pos = torch.topk(all_scores, k=k, largest=True, sorted=False)
    return all_indices.index_select(0, top_pos), top_scores


def main() -> int:
    ap = argparse.ArgumentParser(description="High-throughput K5->KVAE training from packed cached latents.")
    ap.add_argument("--cache-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_targets"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/runs/k5_to_kvae_t4s8_packed"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--temporal-weight", type=float, default=0.35)
    ap.add_argument("--gpu-resident", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--hard-replay", action="store_true")
    ap.add_argument("--hard-warmup", type=int, default=100)
    ap.add_argument("--hard-queue-k", type=int, default=4096)
    ap.add_argument("--hard-prob", type=float, default=0.5)
    ap.add_argument("--benchmark-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.set_float32_matmul_precision("high")
    amp_dtype = None if args.amp == "none" else {"fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    train_dtype = torch.float32 if amp_dtype is None else amp_dtype

    cache = PackedLatentCache.load(args.cache_dir, device=device, dtype=train_dtype, gpu_resident=args.gpu_resident)
    adapter = K5ToKVAEAdapter(hidden_channels=args.hidden_channels, num_blocks=args.num_blocks).to(device=device)
    if args.compile:
        adapter = torch.compile(adapter)
    optim = make_optimizer(adapter.parameters(), lr=args.lr, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()}
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    hard_indices = torch.empty(0, dtype=torch.long, device=device)
    hard_scores = torch.empty(0, dtype=torch.float32, device=device)
    last_stats = {"loss": float("nan"), "latent": float("nan"), "temporal": float("nan")}
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pbar = tqdm(range(1, args.max_steps + 1), desc="train-packed", dynamic_ncols=True)
    for step in pbar:
        batch_indices, used_hard = sample_indices(
            cache_count=cache.count,
            batch_size=args.batch_size,
            device=device,
            hard_indices=hard_indices,
            hard_prob=args.hard_prob,
            hard_replay=args.hard_replay and step >= args.hard_warmup,
        )
        batch = cache.batch(batch_indices, device=device, dtype=train_dtype)
        optim.zero_grad(set_to_none=True)
        total_vec, latent_vec, temporal_vec = loss_vec(
            adapter, batch, amp_dtype=amp_dtype, temporal_weight=args.temporal_weight
        )
        loss = total_vec.mean()
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

        if args.hard_replay and step >= args.hard_warmup and not used_hard:
            hard_indices, hard_scores = update_hard_queue(
                hard_indices=hard_indices,
                hard_scores=hard_scores,
                batch_indices=batch_indices,
                scores=total_vec.float(),
                max_items=args.hard_queue_k,
            )

        if step % args.log_every == 0 or step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            last_stats = {
                "loss": float(total_vec.mean().detach().item()),
                "latent": float(latent_vec.mean().detach().item()),
                "temporal": float(temporal_vec.mean().detach().item()),
            }
            samples_per_s = step * args.batch_size / elapsed
            pbar.set_postfix(
                loss=f"{last_stats['loss']:.5f}",
                latent=f"{last_stats['latent']:.5f}",
                temp=f"{last_stats['temporal']:.5f}",
                hard=int(hard_indices.numel()),
                samples_s=f"{samples_per_s:.0f}",
            )
        if step % args.save_every == 0 and not args.benchmark_only:
            save_checkpoint(args.out_dir / f"adapter_step_{step:06d}.safetensors", adapter, config, step, last_stats)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    summary = {
        **last_stats,
        "elapsed_s": elapsed,
        "steps_per_s": args.max_steps / elapsed,
        "samples_per_s": args.max_steps * args.batch_size / elapsed,
        "peak_mem_gb": peak_gb,
        "cache_count": cache.count,
        "batch_size": args.batch_size,
    }
    with (args.out_dir / "throughput_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not args.benchmark_only:
        save_checkpoint(args.out_dir / "adapter_final.safetensors", adapter, config, args.max_steps, last_stats)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
