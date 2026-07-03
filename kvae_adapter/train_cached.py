from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapter import K5ToKVAEAdapter, make_coord_grid


@dataclass(frozen=True)
class CachedItem:
    path: Path


class CachedPairDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: Path) -> None:
        self.items = [CachedItem(p) for p in sorted(cache_dir.glob("*.safetensors"))]
        if not self.items:
            raise RuntimeError(f"No cached target pairs found under {cache_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.items[idx].path
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            z_k5 = f.get_tensor("z_k5_unscaled")
            z_kvae = f.get_tensor("z_kvae_t4s8")
        return {
            "z": z_k5,
            "target": z_kvae,
            "t0": int(meta.get("t0", 0)),
            "y0": int(meta.get("y0", 0)),
            "x0": int(meta.get("x0", 0)),
            "total_t": int(meta.get("total_t", z_k5.shape[1])),
            "total_h": int(meta.get("total_h", z_k5.shape[2])),
            "total_w": int(meta.get("total_w", z_k5.shape[3])),
            "path": path.name,
        }


def collate_cached(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "z": torch.stack([b["z"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "t0": torch.tensor([b["t0"] for b in batch], dtype=torch.long),
        "y0": torch.tensor([b["y0"] for b in batch], dtype=torch.long),
        "x0": torch.tensor([b["x0"] for b in batch], dtype=torch.long),
        "total_t": torch.tensor([b["total_t"] for b in batch], dtype=torch.long),
        "total_h": torch.tensor([b["total_h"] for b in batch], dtype=torch.long),
        "total_w": torch.tensor([b["total_w"] for b in batch], dtype=torch.long),
        "path": [b["path"] for b in batch],
    }


@dataclass
class HardCached:
    score: float
    step_added: int
    batch: dict[str, Any]
    uses: int = 0


def move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out = dict(batch)
    out["z"] = batch["z"].to(device=device, dtype=dtype, non_blocking=True)
    out["target"] = batch["target"].to(device=device, dtype=dtype, non_blocking=True)
    for key in ("t0", "y0", "x0", "total_t", "total_h", "total_w"):
        out[key] = batch[key].to(device=device, non_blocking=True)
    return out


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def temporal_latent_delta(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] < 2:
        return pred.new_zeros(())
    return F.l1_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def train_loss(
    adapter: K5ToKVAEAdapter,
    batch: dict[str, Any],
    *,
    amp_dtype: torch.dtype | None,
    temporal_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    z = batch["z"]
    target = batch["target"]
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
        pred = adapter(z, coords)
        latent = F.mse_loss(pred.float(), target.float())
        temporal = temporal_latent_delta(pred.float(), target.float())
        loss = latent + temporal_weight * temporal
    return loss, {"loss": float(loss.detach().item()), "latent": float(latent.detach().item()), "temporal": float(temporal.detach().item())}


def save_checkpoint(path: Path, adapter: K5ToKVAEAdapter, config: dict[str, Any], step: int, stats: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, str(path), metadata={"step": str(step)})
    with path.with_suffix(path.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump({"step": step, "stats": stats, "config": config}, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train K5->KVAE adapter from cached KVAE target latents.")
    ap.add_argument("--cache-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_targets"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/runs/k5_to_kvae_t4s8_cached"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--temporal-weight", type=float, default=0.35)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--hard-replay", action="store_true")
    ap.add_argument("--hard-warmup", type=int, default=100)
    ap.add_argument("--hard-queue-k", type=int, default=64)
    ap.add_argument("--hard-prob", type=float, default=0.5)
    ap.add_argument("--hard-max-age", type=int, default=2000)
    ap.add_argument("--hard-max-uses", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp_dtype = None if args.amp == "none" else {"fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    train_dtype = torch.float32 if amp_dtype is None else amp_dtype
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items()}
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    dataset = CachedPairDataset(args.cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cached,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    stream = cycle_loader(loader)
    adapter = K5ToKVAEAdapter(hidden_channels=args.hidden_channels, num_blocks=args.num_blocks).to(device=device)
    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    hard_queue: list[HardCached] = []
    last_stats: dict[str, float] = {}
    start = time.time()
    pbar = tqdm(range(1, args.max_steps + 1), desc="train-cached", dynamic_ncols=True)
    for step in pbar:
        use_hard = (
            args.hard_replay
            and step >= args.hard_warmup
            and hard_queue
            and random.random() < max(0.0, min(1.0, args.hard_prob))
        )
        if use_hard:
            item = random.choice(hard_queue)
            item.uses += 1
            raw_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in item.batch.items()}
        else:
            raw_batch = next(stream)
        batch = move_batch(raw_batch, device, train_dtype)
        optim.zero_grad(set_to_none=True)
        loss, stats = train_loss(adapter, batch, amp_dtype=amp_dtype, temporal_weight=args.temporal_weight)
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
        if args.hard_replay and not use_hard and step >= args.hard_warmup:
            hard_queue.append(HardCached(stats["loss"], step, {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in raw_batch.items()}))
            hard_queue.sort(key=lambda item: item.score, reverse=True)
            del hard_queue[args.hard_queue_k :]
        if args.hard_replay and step % 25 == 0:
            hard_queue[:] = [
                h for h in hard_queue if (step - h.step_added) <= args.hard_max_age and h.uses < args.hard_max_uses
            ]
        last_stats = stats
        if step % args.log_every == 0 or step == 1:
            elapsed = max(time.time() - start, 1e-9)
            pbar.set_postfix(
                loss=f"{stats['loss']:.5f}",
                latent=f"{stats['latent']:.5f}",
                temp=f"{stats['temporal']:.5f}",
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
