from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from .adapter import FEATURE_MODE_CHOICES, FEATURE_MODE_COORDS, K5ToKVAEAdapter, resolve_feature_mode
from .decoded_cache import DecodedTargetCache
from .kvae2_loader import load_kvae2_t4s8
from .losses import (
    charbonnier_per_sample,
    flicker_error_per_sample,
    highpass_charbonnier_per_sample,
    jitter_error_per_sample,
    time_loss_weights,
)
from .metrics import DEFAULT_METRIC_LOG_KEYS, decoded_hard_replay_score, decoded_metric_tensors
from .packed_cache import PackedLatentCache
from .train_packed_cached import (
    make_coords_fast,
    make_optimizer,
    per_sample_temporal,
    sample_indices,
    save_checkpoint,
    update_hard_queue,
)


def adapter_config_from_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        return {}
    with sidecar.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload.get("config", {}))


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(k.startswith("_orig_mod.") for k in state):
        return {k[len("_orig_mod.") :]: v for k, v in state.items()}
    return state


def unwrap_adapter(adapter: torch.nn.Module) -> torch.nn.Module:
    return getattr(adapter, "_orig_mod", adapter)


def _adapt_tensor_for_shape(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if source.shape == target.shape:
        return source
    if source.ndim != target.ndim:
        return None
    if source.ndim >= 2 and source.shape[2:] != target.shape[2:]:
        return None
    if any(s > t for s, t in zip(source.shape, target.shape)):
        return None

    if source.ndim >= 2:
        adapted = torch.zeros_like(target)
    else:
        adapted = target.clone()
    slices = tuple(slice(0, int(s)) for s in source.shape)
    adapted[slices].copy_(source)
    return adapted


def _zero_extra_blocks(adapter: K5ToKVAEAdapter, source_num_blocks: int) -> None:
    for block in list(adapter.blocks)[source_num_blocks:]:
        for module in block.modules():
            if isinstance(module, torch.nn.Conv3d):
                torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


def load_state_compatible(
    adapter: K5ToKVAEAdapter,
    state: dict[str, torch.Tensor],
    *,
    source_num_blocks: int | None,
) -> None:
    target = adapter.state_dict()
    adapted_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    partial: list[str] = []
    for key, value in state.items():
        if key not in target:
            skipped.append(key)
            continue
        adapted = _adapt_tensor_for_shape(value, target[key])
        if adapted is None:
            skipped.append(key)
            continue
        if adapted.shape != value.shape:
            partial.append(key)
        adapted_state[key] = adapted
    missing, unexpected = adapter.load_state_dict(adapted_state, strict=False)
    if source_num_blocks is not None:
        _zero_extra_blocks(adapter, source_num_blocks)
    if skipped or unexpected:
        print(
            json.dumps(
                {
                    "event": "adapter_warm_start_partial",
                    "partial": partial,
                    "skipped": skipped,
                    "missing_count": len(missing),
                    "unexpected": unexpected,
                }
            ),
            flush=True,
        )


def load_trainable_adapter(
    path: Path | None,
    *,
    device: torch.device,
    hidden_channels: int,
    num_blocks: int,
    feature_mode: str,
    init_architecture: str,
) -> tuple[K5ToKVAEAdapter, int, int, str]:
    source_num_blocks: int | None = None
    if path is not None:
        cfg = adapter_config_from_sidecar(path)
        source_num_blocks = int(cfg.get("num_blocks", num_blocks))
        if init_architecture == "auto":
            hidden_channels = int(cfg.get("hidden_channels", hidden_channels))
            num_blocks = source_num_blocks
        if feature_mode == "auto":
            feature_mode = str(cfg.get("feature_mode", FEATURE_MODE_COORDS))
    feature_mode = resolve_feature_mode(feature_mode)
    adapter = K5ToKVAEAdapter(
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        feature_mode=feature_mode,
    ).to(device=device)
    if path is not None:
        state = normalize_state_dict(load_file(str(path), device="cpu"))
        load_state_compatible(adapter, state, source_num_blocks=source_num_blocks)
    adapter.train()
    return adapter, hidden_channels, num_blocks, feature_mode


def latent_loss_vec(
    adapter: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    amp_dtype: torch.dtype | None,
    temporal_weight: float,
    feature_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    autocast_device = "cuda" if batch["z"].device.type == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_dtype is not None):
        coords = make_coords_fast(batch, dtype=batch["z"].dtype, feature_mode=feature_mode)
        pred = adapter(batch["z"], coords)
        latent = (pred.float() - batch["target"].float()).pow(2).flatten(1).mean(dim=1)
        temporal = per_sample_temporal(pred, batch["target"])
        total = latent + temporal_weight * temporal
    return total, latent, temporal


def decoded_loss_vec(
    adapter: torch.nn.Module,
    kvae: torch.nn.Module,
    source_cache: PackedLatentCache,
    decoded_batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    train_dtype: torch.dtype,
    amp_dtype: torch.dtype | None,
    key_weight: float,
    inbetween_weight: float,
    highpass_weight: float,
    flicker_weight: float,
    jitter_weight: float,
    compute_metrics: bool,
    feature_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    source_indices = decoded_batch["source_index"]
    source_batch = source_cache.batch(source_indices, device=device, dtype=train_dtype)
    x_target = decoded_batch["x"].to(device=device, dtype=train_dtype)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_dtype is not None):
        coords = make_coords_fast(source_batch, dtype=source_batch["z"].dtype, feature_mode=feature_mode)
        z_pred = adapter(source_batch["z"], coords)
        x_pred = kvae.decode(z_pred)
        x_pred = x_pred[:, :, : x_target.shape[2], : x_target.shape[3], : x_target.shape[4]]

    weights = time_loss_weights(
        x_target.shape[2],
        key_weight=key_weight,
        inbetween_weight=inbetween_weight,
        device=device,
        dtype=torch.float32,
    ).view(1, 1, x_target.shape[2], 1, 1)
    pixel_vec = charbonnier_per_sample(x_pred, x_target, weight=weights)
    zero = pixel_vec.new_zeros(pixel_vec.shape)
    highpass_vec = (
        highpass_charbonnier_per_sample(x_pred, x_target, weight=weights) if highpass_weight > 0 else zero
    )
    flicker_vec = flicker_error_per_sample(x_pred, x_target) if flicker_weight > 0 else zero
    jitter_vec = jitter_error_per_sample(x_pred, x_target) if jitter_weight > 0 else zero
    total = pixel_vec + highpass_weight * highpass_vec + flicker_weight * flicker_vec + jitter_weight * jitter_vec
    metrics: dict[str, torch.Tensor] = {}
    if compute_metrics:
        with torch.no_grad():
            metrics = decoded_metric_tensors(x_pred.detach(), x_target.detach())
    return total, {
        "decoded_pixel": pixel_vec,
        "decoded_highpass": highpass_vec,
        "decoded_flicker": flicker_vec,
        "decoded_jitter": jitter_vec,
    }, metrics


def sample_decoded_batch(
    decoded_cache: DecodedTargetCache,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    hard_sources: torch.Tensor,
    hard_prob: float,
    hard_replay: bool,
) -> tuple[dict[str, torch.Tensor], bool]:
    use_hard = hard_replay and hard_sources.numel() > 0 and torch.rand((), device=device) < hard_prob
    if use_hard:
        picks = torch.randint(0, hard_sources.numel(), (batch_size,), device=device)
        source_indices = hard_sources.index_select(0, picks)
        return decoded_cache.batch_sources(source_indices, device=device, dtype=dtype), True
    rows = torch.randint(0, decoded_cache.count, (batch_size,), device=device)
    return decoded_cache.batch_rows(rows, device=device, dtype=dtype), False


def tensor_mean(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().item())


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune K5->KVAE adapter with sparse decoded KVAE-recon supervision.")
    ap.add_argument("--cache-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_targets_strat8192"))
    ap.add_argument("--decoded-cache-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_decoded_recon2048"))
    ap.add_argument("--kvae", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--init-adapter", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/runs/k5_to_kvae_t4s8_decoded_ft2048"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--latent-batch-size", type=int, default=512)
    ap.add_argument("--decoded-batch-size", type=int, default=2)
    ap.add_argument("--decoded-every", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--feature-mode", choices=("auto", *FEATURE_MODE_CHOICES), default="auto")
    ap.add_argument("--init-architecture", choices=("auto", "cli"), default="auto")
    ap.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--temporal-weight", type=float, default=0.35)
    ap.add_argument("--decoded-weight", type=float, default=0.25)
    ap.add_argument("--decoded-inbetween-weight", type=float, default=1.0)
    ap.add_argument("--decoded-key-weight", type=float, default=0.25)
    ap.add_argument("--decoded-highpass-weight", type=float, default=0.10)
    ap.add_argument("--decoded-flicker-weight", type=float, default=0.05)
    ap.add_argument("--decoded-jitter-weight", type=float, default=0.05)
    ap.add_argument("--decoded-metrics", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decoded-hard-metric-weight", type=float, default=1.0)
    ap.add_argument("--decoded-hard-phase-weight", type=float, default=1.0)
    ap.add_argument("--decoded-hard-fade-weight", type=float, default=0.75)
    ap.add_argument("--decoded-hard-detail-weight", type=float, default=1.0)
    ap.add_argument("--decoded-hard-grid-weight", type=float, default=0.5)
    ap.add_argument("--decoded-hard-color-weight", type=float, default=0.25)
    ap.add_argument("--gpu-resident", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decoded-gpu-resident", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--hard-replay", action="store_true")
    ap.add_argument("--hard-warmup", type=int, default=100)
    ap.add_argument("--hard-queue-k", type=int, default=4096)
    ap.add_argument("--hard-prob", type=float, default=0.5)
    ap.add_argument("--decoded-hard-queue-k", type=int, default=1024)
    ap.add_argument("--decoded-hard-prob", type=float, default=0.5)
    ap.add_argument("--benchmark-only", action="store_true")
    args = ap.parse_args()

    if args.latent_batch_size <= 0:
        raise ValueError("--latent-batch-size must be positive")
    if args.decoded_batch_size <= 0:
        raise ValueError("--decoded-batch-size must be positive")
    if args.decoded_every <= 0:
        raise ValueError("--decoded-every must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.set_float32_matmul_precision("high")
    amp_dtype = None if args.amp == "none" else {"fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    train_dtype = torch.float32 if amp_dtype is None else amp_dtype

    source_cache = PackedLatentCache.load(args.cache_dir, device=device, dtype=train_dtype, gpu_resident=args.gpu_resident)
    decoded_cache = DecodedTargetCache.load(
        args.decoded_cache_dir,
        device=device,
        dtype=train_dtype,
        gpu_resident=args.decoded_gpu_resident,
    )
    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=train_dtype)
    adapter, hidden_channels, num_blocks, feature_mode = load_trainable_adapter(
        args.init_adapter,
        device=device,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        feature_mode=args.feature_mode,
        init_architecture=args.init_architecture,
    )
    if args.compile:
        adapter = torch.compile(adapter)
    optim = make_optimizer(adapter.parameters(), lr=args.lr, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {k: str(v) for k, v in vars(args).items()}
    config["hidden_channels"] = str(hidden_channels)
    config["num_blocks"] = str(num_blocks)
    config["feature_mode"] = feature_mode
    config["coord_channels"] = str(unwrap_adapter(adapter).coord_channels)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    hard_indices = torch.empty(0, dtype=torch.long, device=device)
    hard_scores = torch.empty(0, dtype=torch.float32, device=device)
    decoded_hard_sources = torch.empty(0, dtype=torch.long, device=device)
    decoded_hard_scores = torch.empty(0, dtype=torch.float32, device=device)

    last_stats = {
        "loss": float("nan"),
        "latent": float("nan"),
        "temporal": float("nan"),
        "decoded": float("nan"),
        "decoded_pixel": float("nan"),
        "decoded_highpass": float("nan"),
        "decoded_flicker": float("nan"),
        "decoded_jitter": float("nan"),
        "decoded_hard_score": float("nan"),
        **{key: float("nan") for key in DEFAULT_METRIC_LOG_KEYS},
    }
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(range(1, args.max_steps + 1), desc="train-decoded-ft", dynamic_ncols=True)
    for step in pbar:
        batch_indices, used_hard = sample_indices(
            cache_count=source_cache.count,
            batch_size=args.latent_batch_size,
            device=device,
            hard_indices=hard_indices,
            hard_prob=args.hard_prob,
            hard_replay=args.hard_replay and step >= args.hard_warmup,
        )
        batch = source_cache.batch(batch_indices, device=device, dtype=train_dtype)
        optim.zero_grad(set_to_none=True)
        total_vec, latent_vec, temporal_vec = latent_loss_vec(
            adapter,
            batch,
            amp_dtype=amp_dtype,
            temporal_weight=args.temporal_weight,
            feature_mode=feature_mode,
        )
        latent_loss = total_vec.mean()
        decoded_loss = latent_loss.new_zeros(())
        decoded_vec = latent_loss.new_zeros((0,))
        decoded_parts = {
            "decoded_pixel": latent_loss.new_zeros((1,)),
            "decoded_highpass": latent_loss.new_zeros((1,)),
            "decoded_flicker": latent_loss.new_zeros((1,)),
            "decoded_jitter": latent_loss.new_zeros((1,)),
        }
        decoded_metric_parts = {key: latent_loss.new_zeros((1,)) for key in DEFAULT_METRIC_LOG_KEYS}
        decoded_hard_score_vec = latent_loss.new_zeros((1,))
        decoded_batch: dict[str, torch.Tensor] | None = None
        used_decoded_hard = False
        if args.decoded_weight > 0 and step % args.decoded_every == 0:
            decoded_batch, used_decoded_hard = sample_decoded_batch(
                decoded_cache,
                batch_size=args.decoded_batch_size,
                device=device,
                dtype=train_dtype,
                hard_sources=decoded_hard_sources,
                hard_prob=args.decoded_hard_prob,
                hard_replay=args.hard_replay and step >= args.hard_warmup,
            )
            decoded_vec, decoded_parts, decoded_metric_parts = decoded_loss_vec(
                adapter,
                kvae,
                source_cache,
                decoded_batch,
                device=device,
                train_dtype=train_dtype,
                amp_dtype=amp_dtype,
                key_weight=args.decoded_key_weight,
                inbetween_weight=args.decoded_inbetween_weight,
                highpass_weight=args.decoded_highpass_weight,
                flicker_weight=args.decoded_flicker_weight,
                jitter_weight=args.decoded_jitter_weight,
                compute_metrics=args.decoded_metrics,
                feature_mode=feature_mode,
            )
            for key in DEFAULT_METRIC_LOG_KEYS:
                decoded_metric_parts.setdefault(key, decoded_vec.new_zeros(decoded_vec.shape))
            decoded_loss = decoded_vec.mean()
            decoded_hard_score_vec = decoded_hard_replay_score(
                decoded_vec,
                decoded_metric_parts,
                metric_weight=args.decoded_hard_metric_weight,
                phase_weight=args.decoded_hard_phase_weight,
                fade_weight=args.decoded_hard_fade_weight,
                detail_weight=args.decoded_hard_detail_weight,
                grid_weight=args.decoded_hard_grid_weight,
                color_weight=args.decoded_hard_color_weight,
            )

        loss = latent_loss + args.decoded_weight * decoded_loss
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
        if (
            args.hard_replay
            and step >= args.hard_warmup
            and decoded_batch is not None
            and not used_decoded_hard
            and decoded_vec.numel() > 0
        ):
            decoded_hard_sources, decoded_hard_scores = update_hard_queue(
                hard_indices=decoded_hard_sources,
                hard_scores=decoded_hard_scores,
                batch_indices=decoded_batch["source_index"],
                scores=decoded_hard_score_vec.float(),
                max_items=args.decoded_hard_queue_k,
            )

        if step % args.log_every == 0 or step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            last_stats = {
                "loss": tensor_mean(loss),
                "latent": tensor_mean(latent_vec),
                "temporal": tensor_mean(temporal_vec),
                "decoded": tensor_mean(decoded_loss),
                "decoded_pixel": tensor_mean(decoded_parts["decoded_pixel"]),
                "decoded_highpass": tensor_mean(decoded_parts["decoded_highpass"]),
                "decoded_flicker": tensor_mean(decoded_parts["decoded_flicker"]),
                "decoded_jitter": tensor_mean(decoded_parts["decoded_jitter"]),
                "decoded_hard_score": tensor_mean(decoded_hard_score_vec),
                **{key: tensor_mean(decoded_metric_parts[key]) for key in DEFAULT_METRIC_LOG_KEYS},
            }
            samples_per_s = step * args.latent_batch_size / elapsed
            pbar.set_postfix(
                loss=f"{last_stats['loss']:.5f}",
                latent=f"{last_stats['latent']:.5f}",
                dec=f"{last_stats['decoded']:.5f}",
                shimmer=f"{last_stats['phase_shimmer_score']:.4f}",
                detail=f"{last_stats['detail_loss_score']:.3f}",
                hard=f"{int(hard_indices.numel())}/{int(decoded_hard_sources.numel())}",
                samples_s=f"{samples_per_s:.0f}",
            )
        if step % args.save_every == 0 and not args.benchmark_only:
            save_checkpoint(
                args.out_dir / f"adapter_step_{step:06d}.safetensors",
                unwrap_adapter(adapter),
                config,
                step,
                last_stats,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    summary = {
        **last_stats,
        "elapsed_s": elapsed,
        "steps_per_s": args.max_steps / elapsed,
        "latent_samples_per_s": args.max_steps * args.latent_batch_size / elapsed,
        "decoded_samples_per_s": args.max_steps * args.decoded_batch_size / (elapsed * args.decoded_every),
        "peak_mem_gb": peak_gb,
        "cache_count": source_cache.count,
        "decoded_cache_count": decoded_cache.count,
        "decoded_reference_source": "kvae_t4s8_recon_cache",
        "latent_batch_size": args.latent_batch_size,
        "decoded_batch_size": args.decoded_batch_size,
    }
    with (args.out_dir / "throughput_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not args.benchmark_only:
        save_checkpoint(args.out_dir / "adapter_final.safetensors", unwrap_adapter(adapter), config, args.max_steps, last_stats)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
