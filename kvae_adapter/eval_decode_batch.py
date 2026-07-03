from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import K5CropDataset, build_dit_samples
from .eval_decode import load_adapter, write_mp4
from .kvae2_loader import load_kvae2_t4s8
from .metrics import decoded_metric_tensors, metric_tensors_to_float_dict
from .train_packed_cached import make_coords_fast


@dataclass(frozen=True)
class EvalCase:
    label: str
    sample_idx: int
    t0: int
    y0: int
    x0: int


def clean_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    label = label.strip("._-")
    return label or "case"


def parse_case(spec: str) -> EvalCase:
    if "=" in spec:
        label, rest = spec.split("=", 1)
    else:
        label, rest = "", spec
    parts = rest.split(":")
    if len(parts) == 3:
        sample_idx, y0, x0 = (int(v) for v in parts)
        t0 = -1
    elif len(parts) == 4:
        sample_idx, t0, y0, x0 = (int(v) for v in parts)
    else:
        raise argparse.ArgumentTypeError(
            "case must be LABEL=SAMPLE:Y:X or LABEL=SAMPLE:T:Y:X, e.g. owl=108:-1:32:96"
        )
    if sample_idx < 0:
        raise argparse.ArgumentTypeError("sample index must be non-negative")
    return EvalCase(label=clean_label(label or f"sample_{sample_idx:06d}"), sample_idx=sample_idx, t0=t0, y0=y0, x0=x0)


def video_to_u8_hwc(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 3, 0).contiguous().cpu()


def resolve_case(dataset: K5CropDataset, case: EvalCase) -> EvalCase:
    sample = dataset.samples[case.sample_idx]
    _, total_t, total_h, total_w = dataset._load_latent(sample).shape
    t0 = (total_t - dataset.t_lat) // 2 if case.t0 < 0 else case.t0
    y0 = (total_h - dataset.crop_lat) // 2 if case.y0 < 0 else case.y0
    x0 = (total_w - dataset.crop_lat) // 2 if case.x0 < 0 else case.x0
    return EvalCase(label=case.label, sample_idx=case.sample_idx, t0=t0, y0=y0, x0=x0)


def batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    case_offset: int,
    cases: list[EvalCase],
    stems: list[str],
) -> list[dict[str, object]]:
    metric_tensors = decoded_metric_tensors(pred, target)
    out = []
    for i, case in enumerate(cases):
        out.append(
            {
                "case_idx": case_offset + i,
                "label": case.label,
                "sample_idx": case.sample_idx,
                "stem": stems[i],
                "t0": case.t0,
                "y0": case.y0,
                "x0": case.x0,
                "reference_source": "legacy_hvae_decode",
                **metric_tensors_to_float_dict(metric_tensors, index=i),
            }
        )
    return out


def numeric_metric_keys(rows: list[dict[str, object]]) -> list[str]:
    skip = {"case_idx", "sample_idx", "t0", "y0", "x0"}
    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in skip:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                keys.add(key)
    return sorted(keys)


def aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in numeric_metric_keys(rows):
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        finite = [v for v in values if math.isfinite(v)]
        if finite:
            out[f"{key}_mean"] = sum(finite) / len(finite)
            out[f"{key}_max"] = max(finite)
    return out


def metric_rankings(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    ranking_keys = [
        "detail_loss_score",
        "phase_shimmer_score",
        "fade_strobe_score",
        "grid_artifact_score",
        "color_mean_l1",
        "temporal_jitter_score",
        "boundary_3_to_0_error",
    ]
    rankings: dict[str, list[dict[str, object]]] = {}
    for key in ranking_keys:
        ranked = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                ranked.append(
                    {
                        "label": row["label"],
                        "sample_idx": row["sample_idx"],
                        "value": float(value),
                    }
                )
        rankings[key] = sorted(ranked, key=lambda item: item["value"], reverse=True)
    return rankings


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    preferred = ["case_idx", "label", "sample_idx", "stem", "t0", "y0", "x0", "reference_source"]
    keys = preferred + [key for key in sorted({key for row in rows for key in row}) if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batched crop decode/eval for K5->KVAE adapter checkpoints.")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("datasets/DiT_latents"))
    ap.add_argument("--kvae", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/eval/k5_to_kvae_t4s8_batch"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--case", action="append", type=parse_case, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--crop-px", type=int, default=256)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--write-videos", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    samples = build_dit_samples(args.data_root)
    dataset = K5CropDataset(samples, crop_px=args.crop_px, frames=args.frames, latent_cache_items=2, zarr_cache_items=2)
    cases = [resolve_case(dataset, case) for case in args.case]
    for case in cases:
        if case.sample_idx >= len(samples):
            raise IndexError(f"sample index {case.sample_idx} is out of range for {len(samples)} samples")

    adapter = load_adapter(args.adapter, device=device, hidden_channels=args.hidden_channels, num_blocks=args.num_blocks)
    kvae = load_kvae2_t4s8(args.kvae, device=device, dtype=dtype)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, object]] = []
    for start in range(0, len(cases), args.batch_size):
        case_batch = cases[start : start + args.batch_size]
        crops = [dataset.get_fixed(case.sample_idx, t0=case.t0, y0=case.y0, x0=case.x0) for case in case_batch]
        z_list, teacher_list, infos = zip(*crops)
        z = torch.stack(list(z_list), dim=0).to(device=device, dtype=dtype)
        teacher = torch.stack(list(teacher_list), dim=0).to(device=device, dtype=dtype)
        batch = {
            "z": z,
            "t0": torch.tensor([i.t0 for i in infos], device=device, dtype=torch.long),
            "y0": torch.tensor([i.y0 for i in infos], device=device, dtype=torch.long),
            "x0": torch.tensor([i.x0 for i in infos], device=device, dtype=torch.long),
            "total_t": torch.tensor([i.total_t for i in infos], device=device, dtype=torch.long),
            "total_h": torch.tensor([i.total_h for i in infos], device=device, dtype=torch.long),
            "total_w": torch.tensor([i.total_w for i in infos], device=device, dtype=torch.long),
        }
        with torch.no_grad(), torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            coords = make_coords_fast(batch, dtype=dtype, feature_mode=adapter.feature_mode)
            z_pred = adapter(z, coords)
            pred = kvae.decode(z_pred)
        pred = pred[:, :, : teacher.shape[2], : teacher.shape[3], : teacher.shape[4]]
        stems = [i.stem for i in infos]
        metrics = batch_metrics(pred, teacher, start, case_batch, stems)
        all_metrics.extend(metrics)

        if args.write_videos:
            for i, metric in enumerate(metrics):
                label = str(metric["label"])
                case_dir = args.out_dir / label
                teacher_i = teacher[i].detach().cpu()
                pred_i = pred[i].detach().cpu()
                write_mp4(case_dir / "teacher.mp4", teacher_i.unsqueeze(0), fps=args.fps)
                write_mp4(case_dir / "adapter_kvae.mp4", pred_i.unsqueeze(0), fps=args.fps)
                write_mp4(case_dir / "side_by_side_teacher_left.mp4", torch.cat([teacher_i, pred_i], dim=-1).unsqueeze(0), fps=args.fps)
                with (case_dir / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(metric, f, indent=2)

    manifest = {
        "format": "kvae_adapter_batch_eval_v1",
        "adapter": str(args.adapter),
        "data_root": str(args.data_root),
        "kvae": str(args.kvae),
        "reference_source": "legacy_hvae_decode",
        "quality_target_note": "These fixed-crop teachers come from legacy HVAE decode; use decoded-cache training/eval metrics for KVAE t4s8 target quality.",
        "crop_px": args.crop_px,
        "frames": args.frames,
        "cases": [asdict(case) | {"stem": samples[case.sample_idx].stem} for case in cases],
        "aggregate_metrics": aggregate_metrics(all_metrics),
        "rankings": metric_rankings(all_metrics),
        "metrics": all_metrics,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_metrics_csv(args.out_dir / "metrics.csv", all_metrics)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
