from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-size throughput sweep for packed cached-latent training.")
    ap.add_argument("--cache-dir", type=Path, default=Path("_ignored/cache/kvae_t4s8_targets"))
    ap.add_argument("--out-dir", type=Path, default=Path("_ignored/runs/packed_benchmark"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", default="256,512,1024")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in [int(v.strip()) for v in args.batches.split(",") if v.strip()]:
        run_dir = args.out_dir / f"batch_{batch}"
        cmd = [
            sys.executable,
            "-m",
            "kvae_adapter.train_packed_cached",
            "--cache-dir",
            str(args.cache_dir),
            "--out-dir",
            str(run_dir),
            "--device",
            args.device,
            "--batch-size",
            str(batch),
            "--max-steps",
            str(args.steps),
            "--hidden-channels",
            str(args.hidden_channels),
            "--num-blocks",
            str(args.num_blocks),
            "--amp",
            args.amp,
            "--benchmark-only",
            "--log-every",
            str(max(1, args.steps // 2)),
        ]
        subprocess.run(cmd, check=True)
        with (run_dir / "throughput_summary.json").open("r", encoding="utf-8") as f:
            result = json.load(f)
        results.append(result)
    with (args.out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for result in results:
        print(
            "batch={batch_size} samples/s={samples_per_s:.0f} steps/s={steps_per_s:.2f} peak_mem_gb={peak_mem_gb:.2f}".format(
                **result
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
