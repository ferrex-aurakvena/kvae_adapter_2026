from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from .kvae2_loader import KVAE2T4S8, load_kvae2_t4s8, state_dict_report


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test the local KVAE-3D-2.0 t4s8 loader.")
    ap.add_argument("--weights", type=Path, default=Path("vae/KVAE_3D_2_0_t4s8.safetensors"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--decode-random", action="store_true", help="Run a random 9x16x16 latent through the decoder.")
    ap.add_argument("--encode-random", action="store_true", help="Run a random 33x128x128 clip through the encoder.")
    args = ap.parse_args()

    state = load_file(str(args.weights), device="cpu")
    report = state_dict_report(KVAE2T4S8(), state)
    print(f"model_keys={report['model_keys']} checkpoint_keys={report['checkpoint_keys']}")
    print(
        "missing={} unexpected={} shape_mismatches={}".format(
            len(report["missing"]), len(report["unexpected"]), len(report["shape_mismatches"])
        )
    )
    if report["missing"] or report["unexpected"] or report["shape_mismatches"]:
        for key in report["missing"][:20]:
            print(f"missing {key}")
        for key in report["unexpected"][:20]:
            print(f"unexpected {key}")
        for key, want, got in report["shape_mismatches"][:20]:
            print(f"shape {key} model={want} ckpt={got}")
        return 1

    dtype = None
    if args.dtype != "auto":
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = load_kvae2_t4s8(args.weights, device=args.device, dtype=dtype)
    print(f"loaded device={args.device} dtype={next(model.parameters()).dtype}")

    if args.decode_random:
        z = torch.randn(1, 16, 9, 16, 16, device=args.device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            x = model.decode(z)
        print(f"decode random: shape={tuple(x.shape)} finite={bool(torch.isfinite(x).all())}")

    if args.encode_random:
        x = torch.randn(1, 3, 33, 128, 128, device=args.device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            z = model.encode(x)
        print(f"encode random: shape={tuple(z.shape)} finite={bool(torch.isfinite(z).all())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
