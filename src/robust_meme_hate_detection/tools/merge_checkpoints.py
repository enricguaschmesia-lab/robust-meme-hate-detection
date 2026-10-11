"""Weight-space merge (model soup) of two CLIPHateMemeClassifier checkpoints.

merged[k] = alpha * A[k] + (1 - alpha) * B[k]   for floating-point tensors;
non-float buffers (masks, position ids) are copied from A. Both checkpoints
must share the architecture (they do: OpenCLIP ViT-B/32 + fusion head).

Usage:
  python scripts/merge_checkpoints.py --ckpt-a <A> --ckpt-b <B> --alpha 0.5 --out <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True, help="Checkpoint A (gets weight alpha)")
    ap.add_argument("--ckpt-b", required=True, help="Checkpoint B (gets weight 1-alpha)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    b = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)
    sa, sb = a["model_state"], b["model_state"]
    if set(sa) != set(sb):
        only_a = set(sa) - set(sb)
        only_b = set(sb) - set(sa)
        raise SystemExit(f"state_dict key mismatch: only_a={list(only_a)[:5]} only_b={list(only_b)[:5]}")

    alpha = float(args.alpha)
    merged = {}
    n_interp = n_copied = 0
    for k, ta in sa.items():
        tb = sb[k]
        if ta.is_floating_point() and ta.shape == tb.shape:
            merged[k] = alpha * ta + (1.0 - alpha) * tb
            n_interp += 1
        else:
            merged[k] = ta.clone()
            n_copied += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": merged, "config": a.get("config"), "seed": a.get("seed", 0),
         "merge": {"alpha": alpha, "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b}},
        out,
    )
    print(f"merged alpha={alpha}: interpolated {n_interp} tensors, copied {n_copied} from A -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
