"""Comprehensive cross-robustness log for the adversarial / combined models.

Reads every relevant eval JSON under ``cluster-results/`` and writes one tracked
markdown summary so the full set of adversarial and combined results is logged
in version control (the raw ``cluster-results/`` JSONs are untracked and can be
lost). Covers, per split (dev / test_seen / test_unseen) and 3-seed mean:

  * Clean AUROC (best threshold).
  * Naturalistic robustness: worst-cell attacked AUROC with its drop
    (Delta AUROC vs clean) and the mean attacked AUROC over all cells.
  * White-box robustness: PGD attacked AUROC at eps=8/255 with its drop, + ASR.

Model rows (each labelled frozen/unfrozen encoders):

  Clean (stage1)                 frozen   -- clean-trained baseline
  Naturalistic KLDrop p0.15      frozen   -- report headline naturalistic recipe
  Naturalistic KLDrop p0.30      frozen   -- naturalistic component of the matrix
  Adversarial                    unfrozen -- PGD adversarial training
  Adversarial (frozen)           frozen   -- head-only control (if run)
  Combined                       unfrozen -- naturalistic + adversarial
  Combined (frozen)              frozen   -- head-only control (seed-0 probe)

Run: ``python scripts/aggregate_cross_robustness.py`` (writes the .md and prints).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "cluster-results"
OUT_MD = REPO / "project_planning" / "phase4" / "cross_robustness_full.md"

SPLITS = ["dev", "test_seen", "test_unseen"]
_DASH = {"dev": "dev", "test_seen": "test-seen", "test_unseen": "test-unseen"}

# (label, encoders, perturbed-prefix, whitebox-prefix, naming)
#   naming "legacy": <prefix>-<dash-split>-seed*  (dev has no split tag)
#   naming "unified": <prefix>-seed*-<dash-split>
MODELS = [
    ("Clean (stage1)",            "frozen",   "perturbed-stage1",                "whitebox-stage1",                "legacy"),
    ("Naturalistic KLDrop p0.15", "frozen",   "perturbed-robust-kldrop-p015",    "whitebox-robust-kldrop-p015",    "legacy"),
    ("Naturalistic KLDrop p0.30", "frozen",   "perturbed-unified-nat",           "whitebox-unified-nat",           "unified"),
    ("Adversarial",               "unfrozen", "perturbed-unified-adv",           "whitebox-unified-adv",           "unified"),
    ("Adversarial (frozen)",      "frozen",   "perturbed-unified-advfrozen",     "whitebox-unified-advfrozen",     "unified"),
    ("Combined",                  "unfrozen", "perturbed-unified-both",          "whitebox-unified-both",          "unified"),
    ("Combined (frozen)",         "frozen",   "perturbed-unified-bothfrozen",    "whitebox-unified-bothfrozen",    "unified"),
]


def _glob(prefix: str, naming: str, split: str) -> str:
    sp = _DASH[split]
    if naming == "legacy":
        return f"{prefix}-seed*" if split == "dev" else f"{prefix}-{sp}-seed*"
    return f"{prefix}-seed*-{sp}"


def _ms(xs: list[float]) -> Optional[tuple[float, float, int]]:
    xs = [x for x in xs if x == x]
    if not xs:
        return None
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0), len(xs)


def _perturbed(prefix, naming, split):
    """Return per-seed lists: clean, nat_worst_auroc, nat_worst_delta, nat_mean, worst_cell_label."""
    cl, nw, nd, nm, cells = [], [], [], [], []
    for d in sorted(RESULTS.glob(_glob(prefix, naming, split))):
        f = d / "perturbed_eval.json"
        if not f.exists():
            continue
        p = json.loads(f.read_text())
        c = p["clean"]["at_best_threshold"]["auroc"]
        cl.append(c)
        worst = max(p["cells"], key=lambda x: c - x["attacked_at_best_threshold"]["auroc"])
        a = worst["attacked_at_best_threshold"]["auroc"]
        nw.append(a); nd.append(c - a)
        cells.append(f"{worst.get('attack')}/{worst.get('severity_level')}")
        nm.append(statistics.mean([x["attacked_at_best_threshold"]["auroc"] for x in p["cells"]]))
    return cl, nw, nd, nm, cells


def _pgd8(prefix, naming, split):
    """Return per-seed lists: clean, pgd8_auroc, pgd8_delta, pgd8_asr."""
    cl, pa, pd, asr = [], [], [], []
    for d in sorted(RESULTS.glob(_glob(prefix, naming, split))):
        f = d / "whitebox_eval.json"
        if not f.exists():
            continue
        p = json.loads(f.read_text())
        c = p["clean"]["at_best_threshold"]["auroc"]
        for cell in p["cells"]:
            if cell.get("attack") == "pgd" and cell.get("epsilon_numerator") == 8:
                a = cell["attacked_at_best_threshold"]["auroc"]
                cl.append(c); pa.append(a); pd.append(c - a)
                asr.append(cell.get("attack_success_rate", float("nan")))
    return cl, pa, pd, asr


def _fmt(t: Optional[tuple[float, float, int]], dp: int = 3) -> str:
    if t is None:
        return "—"
    m, s, n = t
    return f"{m:.{dp}f}±{s:.{dp}f}" if n > 1 else f"{m:.{dp}f}"


def _fmt_ad(a, d) -> str:
    """attacked AUROC (Δ) cell."""
    if a is None:
        return "—"
    am, _, _ = a
    dm = d[0] if d else float("nan")
    av = f"{am:.3f}" if am >= 0.01 else "$<$0.01"
    return f"{av} ($-{dm:.2f}$)" if dm == dm else av


def main() -> int:
    lines: list[str] = []
    lines.append("# Cross-robustness — full results log (adversarial & combined)\n")
    lines.append(
        "Auto-generated by `scripts/aggregate_cross_robustness.py` from the raw "
        "eval JSONs in `cluster-results/`. AUROC, $3$-seed mean$\\pm$std "
        "(single seed where noted by n=1). `Nat worst` = worst single naturalistic "
        "cell (attacked AUROC, drop $\\Delta$ vs clean); `Nat mean` = mean over all "
        "cells; `PGD@8` = white-box PGD $L_\\infty$ at $\\epsilon{=}8/255$.\n"
    )
    for split in SPLITS:
        lines.append(f"\n## `{split}`\n")
        lines.append("| Model | Enc. | n | Clean | Nat worst (Δ) | Nat mean | PGD@8 (Δ) | PGD8 ASR | worst cell |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label, enc, pp, wp, naming in MODELS:
            cl, nw, nd, nm, cells = _perturbed(pp, naming, split)
            wcl, pa, pd, asr = _pgd8(wp, naming, split)
            n = max(len(cl), len(pa))
            if n == 0:
                lines.append(f"| {label} | {enc} | 0 | — | — | — | — | — | — |")
                continue
            worst_cell = max(set(cells), key=cells.count) if cells else "—"
            row = [
                label, enc, str(n),
                _fmt(_ms(cl)),
                _fmt_ad(_ms(nw), nd),
                _fmt(_ms(nm)) if nm else "—",
                _fmt_ad(_ms(pa), pd),
                _fmt(_ms(asr), dp=3),
                worst_cell,
            ]
            lines.append("| " + " | ".join(row) + " |")

    text = "\n".join(lines) + "\n"
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"[written] {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
