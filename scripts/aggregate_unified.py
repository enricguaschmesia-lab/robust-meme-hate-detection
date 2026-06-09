"""Aggregate the unified-robust train/test matrix into one cross-robustness table.

Reads the eval JSONs produced by ``scripts/run_unified_matrix.sh``:

    cluster-results/perturbed-unified-{mode}-seed{N}-{tag}/perturbed_eval.json
    cluster-results/whitebox-unified-{mode}-seed{N}-{tag}/whitebox_eval.json

with mode in {nat, adv, both}, tag in {dev, test-seen, test-unseen}. For each
(mode, split) it aggregates over seeds (mean ± std) three things:

  * Clean macro-F1 (at best threshold).
  * Naturalistic robustness: mean attacked macro-F1 over all perturbation cells
    (attack x severity), plus mean attack-success-rate.
  * White-box robustness: PGD macro-F1 at eps in {1,2,4,8}/255, plus PGD@8 ASR.

Writes ``project_planning/phase4/unified_cross_robustness.md`` and prints it.
This is intentionally separate from ``aggregate_phase4.py`` (which targets the
legacy recipe naming and the committed Phase-4/5 report tables) so the new
matrix does not disturb those artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "cluster-results"

MODES = ["nat", "adv", "both"]
MODE_LABEL = {"nat": "naturalistic", "adv": "adversarial", "both": "both"}
SPLITS = ["dev", "test_seen", "test_unseen"]
PGD_EPS = [1, 2, 4, 8]

_DIR_RE = re.compile(r"^(?:perturbed|whitebox)-unified-(nat|adv|both)-seed(\d+)-(dev|test-seen|test-unseen)$")


def _parse_dir(name: str) -> tuple[str, int, str] | None:
    m = _DIR_RE.match(name)
    if not m:
        return None
    mode, seed, tag = m.group(1), int(m.group(2)), m.group(3)
    return mode, seed, tag.replace("-", "_")


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def _fmt(mean: float, std: float) -> str:
    if mean != mean:
        return "—"
    return f"{mean:.3f}±{std:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", default="macro_f1", choices=("macro_f1", "auroc"),
        help="Classification metric to aggregate from the eval cells.",
    )
    args = parser.parse_args()
    metric = args.metric
    metric_label = {"macro_f1": "macro-F1", "auroc": "AUROC"}[metric]
    suffix = "" if metric == "macro_f1" else f".{metric}"
    out_md = REPO / "project_planning" / "phase4" / f"unified_cross_robustness{suffix}.md"

    # (mode, split) -> metric_name -> list over seeds
    perf: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for d in sorted(RESULTS.glob("perturbed-unified-*")):
        parsed = _parse_dir(d.name)
        f = d / "perturbed_eval.json"
        if not parsed or not f.exists():
            continue
        mode, _seed, split = parsed
        payload: dict[str, Any] = json.loads(f.read_text())
        clean = payload["clean"]["at_best_threshold"][metric]
        cells = payload["cells"]
        cell_f1 = [c["attacked_at_best_threshold"][metric] for c in cells]
        cell_asr = [c.get("attack_success_rate", float("nan")) for c in cells]
        perf[(mode, split)]["clean_f1"].append(clean)
        perf[(mode, split)]["nat_f1"].append(statistics.mean(cell_f1))
        perf[(mode, split)]["nat_asr"].append(statistics.mean([a for a in cell_asr if a == a]))

    for d in sorted(RESULTS.glob("whitebox-unified-*")):
        parsed = _parse_dir(d.name)
        f = d / "whitebox_eval.json"
        if not parsed or not f.exists():
            continue
        mode, _seed, split = parsed
        payload = json.loads(f.read_text())
        for c in payload["cells"]:
            if c.get("attack") != "pgd":
                continue
            eps = c.get("epsilon_numerator")
            perf[(mode, split)][f"pgd{eps}_f1"].append(c["attacked_at_best_threshold"][metric])
            if eps == 8:
                perf[(mode, split)]["pgd8_asr"].append(c.get("attack_success_rate", float("nan")))

    # ---------------------------------------------------------------- render
    lines: list[str] = []
    lines.append(f"# Unified robust models — cross-robustness comparison ({metric_label})\n")
    lines.append(
        "Three models trained with `train.stage1_robust_unified` "
        "(seeds 0/1/2, mean±std). `nat` = naturalistic only (frozen encoders); "
        "`adv` = PGD adversarial (vision unfrozen); `both` = combined.\n"
    )
    lines.append(f"- **Clean** = clean {metric_label} (best threshold).")
    lines.append(f"- **Nat** = mean attacked {metric_label} over all naturalistic cells; **Nat-ASR** = mean attack-success-rate.")
    lines.append(f"- **PGD@k** = {metric_label} under white-box PGD at ε=k/255; **PGD8-ASR** = ASR at ε=8/255.\n")

    for split in SPLITS:
        if not any((m, split) in perf for m in MODES):
            continue
        lines.append(f"\n## Split: `{split}`\n")
        header = "| Model | Clean | Nat | Nat-ASR | PGD@1 | PGD@2 | PGD@4 | PGD@8 | PGD8-ASR |"
        sep = "|---|---|---|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for mode in MODES:
            p = perf.get((mode, split))
            if not p:
                continue
            row = [f"`{mode}` ({MODE_LABEL[mode]})"]
            row.append(_fmt(*_mean_std(p.get("clean_f1", []))))
            row.append(_fmt(*_mean_std(p.get("nat_f1", []))))
            row.append(_fmt(*_mean_std(p.get("nat_asr", []))))
            for eps in PGD_EPS:
                row.append(_fmt(*_mean_std(p.get(f"pgd{eps}_f1", []))))
            row.append(_fmt(*_mean_std(p.get("pgd8_asr", []))))
            lines.append("| " + " | ".join(row) + " |")

    text = "\n".join(lines) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    print(text)
    print(f"[written] {out_md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
