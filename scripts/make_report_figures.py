"""Report-grade figures for the robust-meme-hate-detection project.

Complements `scripts/make_poster_figures.py`. The poster has a fixed 6-figure
budget; the report has appendix space, so this script produces the figures
that justify *premises* and *limitations* underneath the headline story.

Reuses the aggregator's data loaders so the figures stay in sync with the
per-phase tables. Output: `project_planning/report_figures/{fig}.{png,pdf}`
plus a matching `*.caption.txt`.

Figures:
  R1  image-family severity curves (companion to poster Fig 5; text-only)
  R2  OOD vs in-pool ΔAUROC (per recipe, image-side; Phase 5c-1 generalisation)
  R3  consensus-failure ceiling (% examples no recipe in the sweep solves)

Run:  PYTHONPATH=src .venv/bin/python3 scripts/make_report_figures.py
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "project_planning" / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("agg", REPO / "scripts" / "aggregate_phase4.py")
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

_spec_fa = importlib.util.spec_from_file_location("fa", REPO / "scripts" / "failure_analysis.py")
fa = importlib.util.module_from_spec(_spec_fa)
_spec_fa.loader.exec_module(fa)

SPLITS = ("dev", "test_seen", "test_unseen")
SPLIT_LABEL = {"dev": "dev (n=500, 36% pos)",
               "test_seen": "test_seen (n=1000, 49% pos)",
               "test_unseen": "test_unseen (n=2000, 37% pos)"}

# Reuse the poster palette for visual consistency.
RECIPE_COLOR = {
    "clean":         "#7f7f7f",
    "augonly":       "#1f77b4",
    "kl":            "#2ca02c",
    "kldrop":        "#d62728",
    "kldrop-p015":   "#ff9896",
    "kldrop-p050":   "#8c564b",
    "kllowmed":      "#9467bd",
}


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def _save(fig, name: str, caption: str) -> None:
    import matplotlib.pyplot as plt
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    (OUT_DIR / f"{name}.caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"  wrote {png.relative_to(REPO)} + {pdf.name} + caption")


def _payload_for(perturbed, recipe: str, seed: int, split: str = "dev"):
    if recipe in ("clean", "stage1"):
        return perturbed.get((split, f"stage1-seed{seed}"))
    for prefix in (f"robust-{recipe}-seed", f"train-robust-{recipe}-seed"):
        k = f"{prefix}{seed}"
        if (split, k) in perturbed:
            return perturbed[(split, k)]
    return None


def _is_image_attack(name: str) -> bool:
    fam = agg._attack_family(name)
    return fam.startswith("image-") or fam == "typographic"


# =============================================================================
# R1 — Image-family severity curves (companion to poster Fig 5)
# =============================================================================

def fig_image_severity_curves(perturbed_all):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    severities = ("low", "medium", "high")
    for ax, split in zip(axes, SPLITS):
        for recipe in agg.MAIN_VARIANT_ORDER:
            seeds_payloads = [_payload_for(perturbed_all, recipe, s, split) for s in (0, 1, 2)]
            seeds_payloads = [p for p in seeds_payloads if p is not None]
            if not seeds_payloads:
                continue
            means, stds = [], []
            for sev in severities:
                vals_per_seed = []
                for p in seeds_payloads:
                    img_gaps = [c["robustness_gap"]["auroc"] for c in p["cells"]
                                if _is_image_attack(c["attack"])
                                and c["severity_level"] == sev
                                and not str(c["attack"]).startswith("composite_")]
                    if img_gaps:
                        vals_per_seed.append(statistics.mean(img_gaps))
                if vals_per_seed:
                    means.append(statistics.mean(vals_per_seed))
                    stds.append(statistics.pstdev(vals_per_seed) if len(vals_per_seed) > 1 else 0.0)
                else:
                    means.append(float("nan")); stds.append(0.0)
            ax.errorbar(range(3), means, yerr=stds,
                        marker="o", markersize=6, capsize=3,
                        color=RECIPE_COLOR.get(recipe, "k"), label=recipe)
        ax.set_xticks(range(3))
        ax.set_xticklabels(severities)
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.set_ylim(bottom=-0.005)
    axes[0].set_ylabel("Mean image-family ΔAUROC ↓")
    axes[0].set_xlabel("Severity")
    axes[1].set_xlabel("Severity (medium = realistic, high = stress)")
    axes[2].set_xlabel("Severity")
    axes[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Per-severity image-attack ΔAUROC — uniformly an order of magnitude smaller than text",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "R1_image_severity_curves",
          "Mean ΔAUROC across the image-attack families (gaussian_noise, blur, "
          "compression, brightness/contrast, translation/crop, occlusion, "
          "typographic) per severity, per recipe. Companion to poster Fig 5 "
          "(text severity). Note the y-axis scale: image-side ΔAUROC at high "
          "severity is < 0.04 for every recipe on every split — an order of "
          "magnitude smaller than the text-side worst-cell Δ (~0.11). This "
          "asymmetry is the empirical premise for `modality_dropout_text` "
          "rather than `modality_dropout_image`: the brittle modality is text, "
          "so the defense forces the under-used image branch to do work.")


# =============================================================================
# R2 — In-pool vs OOD image attacks (Phase 5c-1 generalisation)
# =============================================================================

def fig_pool_vs_ood(perturbed_all):
    plt = _plt()
    import numpy as np

    pools = agg._in_pool_attacks()
    image_all = set(agg.IMAGE_ATTACKS) | {"typographic"}
    image_in = pools["image"] & image_all
    image_ood = image_all - pools["image"]

    def _mean_per_seed(payloads, names):
        out = []
        for p in payloads:
            gaps = [c["robustness_gap"]["auroc"] for c in p["cells"]
                    if not str(c["attack"]).startswith("composite_")
                    and c["attack"] in names]
            if gaps:
                out.append(sum(gaps) / len(gaps))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    width = 0.35
    for ax, split in zip(axes, SPLITS):
        recipes_here, in_means, in_stds, oo_means, oo_stds = [], [], [], [], []
        for recipe in agg.MAIN_VARIANT_ORDER:
            seeds_payloads = [_payload_for(perturbed_all, recipe, s, split) for s in (0, 1, 2)]
            seeds_payloads = [p for p in seeds_payloads if p is not None]
            if not seeds_payloads:
                continue
            in_vals = _mean_per_seed(seeds_payloads, image_in)
            oo_vals = _mean_per_seed(seeds_payloads, image_ood)
            if not in_vals or not oo_vals:
                continue
            recipes_here.append(recipe)
            in_means.append(statistics.mean(in_vals))
            in_stds.append(statistics.pstdev(in_vals) if len(in_vals) > 1 else 0.0)
            oo_means.append(statistics.mean(oo_vals))
            oo_stds.append(statistics.pstdev(oo_vals) if len(oo_vals) > 1 else 0.0)
        x = np.arange(len(recipes_here))
        ax.bar(x - width / 2, in_means, width, yerr=in_stds, capsize=3,
               color="#1f77b4", label="in-pool image attacks")
        ax.bar(x + width / 2, oo_means, width, yerr=oo_stds, capsize=3,
               color="#ff7f0e", label="held-out (OOD) image attacks")
        ax.set_xticks(x)
        ax.set_xticklabels(recipes_here, rotation=30, ha="right", fontsize=8)
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Mean ΔAUROC ↓ (clean − attacked)")
    axes[0].legend(loc="upper left", fontsize=8)
    in_str = ", ".join(sorted(image_in))
    oo_str = ", ".join(sorted(image_ood))
    fig.suptitle(
        "Image-attack generalisation: in-pool vs held-out (OOD)\n"
        f"in-pool: {in_str}    |    OOD: {oo_str}",
        fontsize=9)
    fig.tight_layout()
    _save(fig, "R2_ood_vs_inpool",
          "Per-recipe mean ΔAUROC on image attacks, partitioned by training-"
          "pool membership. Blue = in-pool, orange = held-out (OOD). For every "
          "robust recipe on every split, the OOD bar is at or below the in-"
          "pool bar — augmentation gains *transfer* to image attacks the model "
          "never saw in training. (Phase 5c-1 finding, now visible.) Text-side "
          "is omitted: held-out text attacks are intrinsically weak on the "
          "clean ckpt (≤ 0.02 ΔAUROC), so the text OOD column has limited "
          "signal — see `phase4/robust_vs_clean.md` for the full table.")


# =============================================================================
# R3 — Consensus-failure ceiling (% examples no recipe in sweep solves)
# =============================================================================

def fig_consensus_ceiling():
    plt = _plt()
    import numpy as np

    recipes = ["stage1", "augonly", "kl", "kldrop", "kldrop-p015",
               "kldrop-p025", "kldrop-p050", "kllowmed"]
    seeds = [0, 1, 2]
    splits_n = {"dev": 500, "test_seen": 1000, "test_unseen": 2000}

    counts = {}
    for split in SPLITS:
        captions = fa._load_captions(split)
        table, _ = fa._build_full_table(recipes, seeds, captions, split=split)
        consensus = fa._consensus_natural_failures(table, recipes, seeds)
        n_total = splits_n[split]
        n_consensus = len(consensus)
        label_counts = {0: 0, 1: 0}
        for e in consensus:
            label_counts[int(e.get("label", 0))] += 1
        counts[split] = {
            "n": n_total,
            "consensus": n_consensus,
            "pct": 100.0 * n_consensus / n_total if n_total else 0.0,
            "by_label": label_counts,
        }

    fig, (ax_pct, ax_label) = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: % consensus failures per split
    x = np.arange(len(SPLITS))
    pcts = [counts[s]["pct"] for s in SPLITS]
    ns = [counts[s]["consensus"] for s in SPLITS]
    totals = [counts[s]["n"] for s in SPLITS]
    bars = ax_pct.bar(x, pcts, color=["#7f7f7f", "#1f77b4", "#d62728"],
                      width=0.55)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=8, rotation=10)
    ax_pct.set_ylabel("% examples consensus-failed by every recipe")
    ax_pct.set_ylim(0, 100)
    ax_pct.axhline(70, ls="--", color="black", alpha=0.45, lw=1)
    ax_pct.text(len(SPLITS) - 0.4, 71, "~70 % ceiling", fontsize=8, ha="right")
    for xi, (pct, n, tot) in enumerate(zip(pcts, ns, totals)):
        ax_pct.text(xi, pct + 1.2, f"{pct:.1f} %\n({n}/{tot})",
                    ha="center", fontsize=8)
    ax_pct.set_title("Consensus residual failures across all 7 recipes × 3 seeds",
                     fontsize=10)

    # right: label composition of consensus failures
    width = 0.35
    label0 = [counts[s]["by_label"][0] for s in SPLITS]
    label1 = [counts[s]["by_label"][1] for s in SPLITS]
    ax_label.bar(x - width / 2, label0, width, color="#1f77b4", label="label=0 (non-hate)")
    ax_label.bar(x + width / 2, label1, width, color="#d62728", label="label=1 (hate)")
    ax_label.set_xticks(x)
    ax_label.set_xticklabels([s for s in SPLITS], fontsize=9)
    ax_label.set_ylabel("Consensus-failure count")
    ax_label.legend(fontsize=8, loc="upper left")
    for xi, (l0, l1) in enumerate(zip(label0, label1)):
        tot = l0 + l1
        if tot:
            ax_label.text(xi - width / 2, l0 + max(1, 0.03 * max(label0 + label1)),
                          f"{100 * l0 / tot:.0f}%", ha="center", fontsize=7)
            ax_label.text(xi + width / 2, l1 + max(1, 0.03 * max(label0 + label1)),
                          f"{100 * l1 / tot:.0f}%", ha="center", fontsize=7)
    ax_label.set_title("Label composition of the consensus-failure set",
                       fontsize=10)
    fig.suptitle("Consensus-failure ceiling: the project's empirical ~70 % wall",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "R3_consensus_failure_ceiling",
          "Left: percent of examples that every recipe in the project's sweep "
          "(`clean`, `augonly`, `kl`, `kldrop`, `kldrop-p015`, `kldrop-p025`, "
          "`kldrop-p050`, `kllowmed`) naturally fails on (majority across 3 "
          "seeds; "
          "`_is_natural_failure` ∈ {B2_clean_wrong, B3_natural_attack_flipped, "
          "B5_composite_only_failure}). Augmentation gains plateau at roughly "
          "70 % — this is the project's empirical ceiling for naturalistic "
          "robustness with the current backbone and training pool. Right: "
          "label composition of the consensus-failure set per split. The "
          "ceiling is not concentrated on one class. See "
          "`project_planning/Phase8_TestFailure_Report.md` §2 for the prose "
          "discussion.")


# =============================================================================
# main
# =============================================================================

def main() -> int:
    print("Loading aggregator data...")
    perturbed_all = agg._load_perturbed()
    print(f"  perturbed: {len(perturbed_all)} (split, key) pairs")
    print()

    print("R1: image_severity_curves")
    fig_image_severity_curves(perturbed_all)
    print("R2: ood_vs_inpool (image)")
    fig_pool_vs_ood(perturbed_all)
    print("R3: consensus_failure_ceiling")
    fig_consensus_ceiling()

    print()
    print(f"All figures in {OUT_DIR.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
