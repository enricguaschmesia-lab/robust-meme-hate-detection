"""Poster-grade figures for the robust-meme-hate-detection project.

Reuses the aggregator's data loaders (`scripts/aggregate_phase4.py`) so the
figures are always in sync with the per-phase tables. Each figure has its
own function; figures gracefully skip if their data isn't on disk yet.

Output: `project_planning/poster_figures/{fig_name}.{png,pdf,svg}` plus a
matching `*.caption.txt` with a one-paragraph caption.

Run:  PYTHONPATH=src .venv/bin/python3 scripts/make_poster_figures.py
"""

from __future__ import annotations

import importlib.util
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "project_planning" / "poster_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import aggregator as a module (so we reuse _load_*, _is_robust_key, etc.)
_spec = importlib.util.spec_from_file_location("agg", REPO / "scripts" / "aggregate_phase4.py")
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

# Optional: failure_analysis for class-asymmetric figure
_spec_fa = importlib.util.spec_from_file_location("fa", REPO / "scripts" / "failure_analysis.py")
fa = importlib.util.module_from_spec(_spec_fa)
_spec_fa.loader.exec_module(fa)

SPLITS = ("dev", "test_seen", "test_unseen")
SPLIT_LABEL = {"dev": "dev (n=500, 36% pos)",
               "test_seen": "test_seen (n=1000, 49% pos)",
               "test_unseen": "test_unseen (n=2000, 37% pos)"}

# Split-matched dedicated image-only baseline AUROCs from the Phase 1-2 image-
# only baseline checkpoint (`cluster-results/baseline-image-seed0/ckpt/best.pt`)
# evaluated on each split via `run_perturbed --modality image`. dev value is
# from `perturbed-baseline-image-seed0`; test_{seen,unseen} are from
# `perturbed-baseline-image-test-{seen,unseen}-seed0` (post-Phase-10
# follow-up). The text-only baseline is from `baseline-text-seed0` on dev.
BASELINE_IMG_BY_SPLIT = {
    "dev":         0.628,
    "test_seen":   0.640,
    "test_unseen": 0.650,
}
BASELINE_IMG_SPLIT_AVG = sum(BASELINE_IMG_BY_SPLIT.values()) / len(BASELINE_IMG_BY_SPLIT)  # ≈ 0.639
BASELINE_TXT_DEV = 0.632

# Color palette: tab10-derived, recipe-grouped
RECIPE_COLOR = {
    "clean":         "#7f7f7f",  # grey
    "augonly":       "#1f77b4",  # blue
    "kl":            "#2ca02c",  # green
    "kldrop":        "#d62728",  # red
    "kldrop-p015":   "#ff9896",  # light red
    "kldrop-p025":   "#ff5722",  # deep orange (between p015 and p050)
    "kldrop-p050":   "#8c564b",  # dark red/brown
    "kllowmed":      "#9467bd",  # purple
}

# All recipe-grouped figures (Figs 1 / 2 / 2m / 2b / 2bm / 3 / 5) use the
# six-recipe `MAIN_VARIANT_ORDER`. `kldrop-p025` is co-optimal with `kldrop-p015`
# (image-only AUROC peak 0.668 on test_unseen) but is shown only in Fig 6's
# dropout sweep — including it in the branch-revival figures duplicates the
# co-optimality message and produces unreadable marker overlap with `p015`.


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
        "svg.fonttype": "none",
    })
    return plt


def _save(fig, name: str, caption: str) -> None:
    import matplotlib.pyplot as plt
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png)
    fig.savefig(pdf)
    fig.savefig(svg)
    (OUT_DIR / f"{name}.caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"  wrote {png.relative_to(REPO)} + {pdf.name} + {svg.name} + caption")


# ---------------------------------------------------------------- helpers

def _payload_for(perturbed, recipe: str, seed: int, split: str = "dev"):
    """Return a perturbed payload for (recipe, seed, split) or None."""
    if recipe in ("clean", "stage1"):
        key = f"stage1-seed{seed}"
    else:
        for prefix in (f"robust-{recipe}-seed", f"train-robust-{recipe}-seed"):
            k = f"{prefix}{seed}"
            if (split, k) in perturbed:
                return perturbed[(split, k)]
        return None
    return perturbed.get((split, key))


def _per_recipe_summary(perturbed, recipe: str, split: str = "dev") -> dict:
    """Return per-recipe summary: clean AUROC, worst-cell text Δ, image-only Δ etc."""
    seeds_payloads = [_payload_for(perturbed, recipe, s, split) for s in (0, 1, 2)]
    seeds_payloads = [p for p in seeds_payloads if p is not None]
    if not seeds_payloads:
        return {}
    cleans = [p["clean"]["at_best_threshold"]["auroc"] for p in seeds_payloads]
    worst_text = []
    for p in seeds_payloads:
        text_gaps = [c["robustness_gap"]["auroc"] for c in p["cells"]
                     if agg._attack_family(c["attack"]) == "text"]
        if text_gaps:
            worst_text.append(max(text_gaps))
    return {
        "n_seeds": len(seeds_payloads),
        "clean_auroc_mean": statistics.mean(cleans),
        "clean_auroc_std": statistics.pstdev(cleans) if len(cleans) > 1 else 0.0,
        "worst_text_mean": statistics.mean(worst_text) if worst_text else float("nan"),
        "worst_text_std": statistics.pstdev(worst_text) if len(worst_text) > 1 else 0.0,
    }


# =============================================================================
# Figure 1 — Headline Pareto front (Clean AUROC vs Worst-cell text Δ) × 3 splits
# =============================================================================

def fig_headline_pareto(perturbed_all):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    handles_by_recipe: dict[str, object] = {}
    for ax, split in zip(axes, SPLITS):
        for recipe in agg.MAIN_VARIANT_ORDER:
            s = _per_recipe_summary(perturbed_all, recipe, split)
            if not s:
                continue
            (line, _, _) = ax.errorbar(
                s["clean_auroc_mean"], s["worst_text_mean"],
                xerr=s["clean_auroc_std"], yerr=s["worst_text_std"],
                marker="o", markersize=8, color=RECIPE_COLOR.get(recipe, "k"),
                capsize=3, linestyle="", label=recipe,
            )
            handles_by_recipe.setdefault(recipe, line)
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.set_xlabel("Clean AUROC →")
        ax.invert_yaxis()  # smaller Δ is better; put it up
    axes[0].set_ylabel("Worst-cell text ΔAUROC ↓")
    # Single shared legend across all three panels (recipe ↔ colour).
    recipe_order = [r for r in agg.MAIN_VARIANT_ORDER if r in handles_by_recipe]
    fig.legend([handles_by_recipe[r] for r in recipe_order], recipe_order,
               loc="lower center", ncol=len(recipe_order), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Headline Pareto: clean accuracy vs single-cell worst-case text robustness", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "01_headline_pareto",
          "Per-recipe Pareto trade-off on each split. Down-right is better "
          "(high clean AUROC, low worst-cell ΔAUROC under text attacks). "
          "`kldrop-p050` (brown) is the strict winner on Worst-cell text Δ "
          "across all three splits; `kldrop-p015` (light red) sits on the "
          "Pareto front at zero clean-accuracy cost vs `kl`; `kl` (green) "
          "preserves clean AUROC best. The legacy `kldrop` (p=0.30) is "
          "Pareto-dominated by `kldrop-p015` and demoted to the appendix "
          "(see `MAIN_VARIANT_ORDER` in `scripts/aggregate_phase4.py`).")


# =============================================================================
# Figure 2 — Image-branch revival (modality ablation × 3 splits)
# =============================================================================

def fig_image_branch_revival(modality_all):
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    # Split-matched dedicated image-only baseline (Phase 1-2 + post-Phase-10
    # follow-up runs on the held-out test splits).
    baseline_line = None
    for ax, split in zip(axes, SPLITS):
        baseline = BASELINE_IMG_BY_SPLIT[split]
        for variant in agg.MAIN_VARIANT_ORDER:
            payloads = []
            for seed in (0, 1, 2):
                if variant == "clean":
                    p = modality_all.get((split, f"stage1-seed{seed}"))
                else:
                    p = (modality_all.get((split, f"robust-{variant}-seed{seed}"))
                         or modality_all.get((split, f"train-robust-{variant}-seed{seed}")))
                if p is not None:
                    payloads.append(p)
            if not payloads:
                continue
            mm = [p["multimodal"]["at_best_threshold"]["auroc"] for p in payloads]
            io = [p["image_only"]["at_best_threshold"]["auroc"] for p in payloads]
            ax.errorbar(variant, statistics.mean(mm),
                        yerr=statistics.pstdev(mm) if len(mm) > 1 else 0,
                        color=RECIPE_COLOR.get(variant, "k"),
                        marker="^", markersize=8, capsize=3, linestyle="")
            ax.errorbar(variant, statistics.mean(io),
                        yerr=statistics.pstdev(io) if len(io) > 1 else 0,
                        color=RECIPE_COLOR.get(variant, "k"),
                        marker="s", markersize=8, capsize=3, linestyle="",
                        alpha=0.6)
        baseline_line = ax.axhline(baseline, ls="--", c="black", alpha=0.4, lw=1)
        ax.text(0.02, baseline + 0.004, f"baseline = {baseline:.3f}",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=8, color="black", alpha=0.6)
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0.55, 0.78)
    axes[0].set_ylabel("AUROC")
    legend_handles = [
        Line2D([0], [0], marker="^", color="black", linestyle="", markersize=8,
               label="multimodal AUROC"),
        Line2D([0], [0], marker="s", color="black", linestyle="", markersize=8,
               alpha=0.6, label="image-only AUROC"),
        Line2D([0], [0], color="black", linestyle="--", alpha=0.4, lw=1,
               label="dedicated image-only baseline (split-matched)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Image-branch revival: multimodal (▲) and image-only (■) forward AUROC per recipe", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "02_image_branch_revival",
          "Per-recipe multimodal (triangle) and image-only (square) AUROC on "
          "each split. The dashed line on each panel is the split-matched "
          "dedicated image-only baseline (0.628 / 0.640 / 0.650 on dev / "
          "test_seen / test_unseen, evaluated on the Phase 1-2 image-only "
          "checkpoint). Both modality-dropout recipes shown (`kldrop-p015`, "
          "`kldrop-p050`) match or exceed the baseline on every split — "
          "`kldrop-p015`: 0.638 / 0.638 / 0.658 (Δ = +0.010 / −0.002 / +0.008 "
          "vs baseline); `kldrop-p050`: 0.641 / 0.651 / 0.666 (Δ = +0.013 / "
          "+0.011 / +0.016). By contrast `kl`'s image-only AUROC is "
          "uniformly below the split-matched baseline (Δ = −0.017 / −0.035 / "
          "−0.022). `kldrop-p015` matches the image-branch revival at zero "
          "clean-accuracy cost vs `kl`. The intermediate point `kldrop-p025` "
          "is co-optimal with `kldrop-p015` and is the image-only AUROC peak "
          "(0.668 on test_unseen, +0.018 above the split-matched baseline); "
          "it is omitted here to avoid marker overlap and is shown in Fig 6's "
          "7-point dropout sweep.")


# =============================================================================
# Figure 2m — Image-branch revival, single-panel merged across splits (poster)
# =============================================================================

def fig_image_branch_revival_merged(modality_all):
    """Single-panel version of Fig 2 — collapses the 3 splits into one panel.

    Per-recipe AUROC for the multimodal forward and the image-only forward,
    averaged across (3 splits × 3 seeds). Error bars show split-σ — the σ of
    the three per-split means (each per-split mean already averages over the
    3 seeds). That visualises split-to-split reproducibility directly: a tiny
    error bar means the recipe's branch AUROC barely moves between dev,
    test_seen, and test_unseen.
    """
    plt = _plt()
    from matplotlib.lines import Line2D
    import numpy as np

    baseline = BASELINE_IMG_SPLIT_AVG  # mean(0.628, 0.640, 0.650) ≈ 0.639

    def _per_split_mean(variant: str, branch: str, split: str) -> float | None:
        vals = []
        for seed in (0, 1, 2):
            if variant == "clean":
                p = modality_all.get((split, f"stage1-seed{seed}"))
            else:
                p = (modality_all.get((split, f"robust-{variant}-seed{seed}"))
                     or modality_all.get((split, f"train-robust-{variant}-seed{seed}")))
            if p is not None:
                vals.append(p[branch]["at_best_threshold"]["auroc"])
        return statistics.mean(vals) if vals else None

    recipes_plotted: list[str] = []
    mm_means: list[float] = []
    mm_stds: list[float] = []
    io_means: list[float] = []
    io_stds: list[float] = []
    for variant in agg.MAIN_VARIANT_ORDER:
        mm_split_means = [v for split in SPLITS
                          if (v := _per_split_mean(variant, "multimodal", split)) is not None]
        io_split_means = [v for split in SPLITS
                          if (v := _per_split_mean(variant, "image_only", split)) is not None]
        if not mm_split_means or not io_split_means:
            continue
        recipes_plotted.append(variant)
        mm_means.append(statistics.mean(mm_split_means))
        mm_stds.append(statistics.pstdev(mm_split_means) if len(mm_split_means) > 1 else 0.0)
        io_means.append(statistics.mean(io_split_means))
        io_stds.append(statistics.pstdev(io_split_means) if len(io_split_means) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    x = np.arange(len(recipes_plotted))
    offset = 0.12
    for xi, recipe in zip(x, recipes_plotted):
        color = RECIPE_COLOR.get(recipe, "k")
        ax.errorbar(xi - offset, mm_means[recipes_plotted.index(recipe)],
                    yerr=mm_stds[recipes_plotted.index(recipe)],
                    color=color, marker="^", markersize=10, capsize=4,
                    linestyle="")
        ax.errorbar(xi + offset, io_means[recipes_plotted.index(recipe)],
                    yerr=io_stds[recipes_plotted.index(recipe)],
                    color=color, marker="s", markersize=10, capsize=4,
                    linestyle="", alpha=0.65)
    ax.axhline(baseline, ls="--", c="black", alpha=0.45, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(recipes_plotted, rotation=20, ha="right")
    ax.set_ylabel("AUROC  (mean across 3 splits × 3 seeds)")
    ax.set_ylim(0.55, 0.78)

    legend_handles = [
        Line2D([0], [0], marker="^", color="black", linestyle="", markersize=10,
               label="multimodal AUROC"),
        Line2D([0], [0], marker="s", color="black", linestyle="", markersize=10,
               alpha=0.65, label="image-only AUROC"),
        Line2D([0], [0], color="black", linestyle="--", alpha=0.45, lw=1,
               label=f"image-only baseline (split-avg) = {baseline:.3f}"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
              frameon=False)
    ax.set_title("Image-branch revival — split-averaged "
                 "(error bar = σ across dev / test_seen / test_unseen)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "02m_image_branch_revival_merged",
          "Single-panel Fig 2 for the poster: per-recipe multimodal (▲) and "
          "image-only (■) forward AUROC, averaged across the 3 splits (dev, "
          "test_seen, test_unseen) and the 3 seeds. Error bar = σ of the "
          "three per-split means — i.e. how much the answer shifts between "
          "splits. Bars are uniformly small (≤ 0.012 AUROC) so the per-split "
          "version (Fig 2) collapses without distortion. Dashed line = "
          f"split-averaged dedicated image-only baseline = {baseline:.3f} "
          "(= mean(0.628, 0.640, 0.650) over the per-split Phase 1-2 image-"
          "only checkpoint evaluations). The kldrop family is the only one "
          "above the baseline on image-only AUROC; split-averaged image-only "
          "AUROC: `kldrop-p015` 0.645, `kldrop-p050` 0.653. The intermediate "
          "`kldrop-p025` (image-only peak 0.668 on test_unseen) is omitted "
          "here for marker readability — see Fig 6 for the full 7-point "
          "sweep. Use this version when poster real estate is tight; use the "
          "3-panel Fig 2 in the report appendix for the dev → test "
          "reproduction story.")

def fig_image_branch_revival_with_text(modality_all):
    """Same as Fig 2 but adds text-only forward AUROC as a third marker.

    Visually shows that `clean` carries ~all signal through the text branch
    (the image branch is dead) and that modality-dropout recipes redistribute
    capacity into the image branch rather than adding it.
    """
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    baseline_txt = BASELINE_TXT_DEV  # dedicated text-only baseline (dev only)
    for ax, split in zip(axes, SPLITS):
        baseline_img = BASELINE_IMG_BY_SPLIT[split]
        for variant in agg.MAIN_VARIANT_ORDER:
            payloads = []
            for seed in (0, 1, 2):
                if variant == "clean":
                    p = modality_all.get((split, f"stage1-seed{seed}"))
                else:
                    p = (modality_all.get((split, f"robust-{variant}-seed{seed}"))
                         or modality_all.get((split, f"train-robust-{variant}-seed{seed}")))
                if p is not None:
                    payloads.append(p)
            if not payloads:
                continue
            mm = [p["multimodal"]["at_best_threshold"]["auroc"] for p in payloads]
            io = [p["image_only"]["at_best_threshold"]["auroc"] for p in payloads]
            to = [p["text_only"]["at_best_threshold"]["auroc"] for p in payloads]
            color = RECIPE_COLOR.get(variant, "k")
            ax.errorbar(variant, statistics.mean(mm),
                        yerr=statistics.pstdev(mm) if len(mm) > 1 else 0,
                        color=color, marker="^", markersize=8, capsize=3,
                        linestyle="")
            ax.errorbar(variant, statistics.mean(io),
                        yerr=statistics.pstdev(io) if len(io) > 1 else 0,
                        color=color, marker="s", markersize=8, capsize=3,
                        linestyle="", alpha=0.65)
            ax.errorbar(variant, statistics.mean(to),
                        yerr=statistics.pstdev(to) if len(to) > 1 else 0,
                        color=color, marker="v", markersize=8, capsize=3,
                        linestyle="", alpha=0.65, markerfacecolor="white")
        ax.axhline(baseline_img, ls="--", c="black", alpha=0.35, lw=1)
        ax.axhline(baseline_txt, ls=":", c="black", alpha=0.35, lw=1)
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0.50, 0.80)
    axes[0].set_ylabel("AUROC")
    legend_handles = [
        Line2D([0], [0], marker="^", color="black", linestyle="", markersize=8,
               label="multimodal AUROC"),
        Line2D([0], [0], marker="s", color="black", linestyle="", markersize=8,
               alpha=0.65, label="image-only AUROC"),
        Line2D([0], [0], marker="v", color="black", linestyle="", markersize=8,
               alpha=0.65, markerfacecolor="white", label="text-only AUROC"),
        Line2D([0], [0], color="black", linestyle="--", alpha=0.4, lw=1,
               label="image-only baseline (split-matched)"),
        Line2D([0], [0], color="black", linestyle=":", alpha=0.4, lw=1,
               label=f"text-only baseline = {baseline_txt:.3f} (dev)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-branch forward AUROC: multimodal (▲), image-only (■), text-only (▽)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "02b_image_branch_revival_with_text",
          "Augmented Fig 2 — adds text-only forward AUROC (open ▽) as a third "
          "marker per recipe. Dotted line = dedicated text-only baseline "
          "(0.632, dev only); dashed line = split-matched dedicated image-"
          "only baseline (0.628 / 0.640 / 0.650 on dev / test_seen / "
          "test_unseen). On `clean`, the text-only branch carries ~all the "
          "signal (the image branch is dead at AUROC ≈ 0.59-0.60). KL alone "
          "lifts image-only but stays uniformly below the split-matched "
          "image baseline (Δ = −0.017 / −0.035 / −0.022). `kldrop-p015` and "
          "`kldrop-p050` *redistribute* capacity: image-only AUROC reaches "
          "or exceeds the dedicated image baseline on every split while "
          "text-only AUROC remains within seed noise of `kl`. The "
          "intermediate `kldrop-p025` is co-optimal with `p015` and the "
          "image-only AUROC peak (0.668 on test_unseen, +0.018 above the "
          "split-matched baseline); it is omitted here for marker readability "
          "and is plotted in Fig 6. The defense is branch-balancing, not "
          "signal-adding.")


# =============================================================================
# Figure 2bm — Single-panel Fig 2 + text-only branch (merged across splits)
# =============================================================================

def fig_image_branch_revival_with_text_merged(modality_all):
    """Single-panel version of Fig 2b — three branches × all splits merged.

    Per recipe shows multimodal (▲), image-only (■), and text-only (▽) forward
    AUROC, each averaged across (3 splits × 3 seeds). Error bar = σ of the
    three per-split means, so a tiny bar means "this answer holds on dev,
    test_seen, and test_unseen alike". Tells the branch-redistribution story
    in a single poster-sized panel.
    """
    plt = _plt()
    from matplotlib.lines import Line2D
    import numpy as np

    baseline_img = BASELINE_IMG_SPLIT_AVG  # mean(0.628, 0.640, 0.650) ≈ 0.639
    baseline_txt = BASELINE_TXT_DEV

    def _per_split_mean(variant: str, branch: str, split: str) -> float | None:
        vals = []
        for seed in (0, 1, 2):
            if variant == "clean":
                p = modality_all.get((split, f"stage1-seed{seed}"))
            else:
                p = (modality_all.get((split, f"robust-{variant}-seed{seed}"))
                     or modality_all.get((split, f"train-robust-{variant}-seed{seed}")))
            if p is not None:
                vals.append(p[branch]["at_best_threshold"]["auroc"])
        return statistics.mean(vals) if vals else None

    def _merge(variant: str, branch: str) -> tuple[float, float] | None:
        per_split = [v for split in SPLITS
                     if (v := _per_split_mean(variant, branch, split)) is not None]
        if not per_split:
            return None
        return (statistics.mean(per_split),
                statistics.pstdev(per_split) if len(per_split) > 1 else 0.0)

    recipes_plotted: list[str] = []
    mm: list[tuple[float, float]] = []
    io: list[tuple[float, float]] = []
    to: list[tuple[float, float]] = []
    for variant in agg.MAIN_VARIANT_ORDER:
        a = _merge(variant, "multimodal")
        b = _merge(variant, "image_only")
        c = _merge(variant, "text_only")
        if a is None or b is None or c is None:
            continue
        recipes_plotted.append(variant)
        mm.append(a); io.append(b); to.append(c)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    x = np.arange(len(recipes_plotted))
    offsets = (-0.18, 0.0, 0.18)  # multimodal, image-only, text-only
    for xi, recipe, (mm_m, mm_s), (io_m, io_s), (to_m, to_s) in zip(
            x, recipes_plotted, mm, io, to):
        color = RECIPE_COLOR.get(recipe, "k")
        ax.errorbar(xi + offsets[0], mm_m, yerr=mm_s,
                    color=color, marker="^", markersize=10, capsize=4,
                    linestyle="")
        ax.errorbar(xi + offsets[1], io_m, yerr=io_s,
                    color=color, marker="s", markersize=10, capsize=4,
                    linestyle="", alpha=0.65)
        ax.errorbar(xi + offsets[2], to_m, yerr=to_s,
                    color=color, marker="v", markersize=10, capsize=4,
                    linestyle="", alpha=0.65, markerfacecolor="white")
    ax.axhline(baseline_img, ls="--", c="black", alpha=0.4, lw=1)
    ax.axhline(baseline_txt, ls=":", c="black", alpha=0.4, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(recipes_plotted, rotation=20, ha="right")
    ax.set_ylabel("AUROC  (mean across 3 splits × 3 seeds)")
    ax.set_ylim(0.50, 0.80)

    legend_handles = [
        Line2D([0], [0], marker="s", color="black", linestyle="", markersize=10,
               alpha=0.65, label="image-only AUROC"),
        Line2D([0], [0], color="black", linestyle="--", alpha=0.4, lw=1,
               label=f"image-only baseline (split-avg) = {baseline_img:.3f}"),
        Line2D([0], [0], marker="v", color="black", linestyle="", markersize=10,
               alpha=0.65, markerfacecolor="white", label="text-only AUROC"),
        Line2D([0], [0], color="black", linestyle=":", alpha=0.4, lw=1,
               label=f"text-only baseline = {baseline_txt:.3f} (dev)"),
        Line2D([0], [0], marker="^", color="black", linestyle="", markersize=10,
               label="multimodal AUROC"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=3,
              fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.30))
    ax.set_title("Per-branch forward AUROC, split-averaged "
                 "(bar = σ across dev / test_seen / test_unseen)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "02bm_image_branch_revival_with_text_merged",
          "Single-panel merge of Fig 2b for the poster. Per recipe, three "
          "markers: multimodal (▲), image-only (■), text-only (open ▽), each "
          "averaged across (3 splits × 3 seeds). Error bar = σ of the three "
          "per-split means; uniformly ≤ 0.012 AUROC so the merge is "
          "distortion-free. Dashed line = split-averaged dedicated image-only "
          f"baseline = {baseline_img:.3f} (= mean of split-matched dev / "
          "test_seen / test_unseen baselines 0.628 / 0.640 / 0.650); dotted "
          f"line = dedicated text-only baseline = {baseline_txt:.3f} (dev). "
          "Reads the branch-redistribution story in one panel: on `clean` "
          "the text branch carries the signal (text-only ≈ baseline; image-"
          "only well below baseline). `kl` lifts image-only modestly but "
          "stays below the split-averaged image baseline. `kldrop-p015` and "
          "`kldrop-p050` push image-only at or above the dedicated baseline "
          "while text-only stays within seed noise of `kl` — the defense "
          "rebalances capacity, it does not add signal. The intermediate "
          "`kldrop-p025` is co-optimal with `p015` and the image-only AUROC "
          "peak (0.668 on test_unseen, +0.018 above the split-matched "
          "baseline); it is plotted in Fig 6 only, to keep markers readable "
          "here. Use this in place of poster Fig 2 when the augmented branch "
          "story is the framing; keep 3-panel Figs 2 / 2b for the report.")


# =============================================================================
# Figure 3 — Composite escalation: ΔAUROC heatmap (recipes × composite types)
# =============================================================================

def fig_composite_escalation(perturbed_all, split: str = "test_unseen"):
    import matplotlib.pyplot as plt
    import numpy as np

    plt = _plt()
    composite_types = ("composite_2text", "composite_2image",
                       "composite_text_image", "composite_2text_2image")
    # `mixed` severity dropped per Phase 9a finding (mixed ≈ medium for every
    # recipe × composite type × split) — adding it as a third column added
    # near-duplicate cells and stretched the heatmap without story gain.
    severities_order = ("medium", "high")
    recipes = [r for r in agg.MAIN_VARIANT_ORDER if r != "clean"] + ["clean"]  # clean last

    # Build matrix: row = recipe, col = (composite_type, severity)
    cols = [(c, s) for c in composite_types for s in severities_order]
    mat = np.full((len(recipes), len(cols)), np.nan)
    for ri, recipe in enumerate(recipes):
        seeds_payloads = [_payload_for(perturbed_all, recipe, s, split) for s in (0, 1, 2)]
        seeds_payloads = [p for p in seeds_payloads if p is not None]
        if not seeds_payloads:
            continue
        for ci, (ctype, sev) in enumerate(cols):
            vals = []
            for p in seeds_payloads:
                cells = [c for c in p["cells"]
                         if c.get("attack") == ctype and c.get("severity_level") == sev]
                if cells:
                    vals.extend(c["robustness_gap"]["auroc"] for c in cells)
            if vals:
                mat[ri, ci] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-np.nanmax(np.abs(mat)),
                   vmax=np.nanmax(np.abs(mat)))
    ax.set_yticks(range(len(recipes)))
    ax.set_yticklabels(recipes)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c.replace('composite_','')}\n{s}" for c, s in cols],
                       rotation=0, fontsize=8)
    for ri in range(len(recipes)):
        for ci in range(len(cols)):
            v = mat[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > np.nanmax(np.abs(mat)) * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ΔAUROC (clean − attacked)")
    ax.set_title(f"Composite-attack escalation — {SPLIT_LABEL[split]}", fontsize=11)
    fig.tight_layout()
    _save(fig, "03_composite_escalation",
          "ΔAUROC under composite attacks (rows = recipes, columns = composite "
          "type × severity ∈ {medium, high}). Lower (cool) is more robust. "
          "Composite attacks are the strongest threat in the benchmark on "
          "test_unseen: clean-baseline `composite_2text_2image` at high severity "
          f"reaches ΔAUROC > 0.13. `kldrop-p050` (brown) is the strongest "
          "composite defender — lowest cell on every (type × severity) column "
          "on test_unseen (composite_2text med Δ = 0.035, −51 % vs clean). "
          "`kldrop-p015` (light red) is close behind at lower clean-accuracy "
          "cost. The `mixed`-severity column is omitted: Phase 9a confirmed "
          "`mixed ≈ medium` for every (recipe, composite type, split), so the "
          "column added near-duplicate cells. Single-perturbation rankings "
          "(e.g. Worst-cell text Δ) understate the realistic adversarial-user "
          "threat.")


# =============================================================================
# Figure 4 — Class-asymmetric trade-off (kldrop wins vs kl wins, by label)
# =============================================================================

def fig_class_asymmetric():
    plt = _plt()
    from collections import Counter

    # Test_unseen-only single-panel version for the poster (n=2000, the
    # largest split and the strongest asymmetry: 97 % label-0 on kldrop-p015's
    # corrections, CI [94 %, 99 %], n=131). The 3-panel dev/test_seen/unseen
    # version lives in the report appendix.
    pair_a, pair_b = "kldrop-p015", "kl"
    split = "test_unseen"
    captions = fa._load_captions(split)
    table, _ = fa._build_full_table(
        ["stage1", "augonly", "kl", "kldrop", "kldrop-p015",
         "kldrop-p050", "kllowmed"],
        [0, 1, 2], captions, split=split)
    a_wins, b_wins = fa._disagreement_examples(table, pair_a, pair_b, [0, 1, 2])
    a = Counter(e["label"] for e in a_wins)
    b = Counter(e["label"] for e in b_wins)
    a0, a1 = a.get(0, 0), a.get(1, 0)
    b0, b1 = b.get(0, 0), b.get(1, 0)
    tot_a, tot_b = a0 + a1, b0 + b1

    # ---- Horizontal stacked bars: each row = one "wins" direction; segments
    #      coloured by gold label. Bar length = total disagreements (n);
    #      coloured-segment width = within-bar class composition. The
    #      asymmetry reads directly as "this bar is mostly blue, that bar is
    #      mostly red".
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bar_h = 0.55
    y_top, y_bot = 1, 0

    # Top row: kldrop-p015 corrects vs kl  (mostly label=0)
    ax.barh(y_top, a0, bar_h, color="#1f77b4", label="label=0 (non-hate)")
    ax.barh(y_top, a1, bar_h, left=a0, color="#d62728", label="label=1 (hate)")
    # Bottom row: kl corrects vs kldrop-p015  (mostly label=1)
    ax.barh(y_bot, b0, bar_h, color="#1f77b4")
    ax.barh(y_bot, b1, bar_h, left=b0, color="#d62728")

    # In-segment percentage callouts on the dominant colour
    if tot_a:
        ax.text(a0 / 2, y_top, f"{100 * a0 / tot_a:.0f}% non-hate",
                ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
    if tot_b:
        ax.text(b0 + b1 / 2, y_bot, f"{100 * b1 / tot_b:.0f}% hate",
                ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")

    # Total-n labels at the right end of each bar
    xpad = max(tot_a, tot_b) * 0.02
    ax.text(tot_a + xpad, y_top, f"n = {tot_a}", va="center", fontsize=10)
    ax.text(tot_b + xpad, y_bot, f"n = {tot_b}", va="center", fontsize=10)

    ax.set_yticks([y_top, y_bot])
    ax.set_yticklabels(
        [f"`{pair_a}` corrects\n(vs `{pair_b}`)",
         f"`{pair_b}` corrects\n(vs `{pair_a}`)"], fontsize=10)
    ax.set_xlabel("Number of disagreement examples on test_unseen "
                  "(n=2000, 3-seed majority bucket)")
    ax.set_xlim(0, max(tot_a, tot_b) * 1.12)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"`{pair_a}` corrections target false positives — "
        f"{100 * a0 / tot_a:.0f}% non-hate (95 % CI [94, 99], n={tot_a}); "
        f"`{pair_b}` corrections target false negatives ({100 * b1 / tot_b:.0f}% hate, n={tot_b})",
        fontsize=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.tight_layout()

    _save(fig, "04_class_asymmetric_tradeoff",
          f"Per-example disagreements between `{pair_a}` and `{pair_b}` on "
          "test_unseen (n=2000, 3-seed majority bucket). Two horizontal "
          f"stacked bars: top = examples where `{pair_a}` is naturally robust "
          f"and `{pair_b}` fails; bottom = the reverse. Each bar is split by "
          "gold label (blue = label=0 non-hate, red = label=1 hate); bar "
          "length = total disagreements (n shown). The asymmetry is direct: "
          f"`{pair_a}` corrections are {100*a0/tot_a:.0f} % non-hate "
          f"(95 % CI [94, 99], n={tot_a}) — the recipe acts as a false-"
          f"positive veto. `{pair_b}` corrections are {100*b1/tot_b:.0f} % "
          f"hate (n={tot_b}) — the opposite class. Mechanism: the kldrop-"
          "p015-revived image branch (see Fig 2 / 6) vetoes the text branch "
          "when the caption looks superficially hateful but the image does "
          "not support it. Strongest single deployment-conditional finding "
          "of the project. Dev / test_seen panels (smaller n, same "
          "direction) live in the report appendix.")


# =============================================================================
# Figure 5 — Per-severity text-family ΔAUROC curves
# =============================================================================

def fig_severity_curves(perturbed_all):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    severities = ("low", "medium", "high")
    for ax, split in zip(axes, SPLITS):
        for recipe in agg.MAIN_VARIANT_ORDER:
            seeds_payloads = [_payload_for(perturbed_all, recipe, s, split) for s in (0, 1, 2)]
            seeds_payloads = [p for p in seeds_payloads if p is not None]
            if not seeds_payloads:
                continue
            means = []
            stds = []
            for sev in severities:
                vals_per_seed = []
                for p in seeds_payloads:
                    text_gaps = [c["robustness_gap"]["auroc"] for c in p["cells"]
                                 if agg._attack_family(c["attack"]) == "text"
                                 and c["severity_level"] == sev]
                    if text_gaps:
                        vals_per_seed.append(statistics.mean(text_gaps))
                if vals_per_seed:
                    means.append(statistics.mean(vals_per_seed))
                    stds.append(statistics.pstdev(vals_per_seed) if len(vals_per_seed) > 1 else 0.0)
                else:
                    means.append(float("nan"))
                    stds.append(0.0)
            ax.errorbar(range(3), means, yerr=stds,
                        marker="o", markersize=6, capsize=3,
                        color=RECIPE_COLOR.get(recipe, "k"), label=recipe)
        ax.set_xticks(range(3))
        ax.set_xticklabels(severities)
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
    axes[0].set_ylabel("Mean text-family ΔAUROC ↓")
    axes[0].set_xlabel("Severity")
    axes[1].set_xlabel("Severity (medium = realistic, high = stress)")
    axes[2].set_xlabel("Severity")
    axes[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Per-severity text-attack ΔAUROC — monotone; kldrop-p050 lowest at every severity", fontsize=11)
    fig.tight_layout()
    _save(fig, "05_severity_curves",
          "Mean ΔAUROC for the text-attack family per severity, per recipe. "
          "Lower is better. Monotone low ≤ medium ≤ high for every (recipe, "
          "split) row (sanity gate). `kldrop-p050` (brown) is the lowest at "
          "every severity on every split; `kldrop-p015` (light red) is close "
          "behind at zero clean-accuracy cost vs `kl`. The medium column is "
          "the internet-realistic threat headline.")


# =============================================================================
# Figure 6 — Modality-dropout-rate sweep
# =============================================================================

def fig_dropout_sweep(perturbed_all, modality_all):
    plt = _plt()
    # Phase 9c finer sweep: 7-point grid p ∈ {0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50}.
    # p=0 is the kl recipe (no dropout); p=0.30 is the original kldrop (now Pareto-dominated).
    # Plotted on test_unseen — the most naturalistic held-out split, where image-
    # only AUROC peaks at 0.668 at p=0.25 (vs 0.644 on dev). Hyperparameter `p`
    # was selected on dev in Phase 9c; this figure is post-hoc visualisation of
    # the same evaluation on the held-out test_unseen split.
    SWEEP_SPLIT = "test_unseen"
    p_to_recipe = [
        (0.00, "kl"),
        (0.10, "kldrop-p010"),
        (0.15, "kldrop-p015"),
        (0.20, "kldrop-p020"),
        (0.25, "kldrop-p025"),
        (0.30, "kldrop"),
        (0.50, "kldrop-p050"),
    ]
    fig, ax1 = plt.subplots(figsize=(5.8, 5.2))

    have_p015 = any(_payload_for(perturbed_all, "kldrop-p015", s, SWEEP_SPLIT) for s in (0, 1, 2))
    if not have_p015:
        print(f"  fig_dropout_sweep: kldrop-p015 data on {SWEEP_SPLIT} not yet on disk — skipping")
        plt.close(fig)
        return

    ps = []
    img_means = []
    img_stds = []
    clean_means = []
    clean_stds = []
    for p_val, recipe in p_to_recipe:
        ma = []
        for seed in (0, 1, 2):
            if recipe == "kl":
                k = f"robust-kl-seed{seed}"
            elif recipe.startswith("kldrop"):
                k = f"robust-{recipe}-seed{seed}"
            payload = modality_all.get((SWEEP_SPLIT, k))
            if payload is not None:
                ma.append(payload["image_only"]["at_best_threshold"]["auroc"])
        cl = []
        for seed in (0, 1, 2):
            if recipe == "kl":
                k = f"robust-kl-seed{seed}"
            else:
                k = f"robust-{recipe}-seed{seed}"
            payload = perturbed_all.get((SWEEP_SPLIT, k))
            if payload is not None:
                cl.append(payload["clean"]["at_best_threshold"]["auroc"])
        if ma and cl:
            ps.append(p_val)
            img_means.append(statistics.mean(ma))
            img_stds.append(statistics.pstdev(ma) if len(ma) > 1 else 0)
            clean_means.append(statistics.mean(cl))
            clean_stds.append(statistics.pstdev(cl) if len(cl) > 1 else 0)

    if not ps:
        plt.close(fig); return

    baseline_img = BASELINE_IMG_BY_SPLIT[SWEEP_SPLIT]  # 0.650 on test_unseen
    ax1.errorbar(ps, clean_means, yerr=clean_stds, marker="s", capsize=4,
                 color="#1f77b4", label="clean (multimodal) AUROC")
    ax1.errorbar(ps, img_means, yerr=img_stds, marker="o", capsize=4,
                 color="#d62728", label="image-only AUROC")
    ax1.axhline(baseline_img, ls=":", c="grey", alpha=0.7, lw=1.2,
                label=f"dedicated image-only baseline ({baseline_img:.3f})")
    ax1.set_xlabel("modality_dropout_text  p")
    ax1.set_ylabel("AUROC ↑")

    ax1.set_xticks(ps)
    ax1.set_title("Modality-dropout-rate sweep (test_unseen, 3-seed mean ± σ)", fontsize=11)
    ax1.legend(loc="center right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    _save(fig, "06_dropout_sweep",
          "Modality-dropout-rate sweep on test_unseen — 7 points: p ∈ {0, "
          "0.10, 0.15, 0.20, 0.25, 0.30, 0.50} (Phase 9c). x-axis = "
          "`modality_dropout_text` (p=0.00 is the `kl` recipe with no dropout; "
          "p=0.30 is the legacy `kldrop`, now Pareto-dominated). Both series "
          "share a single AUROC y-axis. Image-only AUROC (red ●) rises "
          "monotonically with p and peaks at 0.668 at p=0.25, +0.018 above "
          f"the split-matched dedicated image-only baseline of {baseline_img:.3f} "
          "(grey dotted, from `perturbed-baseline-image-test-unseen-seed0`). "
          "At p=0 (`kl`, no dropout) image-only AUROC is 0.628 — 0.022 below "
          "the dedicated baseline, confirming the image branch is under-"
          "trained without modality dropout. Clean (multimodal) AUROC (blue "
          "■) stays inside seed noise through p ≤ 0.25 then drops by p = "
          "0.30 — the 'first 25 % is free' window. Hyperparameter `p` was "
          "selected on dev in Phase 9c; this figure is the post-hoc "
          "visualisation of the same evaluation on the most naturalistic "
          "held-out split. Shape reproduces on dev and test_seen (see "
          "report appendix).")


# =============================================================================
# Figure 7 — Combined Fig 2bm (per-branch AUROC) + Fig 6 (dropout sweep)
#            Side-by-side single image for the poster (shared AUROC y-axis).
# =============================================================================

def fig_combined_revival_and_sweep(modality_all, perturbed_all):
    """One-image side-by-side composition of Fig 2bm and Fig 6 for the poster.

    Both panels share the AUROC y-axis. The dedicated image-only baseline is
    split-specific (0.628 / 0.640 / 0.650 on dev / test_seen / test_unseen);
    the left panel draws the split-averaged value ≈ 0.639, the right panel
    draws the test_unseen value 0.650. The two horizontal references differ
    by ≈ 0.011 AUROC — small enough that the figure still reads as a unified
    composition while remaining methodologically correct per panel.
    """
    plt = _plt()
    from matplotlib.lines import Line2D
    import numpy as np

    baseline_img_left = BASELINE_IMG_SPLIT_AVG       # ≈ 0.639 for split-averaged panel
    baseline_img_right = BASELINE_IMG_BY_SPLIT["test_unseen"]  # 0.650
    baseline_txt = BASELINE_TXT_DEV                  # 0.632 (dev only)

    # ---------- LEFT PANEL DATA: per-recipe split-averaged AUROC × 3 branches
    def _per_split_mean(variant: str, branch: str, split: str):
        vals = []
        for seed in (0, 1, 2):
            if variant == "clean":
                p = modality_all.get((split, f"stage1-seed{seed}"))
            else:
                p = (modality_all.get((split, f"robust-{variant}-seed{seed}"))
                     or modality_all.get((split, f"train-robust-{variant}-seed{seed}")))
            if p is not None:
                vals.append(p[branch]["at_best_threshold"]["auroc"])
        return statistics.mean(vals) if vals else None

    def _merge(variant, branch):
        per_split = [v for split in SPLITS
                     if (v := _per_split_mean(variant, branch, split)) is not None]
        if not per_split:
            return None
        return (statistics.mean(per_split),
                statistics.pstdev(per_split) if len(per_split) > 1 else 0.0)

    recipes_plotted: list[str] = []
    mm: list[tuple[float, float]] = []
    io: list[tuple[float, float]] = []
    to: list[tuple[float, float]] = []
    for variant in agg.MAIN_VARIANT_ORDER:
        a, b, c = _merge(variant, "multimodal"), _merge(variant, "image_only"), _merge(variant, "text_only")
        if a is None or b is None or c is None:
            continue
        recipes_plotted.append(variant)
        mm.append(a); io.append(b); to.append(c)

    # ---------- RIGHT PANEL DATA: dropout-rate sweep on test_unseen
    SWEEP_SPLIT = "test_unseen"
    p_to_recipe = [
        (0.00, "kl"), (0.10, "kldrop-p010"), (0.15, "kldrop-p015"),
        (0.20, "kldrop-p020"), (0.25, "kldrop-p025"), (0.30, "kldrop"),
        (0.50, "kldrop-p050"),
    ]
    have_p015 = any(_payload_for(perturbed_all, "kldrop-p015", s, SWEEP_SPLIT) for s in (0, 1, 2))
    if not (recipes_plotted and have_p015):
        print("  fig_combined: required data not on disk — skipping")
        return

    ps, img_means, img_stds, clean_means, clean_stds = [], [], [], [], []
    for p_val, recipe in p_to_recipe:
        ma = []
        for seed in (0, 1, 2):
            k = f"robust-{recipe}-seed{seed}" if recipe != "kl" else f"robust-kl-seed{seed}"
            payload = modality_all.get((SWEEP_SPLIT, k))
            if payload is not None:
                ma.append(payload["image_only"]["at_best_threshold"]["auroc"])
        cl = []
        for seed in (0, 1, 2):
            k = f"robust-{recipe}-seed{seed}" if recipe != "kl" else f"robust-kl-seed{seed}"
            payload = perturbed_all.get((SWEEP_SPLIT, k))
            if payload is not None:
                cl.append(payload["clean"]["at_best_threshold"]["auroc"])
        if ma and cl:
            ps.append(p_val)
            img_means.append(statistics.mean(ma))
            img_stds.append(statistics.pstdev(ma) if len(ma) > 1 else 0)
            clean_means.append(statistics.mean(cl))
            clean_stds.append(statistics.pstdev(cl) if len(cl) > 1 else 0)

    # ---------- FIGURE
    fig, (ax_L, ax_R) = plt.subplots(
        1, 2, figsize=(12.0, 5.9), sharey=True,
        gridspec_kw={"wspace": 0.06, "width_ratios": [1.05, 1.0]},
    )

    # ---------- LEFT PANEL: per-branch AUROC per recipe
    x = np.arange(len(recipes_plotted))
    offsets = (-0.18, 0.0, 0.18)
    for xi, recipe, (mm_m, mm_s), (io_m, io_s), (to_m, to_s) in zip(
            x, recipes_plotted, mm, io, to):
        color = RECIPE_COLOR.get(recipe, "k")
        ax_L.errorbar(xi + offsets[0], mm_m, yerr=mm_s, color=color,
                      marker="^", markersize=9, capsize=3, linestyle="")
        ax_L.errorbar(xi + offsets[1], io_m, yerr=io_s, color=color,
                      marker="s", markersize=9, capsize=3, linestyle="",
                      alpha=0.65)
        ax_L.errorbar(xi + offsets[2], to_m, yerr=to_s, color=color,
                      marker="v", markersize=9, capsize=3, linestyle="",
                      alpha=0.65, markerfacecolor="white")
    ax_L.axhline(baseline_img_left, ls="--", c="black", alpha=0.45, lw=1)
    ax_L.axhline(baseline_txt, ls=":", c="black", alpha=0.45, lw=1)
    ax_L.set_xticks(x)
    ax_L.set_xticklabels(recipes_plotted, rotation=20, ha="right")
    ax_L.set_ylabel("AUROC")
    ax_L.set_ylim(0.575, 0.770)
    ax_L.set_title("(a) Per-branch AUROC, split-averaged",
                   fontsize=10)

    left_legend = [
        Line2D([0], [0], marker="^", color="black", linestyle="", markersize=9,
               label="multimodal"),
        Line2D([0], [0], marker="s", color="black", linestyle="", markersize=9,
               alpha=0.65, label="image-only"),
        Line2D([0], [0], marker="v", color="black", linestyle="", markersize=9,
               alpha=0.65, markerfacecolor="white", label="text-only"),
        Line2D([0], [0], color="black", linestyle=":", alpha=0.45, lw=1,
               label=f"text-only baseline ({baseline_txt:.3f})"),
        Line2D([0], [0], color="black", linestyle="--", alpha=0.45, lw=1,
               label=f"image baseline ({baseline_img_left:.3f}, split-avg)"),
    ]
    ax_L.legend(handles=left_legend, loc="upper left", fontsize=8,
                framealpha=0.92, ncol=2, frameon=False,
                bbox_to_anchor=(0.0, -0.13))

    # ---------- RIGHT PANEL: dropout-rate sweep
    ax_R.errorbar(ps, clean_means, yerr=clean_stds, marker="s", capsize=4,
                  color="#1f77b4", label="clean (multimodal) AUROC")
    ax_R.errorbar(ps, img_means, yerr=img_stds, marker="o", capsize=4,
                  color="#d62728", label="image-only AUROC")
    ax_R.axhline(baseline_img_right, ls="--", c="black", alpha=0.45, lw=1,
                 label=f"image baseline ({baseline_img_right:.3f})")
    ax_R.set_xticks(ps)
    ax_R.set_xlabel("modality_dropout_text  p")
    ax_R.set_title("(b) Dropout-rate sweep (test_unseen)",
                   fontsize=10)
    ax_R.legend(loc="upper right", fontsize=8, framealpha=0.92,
                ncol=1, frameon=False, bbox_to_anchor=(1.0, -0.13))

    fig.suptitle("Image-branch revival: redistribution mechanism (a) "
                 "and dropout-rate dose-response (b)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "07_combined_revival_and_sweep",
          "Side-by-side merge of Fig 2bm and Fig 6 for the poster — one image, "
          "two panels, shared AUROC y-axis. Dedicated image-only baseline is "
          "split-specific (0.628 / 0.640 / 0.650 on dev / test_seen / "
          "test_unseen); (a) draws the split-averaged value ≈ 0.639, (b) "
          "draws the test_unseen value 0.650 — the two dashed references "
          "differ by ≈ 0.011 AUROC. (a) Per-recipe per-branch forward AUROC, "
          "averaged across 3 splits × 3 seeds (error bar = σ of the 3 per-"
          "split means; uniformly ≤ 0.012 AUROC). `clean` carries signal only "
          "through text (text-only ≈ baseline; image-only well below); `kl` "
          "lifts image-only modestly but remains below the split-averaged "
          "image baseline; `kldrop-p015` and `kldrop-p050` push image-only "
          "to or above the dedicated baseline while text-only stays within "
          "seed noise of `kl`. (b) Modality-dropout-rate sweep on test_unseen "
          "(Phase 9c): 7 points p ∈ {0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50}. "
          "Image-only AUROC (red ●) rises monotonically and peaks at 0.668 "
          "at p = 0.25 — +0.018 above the split-matched dedicated image-only "
          "baseline of 0.650 (dashed). At p = 0 (`kl`) image-only is 0.628, "
          "−0.022 below the baseline — confirming the image branch is under-"
          "trained without modality dropout. Clean (multimodal) AUROC (blue "
          "■) stays inside seed noise through p ≤ 0.25 then drops by p = "
          "0.30 — the 'first 25 % is free' window. Hyperparameter `p` was "
          "chosen on dev in Phase 9c; (b) is the post-hoc visualisation of "
          "the same evaluation on the most naturalistic held-out split. Use "
          "this composite in place of separate Fig 2bm + Fig 6 when poster "
          "real estate allows one wide slot.")


# =============================================================================
# main
# =============================================================================

def main() -> int:
    print(f"Loading aggregator data...")
    perturbed_all = agg._load_perturbed()
    whitebox_all = agg._load_whitebox()
    modality_all = agg._load_modality_ablation()
    print(f"  perturbed: {len(perturbed_all)} (split, key) pairs")
    print(f"  whitebox:  {len(whitebox_all)} pairs")
    print(f"  modality:  {len(modality_all)} pairs")
    print()

    print("Figure 1: headline_pareto")
    fig_headline_pareto(perturbed_all)
    print("Figure 2: image_branch_revival")
    fig_image_branch_revival(modality_all)
    print("Figure 2m: image_branch_revival_merged (single-panel; poster-side)")
    fig_image_branch_revival_merged(modality_all)
    print("Figure 2b: image_branch_revival_with_text (augmented; report-side)")
    fig_image_branch_revival_with_text(modality_all)
    print("Figure 2bm: image_branch_revival_with_text_merged (single-panel; poster-side)")
    fig_image_branch_revival_with_text_merged(modality_all)
    print("Figure 3: composite_escalation (test_unseen)")
    fig_composite_escalation(perturbed_all, split="test_unseen")
    print("Figure 4: class_asymmetric_tradeoff")
    fig_class_asymmetric()
    print("Figure 5: severity_curves")
    fig_severity_curves(perturbed_all)
    print("Figure 6: dropout_sweep (skipped until kldrop-p015/p050 land)")
    fig_dropout_sweep(perturbed_all, modality_all)
    print("Figure 7: combined Fig 2bm + Fig 6 side-by-side (poster wide slot)")
    fig_combined_revival_and_sweep(modality_all, perturbed_all)

    print()
    print(f"All figures in {OUT_DIR.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
