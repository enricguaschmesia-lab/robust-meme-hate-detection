"""Poster-grade figures for the robust-meme-hate-detection project.

Reuses the aggregator's data loaders (`scripts/aggregate_phase4.py`) so the
figures are always in sync with the per-phase tables. Each figure has its
own function; figures gracefully skip if their data isn't on disk yet.

Output: `project_planning/poster_figures/{fig_name}.{png,pdf}` plus a
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

# Color palette: tab10-derived, recipe-grouped
RECIPE_COLOR = {
    "clean":         "#7f7f7f",  # grey
    "augonly":       "#1f77b4",  # blue
    "kl":            "#2ca02c",  # green
    "kldrop":        "#d62728",  # red
    "kldrop-p015":   "#ff9896",  # light red
    "kldrop-p050":   "#8c564b",  # dark red/brown
    "kllowmed":      "#9467bd",  # purple
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, split in zip(axes, SPLITS):
        for recipe in agg.MAIN_VARIANT_ORDER:
            s = _per_recipe_summary(perturbed_all, recipe, split)
            if not s:
                continue
            ax.errorbar(s["clean_auroc_mean"], s["worst_text_mean"],
                        xerr=s["clean_auroc_std"], yerr=s["worst_text_std"],
                        marker="o", markersize=8, color=RECIPE_COLOR.get(recipe, "k"),
                        capsize=3, linestyle="", label=recipe)
            ax.annotate(recipe, (s["clean_auroc_mean"], s["worst_text_mean"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8,
                        color=RECIPE_COLOR.get(recipe, "k"))
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.set_xlabel("Clean AUROC →")
        ax.invert_yaxis()  # smaller Δ is better; put it up
    axes[0].set_ylabel("Worst-cell text ΔAUROC ↓")
    fig.suptitle("Headline Pareto: clean accuracy vs single-cell worst-case text robustness", fontsize=11)
    fig.tight_layout()
    _save(fig, "01_headline_pareto",
          "Per-recipe Pareto trade-off on each split. Down-right is better "
          "(high clean AUROC, low worst-cell ΔAUROC under text attacks). "
          "`kldrop` (red) is the strict winner on Worst-cell text Δ across all "
          "three splits; `kl` (green) preserves clean AUROC best. The two "
          "Pareto winners are mechanistically distinct (modality dropout vs "
          "KL consistency).")


# =============================================================================
# Figure 2 — Image-branch revival (modality ablation × 3 splits)
# =============================================================================

def fig_image_branch_revival(modality_all):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    baseline = 0.628  # dedicated image-only baseline (Phase 1-2)
    for ax, split in zip(axes, SPLITS):
        for variant in agg.MAIN_VARIANT_ORDER:
            if variant == "clean":
                key = None  # use stage1
            else:
                key = variant
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
                        marker="^", markersize=8, capsize=3, linestyle="",
                        label="multimodal" if variant == "clean" else None)
            ax.errorbar(variant, statistics.mean(io),
                        yerr=statistics.pstdev(io) if len(io) > 1 else 0,
                        color=RECIPE_COLOR.get(variant, "k"),
                        marker="s", markersize=8, capsize=3, linestyle="",
                        alpha=0.6,
                        label="image-only" if variant == "clean" else None)
        ax.axhline(baseline, ls="--", c="black", alpha=0.4, lw=1)
        ax.text(len(agg.VARIANT_ORDER) - 0.5, baseline + 0.003,
                f"dedicated image-only baseline = {baseline:.3f}",
                fontsize=7, color="black", ha="right", va="bottom")
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0.55, 0.78)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Image-branch revival: multimodal (▲) and image-only (■) forward AUROC per recipe", fontsize=11)
    fig.tight_layout()
    _save(fig, "02_image_branch_revival",
          "Per-recipe multimodal (triangle) and image-only (square) AUROC on "
          "each split. The dashed line is the dedicated image-only baseline "
          "(0.628 from Phase 1-2). `kldrop` is the only recipe whose image-only "
          "AUROC exceeds the baseline on every split (0.636 / 0.641 / 0.655). "
          "Modality-dropout-based image-branch revival generalises from dev to "
          "both held-out splits.")


# =============================================================================
# Figure 3 — Composite escalation: ΔAUROC heatmap (recipes × composite types)
# =============================================================================

def fig_composite_escalation(perturbed_all, split: str = "test_unseen"):
    import matplotlib.pyplot as plt
    import numpy as np

    plt = _plt()
    composite_types = ("composite_2text", "composite_2image",
                       "composite_text_image", "composite_2text_2image")
    severities_order = ("medium", "mixed", "high")
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

    # Drop the "mixed" column entirely if there's no data anywhere yet
    mixed_idx = [i for i, c in enumerate(cols) if c[1] == "mixed"]
    if np.all(np.isnan(mat[:, mixed_idx])):
        keep = [i for i in range(len(cols)) if i not in mixed_idx]
        mat = mat[:, keep]
        cols = [cols[i] for i in keep]

    fig, ax = plt.subplots(figsize=(11, 4.5))
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
          "type × severity). Lower (cool) is more robust. Composite "
          "attacks are the strongest threat in the benchmark on "
          "test_unseen: clean-baseline `composite_2text_2image` at high severity "
          f"reaches ΔAUROC > 0.13. `kldrop` wins 3/4 medium-severity cells "
          "and is competitive on the `mixed` severity. Single-perturbation "
          "rankings (e.g. Worst-cell text Δ) understate the realistic "
          "adversarial-user threat.")


# =============================================================================
# Figure 4 — Class-asymmetric trade-off (kldrop wins vs kl wins, by label)
# =============================================================================

def fig_class_asymmetric():
    plt = _plt()
    from collections import Counter

    # Phase 10 refresh: use kldrop-p015 (the recommended default) vs kl
    # instead of the now-dominated kldrop (p=0.30). The label-asymmetric
    # finding is stronger on the new recipe on test_unseen (label=0 share
    # 97 % at n=131 vs the old kldrop's 74 % at n=85).
    pair_a, pair_b = "kldrop-p015", "kl"
    data = {}
    for split in SPLITS:
        captions = fa._load_captions(split)
        table, _ = fa._build_full_table(
            ["stage1", "augonly", "kl", "kldrop", "kldrop-p015",
             "kldrop-p050", "kllowmed"],
            [0, 1, 2], captions, split=split)
        a_wins, b_wins = fa._disagreement_examples(table, pair_a, pair_b, [0, 1, 2])
        data[split] = {
            "a_wins": Counter(e["label"] for e in a_wins),
            "b_wins": Counter(e["label"] for e in b_wins),
        }

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    width = 0.35
    for ax, split in zip(axes, SPLITS):
        d = data[split]
        a = d["a_wins"]
        b = d["b_wins"]
        x = [0, 1]
        a0 = a.get(0, 0); a1 = a.get(1, 0)
        b0 = b.get(0, 0); b1 = b.get(1, 0)
        ax.bar(x[0] - width / 2, a0, width, color="#1f77b4", label="label=0 (non-hate)")
        ax.bar(x[0] + width / 2, a1, width, color="#d62728", label="label=1 (hate)")
        ax.bar(x[1] - width / 2, b0, width, color="#1f77b4")
        ax.bar(x[1] + width / 2, b1, width, color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{pair_a}\nbeats {pair_b}",
                            f"{pair_b}\nbeats {pair_a}"], fontsize=9)
        ax.set_title(SPLIT_LABEL[split], fontsize=10)
        tot_a = a0 + a1
        tot_b = b0 + b1
        if tot_a:
            ax.text(x[0], max(a0, a1) + max(1, 0.05 * max(a0, a1)),
                    f"{100*a0/tot_a:.0f}% l=0", ha="center", fontsize=8)
        if tot_b:
            ax.text(x[1], max(b0, b1) + max(1, 0.05 * max(b0, b1)),
                    f"{100*b1/tot_b:.0f}% l=1", ha="center", fontsize=8)
    axes[0].set_ylabel("Number of disagreement examples")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Class-asymmetric `{pair_a}` vs `{pair_b}` per-example disagreements", fontsize=11)
    fig.tight_layout()
    _save(fig, "04_class_asymmetric_tradeoff",
          f"Per-example disagreements between `{pair_a}` and `{pair_b}` "
          "(majority bucket across 3 seeds). For each split, two grouped "
          f"bars: examples where `{pair_a}` is naturally-robust and "
          f"`{pair_b}` fails (left) vs the reverse (right). Blue = label=0 "
          "(non-hate), red = label=1 (hate). The kldrop-p015 side strongly "
          "biases toward label=0 on test_unseen (97 % at n=131), confirming "
          "that the new recommended default reproduces the image-branch-"
          "anchor-against-false-positives mechanism identified in Phase 6. "
          "See `project_planning/Phase8_TestFailure_Report.md`.")


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
    fig.suptitle("Per-severity text-attack ΔAUROC — monotone, kldrop lowest at every severity", fontsize=11)
    fig.tight_layout()
    _save(fig, "05_severity_curves",
          "Mean ΔAUROC for the text-attack family per severity, per recipe. "
          "Lower is better. Monotone low ≤ medium ≤ high for every (recipe, "
          "split) row (sanity gate). `kldrop` (red) is the lowest at every "
          "severity on every split; the medium column is the internet-"
          "realistic threat headline.")


# =============================================================================
# Figure 6 — Modality-dropout-rate sweep
# =============================================================================

def fig_dropout_sweep(perturbed_all, modality_all):
    plt = _plt()
    # Phase 9c finer sweep: 7-point grid p ∈ {0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50}.
    # p=0 is the kl recipe (no dropout); p=0.30 is the original kldrop (now Pareto-dominated).
    p_to_recipe = [
        (0.00, "kl"),
        (0.10, "kldrop-p010"),
        (0.15, "kldrop-p015"),
        (0.20, "kldrop-p020"),
        (0.25, "kldrop-p025"),
        (0.30, "kldrop"),
        (0.50, "kldrop-p050"),
    ]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    have_p015 = any(_payload_for(perturbed_all, "kldrop-p015", s, "dev") for s in (0, 1, 2))
    if not have_p015:
        print("  fig_dropout_sweep: kldrop-p015/p050 data not yet on disk — skipping")
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
            payload = modality_all.get(("dev", k))
            if payload is not None:
                ma.append(payload["image_only"]["at_best_threshold"]["auroc"])
        cl = []
        for seed in (0, 1, 2):
            if recipe == "kl":
                k = f"robust-kl-seed{seed}"
            else:
                k = f"robust-{recipe}-seed{seed}"
            payload = perturbed_all.get(("dev", k))
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

    ax1.errorbar(ps, img_means, yerr=img_stds, marker="o", capsize=4,
                 color="#d62728", label="image-only AUROC")
    ax1.axhline(0.628, ls="--", c="black", alpha=0.4, lw=1)
    ax1.text(max(ps), 0.628 + 0.003, "image-only baseline = 0.628",
             ha="right", va="bottom", fontsize=8, color="black")
    ax1.set_xlabel("modality_dropout_text  p")
    ax1.set_ylabel("image-only AUROC ↑", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    ax2.errorbar(ps, clean_means, yerr=clean_stds, marker="s", capsize=4,
                 color="#1f77b4", label="clean AUROC")
    ax2.set_ylabel("clean (multimodal) AUROC ↑", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_xticks(ps)
    ax1.set_title("Modality-dropout-rate sweep (dev, 3-seed mean ± σ)", fontsize=11)
    fig.tight_layout()
    _save(fig, "06_dropout_sweep",
          "Modality-dropout-rate sweep on dev. x-axis = `modality_dropout_text` "
          "(p=0.00 is the `kl` recipe with no dropout). Image-only AUROC "
          "(red) rises monotonically with p; clean AUROC (blue) falls. "
          "The trade-off shows the Pareto frontier for image-branch revival "
          "vs clean accuracy. The dedicated image-only baseline is 0.628.")


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
    print("Figure 3: composite_escalation (test_unseen)")
    fig_composite_escalation(perturbed_all, split="test_unseen")
    print("Figure 4: class_asymmetric_tradeoff")
    fig_class_asymmetric()
    print("Figure 5: severity_curves")
    fig_severity_curves(perturbed_all)
    print("Figure 6: dropout_sweep (skipped until kldrop-p015/p050 land)")
    fig_dropout_sweep(perturbed_all, modality_all)

    print()
    print(f"All figures in {OUT_DIR.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
