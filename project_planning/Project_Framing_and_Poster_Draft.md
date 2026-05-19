# Project Framing & A0 Poster Draft

> Source for project framing: `Project_Info.md`, `Literature_Review_and_Project_Ideas.md`, `Final_Project_Proposals.md`, `ProjectIdeaHenrik*.md`, `model_architecture.md`, `Phase{1-2,3_4,5,6,7,8,9a,9b}_*_Report.md`, `Handoff_2026-05-19-final.md`.
> Source for poster geometry: `EE559_A0_poster_template.svg` (A0 portrait, viewBox 3178.58 × 4493.86; two-column layout with two highlighted panels (#efefef) reserved for Motivation and Take-away).
>
> Henrik owns the white-box adversarial side (PGD/FGSM attacks + adversarial training). All such slots are marked `[[HENRIK: …]]`.

---

## 1. Project framing

### 1.1 One-sentence pitch

> Safer online spaces need hate-meme detectors that stay reliable when a meme is **obfuscated by an everyday user** *and* when it is **perturbed by a white-box adversary**; we audit a CLIP-fusion classifier on Hateful Memes under both threat models and show that a tiny pair of training-time interventions — KL-consistency + text-modality dropout at p ∈ [0.15, 0.25] — closes most of the naturalistic gap at **zero clean-accuracy cost**, while `[[HENRIK: adversarial training]]` closes the white-box gap.

### 1.2 Working title

**Trustworthy Multimodal Hate-Meme Detection:
A Robustness Audit of CLIP-Fusion Classifiers Under Naturalistic and Adversarial Perturbations**

### 1.3 Narrative spine

A multimodal hate detector becomes *trustworthy* only when it passes two complementary robustness tests:

1. **The user test** — does it survive everyday obfuscations such as typos, leetspeak, JPEG compression, blur, brightness shifts, and combinations a frustrated user makes to bypass moderation?  *(naturalistic — Enric)*
2. **The attacker test** — does it survive a malicious actor with model access who crafts small, imperceptible image perturbations?  *(adversarial — `[[HENRIK]]`)*

A single classifier, two threat models, two parallel "audit + defense" loops. One classifier, one figure of merit (ΔAUROC clean → attacked), two teammates.

### 1.4 Why this framing fits the course

| EE-559 requirement (`Project_Info.md`) | How the framing satisfies it |
|---|---|
| "Foster healthier online interactions by automatically identifying hate speech" | Anchors on the *deployment-trust* question — *when* can the classifier be relied on. |
| "Prioritize accuracy and context comprehension" | Multimodal classifier whose dual-branch reasoning is interrogated and improved (modality-ablation + image-branch revival). |
| "Differentiate harmful hate speech from legitimate critical discourse or satire" | The class-asymmetric finding (97 % of defense-induced prediction changes are on label = 0) is exactly *avoiding over-flagging benign content*. |
| Multimodal: text, image, memes | Hateful Memes is the canonical compositional benchmark; both modalities are perturbed and studied. |
| Group mini-project, ~2 months | Two co-equal threat-model sides → two teammates → coherent single paper. |

### 1.5 Contributions (3-bullet form for abstract / intro)

1. A controlled multi-modal, multi-severity **perturbation benchmark** on Hateful Memes: 57 single-cell + 16 composite cells (incl. realistic mixed-severity) on text and image modalities, evaluated on dev (n=500), test_seen (n=1000) and test_unseen (n=2000).
2. A **lightweight training-time recipe** (KL-consistency between clean and perturbed views + text-modality dropout, p ∈ [0.15, 0.25]) that revives the under-used image branch (image-only AUROC 0.628 → 0.668 on test_unseen) at **zero clean-accuracy cost**, with a deployment-relevant **class-asymmetric mechanism** (97 % label = 0 disagreements, bootstrap 95 % CI 94 – 99 %, n = 131).
3. `[[HENRIK]]` A **white-box ε-curve audit** (FGSM/PGD at ε ∈ {1, 2, 4, 8}/255) and a **PGD adversarial-training defense** that closes the ε ≥ 2/255 gap left open by the naturalistic recipe.

### 1.6 What we are explicitly *not* claiming

- Not SOTA on Hateful Memes. The contribution is a robustness audit + recipe, not a leaderboard push.
- Not a deployable moderation system. A research prototype; no human-subject study.
- Not generalisable beyond English-language memes and a single CLIP backbone.

---

## 2. A0 poster draft (against `EE559_A0_poster_template.svg`)

### 2.1 Template anatomy

A0 portrait, viewBox `3178.58 × 4493.86`. Two-column body. Two highlight panels (light-grey background `#efefef`) are reserved for **Motivation** (top-left) and **Take-away** (lower-right). One large grey rectangle inside the lower-left column (`#b7b7b7`) is the **hero-figure** slot.

| Slot | x range | y range | Size (w × h) | Header bar (y) | Fill | Intended use |
|---|---|---|---|---|---|---|
| Title banner | full width | 111 – 245 | 3081 × 134 | – | transparent | Project title |
| Subtitle 1 | full width | 246 – 337 | 3081 × 91 | – | transparent | Subtitle / one-line pitch |
| Subtitle 2 | full width | 325 – 403 | 3081 × 77 | – | transparent | Authors + affiliations |
| EPFL logo | top-left | 114 – 215 | 374 × 100 | – | transparent | EPFL / EE-559 logo |
| Lab/sponsor logo | top-right | 118 – 252 | 318 × 134 | – | transparent | Course / lab logo |
| **Panel A** (left col) | 97 – 1538 | 483 – 1232 | 1440 × 749 | 487 – 595 | **#efefef** highlight | Motivation |
| **Panel B** (left col) | 105 – 1523 | 1281 – 2241 | 1418 × 960 | 1281 – 1389 | transparent | Model & two-threat framework |
| **Panel C** (left col, hero) | 111 – 1524 | 2257 – 3500 | 1413 × 1243 | 2257 – 2364 | inner grey 119 – 1530 × 2611 – 3469 (`#b7b7b7`) | Naturalistic results + Pareto hero figure |
| **Panel D** (left col) | 111 – 1524 | 3544 – 4289 | 1413 × 745 | 3544 – 3651 | transparent | Adversarial results `[[HENRIK]]` |
| **Panel E** (right col) | 1652 – 3069 | 496 – 2559 | 1417 × 2063 | 496 – 604 | transparent | Methods deep-dive (tallest panel) |
| **Panel F** (right col) | 1652 – 3069 | 2612 – 3225 | 1417 × 613 | 2612 – 2720 | transparent | Class-asymmetric mechanism |
| **Panel G** (right col) | 1641 – 3081 | 3285 – 3971 | 1440 × 686 | 3288 – 3413 | **#efefef** highlight | Take-away |
| References / acks | 1595 – 3092 | 4016 – 4420 | 1497 × 403 | – | transparent | References + acknowledgments |
| Code/data strip | 109 – 1069 | 4315 – 4400 | 960 × 84 | – | transparent | Code + dataset link |

Reading flow assumed by the layout: full left column top-to-bottom (A → B → C → D), then full right column top-to-bottom (E → F → G → refs). A reader who only glances catches the two highlights (Motivation A, Take-away G) and the hero figure inside C — that triplet alone must transmit the project.

### 2.2 Title + author banner

```
TITLE (banner, y 111–245):
   Trustworthy Multimodal Hate-Meme Detection
   ──────────────────────────────────────────
SUBTITLE 1 (y 246–337):
   A Robustness Audit of CLIP-Fusion Classifiers Under
   Naturalistic and Adversarial Perturbations
SUBTITLE 2 (y 325–403):
   Enric Guasch · [[HENRIK: full name]]   ·   EE-559 Deep Learning · EPFL · May 2026
```

The single hook for someone walking past at 3 m: *"hate-meme detectors, audited under the two threats they actually meet."*

---

### 2.3 Panel-by-panel content

#### Panel A — *Why robustness, why now* (top-left, highlighted, 1440 × 749)

**Header text:** `1 · Why we need to audit hate-meme detectors`

**Body (≈ 4 short bullets + 1 visual cue):**

- Hateful memes are **compositional**: image and text are individually benign, hateful together. CLIP-based fusion classifiers handle the clean compositional case well — but content moderation does *not* see clean inputs.
- Real users **obfuscate** when their content gets flagged: `hate → h@te`, character padding, JPEG compression, brightness shifts. These edits preserve meaning for humans and break detectors.
- Real adversaries **attack**: small, imperceptible pixel perturbations can flip predictions of any gradient-accessible classifier.
- Today's hate-detector papers report clean-benchmark numbers. **A deployable system needs reliability under both regimes.**

**Visual cue (small inset, ~700 × 320 inside the panel):**
A side-by-side trio:
```
   clean meme         h@te / blur version         PGD ε=4/255
   p(hateful)=0.84    p(hateful)=0.31             p(hateful)=0.12
```
(Use a synthetic placeholder meme or one from the safe HM example set — never a real hateful image; follow the ethical guidance in `ProjectIdeaHenrik_Feasibility_Roadmap.md` § 11.)

---

#### Panel B — *Model & two-threat framework* (mid-left, 1418 × 960)

**Header text:** `2 · One classifier, two threat models`

**Body, two sub-sections side-by-side or stacked:**

**B.1 — Architecture (≈ left half, 700 × 900):**
- CLIP ViT-B/32 dual encoder (OpenCLIP, `laion2b_s34b_b79k`); both encoders **frozen**.
- Fusion: `concat(t, v, |t − v|, t ⊙ v) ∈ ℝ^2048` → LayerNorm → Dropout(0.2) → Linear(2048 → 512) → GELU → Dropout(0.2) → Linear(512 → 1).
- ~152 M params total, **~1.3 M trainable**.
- `Normalize(mean, std)` is a `nn.Module` **inside** the classifier so attacks see raw `[0, 1]` pixels (PGD-ready forward graph).
- Trained 10 epochs, AdamW, bf16, batch 128, 3 seeds, single A100.

**(Architecture diagram, ~700 × 600.)** Recommended diagram source: adapt the ASCII figure in `model_architecture.md` § 1 into a clean vector graphic.

**B.2 — Two threat models (≈ right half or below, 700 × 900):**

```
THREAT MODEL 1 — the user           THREAT MODEL 2 — the adversary
─────────────────────────────       ──────────────────────────────
- black-box                          - white-box (model + grads)
- naturalistic, label-preserving     - imperceptible ℓ∞ pixel pert.
- 12 perturbation families           - FGSM, PGD; ε ∈ {1,2,4,8}/255
  × 3 severities + mixed             - image branch only
- single-cell + composite (2t / 2i   - evaluation on test_seen,
  / 2t+2i)                            test_unseen
- evaluated on all 3 splits          [[HENRIK]]
ENRIC                                HENRIK
```

Splits: **dev (n = 500)**, **test_seen (n = 1000, 49 % pos)**, **test_unseen (n = 2000, 37.5 % pos)**.

---

#### Panel C — *Defending the naturalistic case* (lower-left, 1413 × 1243, **hero figure inside**)

**Header text:** `3 · Naturalistic audit & a free-lunch defense`

**Body — narrative band above hero figure (~1413 × 240, y 2364 – 2611):**

- Audit grid: 57 single-cell + 16 composite cells × 9 recipes × 3 seeds × 3 splits.
- **Text is the brittle modality.** Worst-cell text ΔAUROC on the clean baseline reaches **0.115 / 0.129 / 0.110** (dev / test_seen / test_unseen). Worst image cell is uniformly smaller.
- **Composites are the worst threat.** `composite_2text_2image` high severity reaches ΔAUROC **0.140** on the clean baseline (test_unseen).
- **Defense:** during training, sample a perturbed view per example, add
  `L = BCE(y, p_clean) + α·BCE(y, p_pert) + β·KL(p_clean ‖ p_pert)` (α = 1, β = 0.5);
  with prob *p* drop the **text branch** (`text → ""`) so the head must commit to a useful image representation.
- **Sweep result (hero figure below):** clean AUROC stays inside seed noise for **p ≤ 0.25**, while image-only AUROC climbs monotonically. The Pareto front sits at **p ∈ [0.15, 0.25]**.

**Hero figure (the inner #b7b7b7 box, 1411 × 858, y 2611 – 3469):**

```
                Pareto plot — Clean AUROC × Image-only AUROC
                (test_unseen; 3-seed mean ± σ; 7 dropout rates)

         0.75 ┤ clean ●                   ★ kldrop-p015
   Clean      │                              ★ kldrop-p025
   AUROC      │     ● kl                      ★ kldrop-p010
              │                                       ★ kldrop-p020
         0.72 ┤                                              ● kldrop (p=.30)  [dominated]
              │                                              ★ kldrop-p050
              │                                  ● augonly
         0.69 ┼────────┬────────┬────────┬────────┬────────┬────────
              0.58    0.60    0.62    0.64    0.66    0.68
                                  Image-only AUROC

   ★ = our 7-point text-modality-dropout sweep         ● = baselines
```

Use `scripts/make_poster_figures.py` → figure 06 (kldrop sweep, 7 points) for the actual rendering. Caption (one line, under the figure): *"7-point sweep: p ∈ [0.15, 0.25] gives image-branch revival at zero clean-accuracy cost; p = 0.30 is Pareto-dominated."*

---

#### Panel D — *Adversarial audit & defense* (bottom-left, 1413 × 745)  `[[HENRIK]]`

**Header text:** `4 · Adversarial audit & adversarial-trained defense`  *(`[[HENRIK]]`)*

**Body sketch (placeholder content for Henrik to fill):**

- **Audit (`[[HENRIK]]`):** PGD/FGSM at ε ∈ {1, 2, 4, 8}/255 on the image branch. Expected pattern (from the naturalistic side's whitebox audit, see `Phase7_Completion_Report.md`): clean-trained baseline AUROC collapses by ε = 2/255 and is below 0.55 by ε = 8/255 on every recipe.
- **Defense (`[[HENRIK]]`):** Madry-style adversarial training — inner PGD at ε = `[[HENRIK]]`, K = `[[HENRIK]]` steps, schedule `[[HENRIK]]`. Reuse `attacks/pgd.py` (`fgsm_image`, `pgd_image`); wrap the inner-max into `stage1_robust.py`.
- **Result (`[[HENRIK]]`):** ε-curve plot — baseline vs adversarially-trained; expected story: adv-training recovers AUROC at ε ≤ 4/255 at the cost of `[[HENRIK]]` pp clean AUROC.
- **Composition (`[[HENRIK, optional]]`):** does `kldrop-p015 + adv-training` stack, or trade off? Two bars per ε-cell.

**Visual (~1413 × 450, inside the panel):** the ε-curve figure produced by Henrik's pipeline. Until then, drop a placeholder rectangle labelled `Henrik: PGD ε-curve`.

---

#### Panel E — *Methods deep-dive* (top-right, 1417 × 2063, **tallest panel**)

**Header text:** `5 · Perturbation suite, defenses, and evaluation protocol`

This panel is the methodological back-bone the body can refer back to. Split into 4 stacked sub-sections.

**E.1 — Text perturbation families (≈ 1417 × 500):**

| Family | Example | Severities |
|---|---|---|
| Leetspeak | `hate → h@te`, `i → 1` | low / med / high |
| Char-insertion | `hate → h.a.t.e` | 5 / 10 / 20 % |
| Char-deletion | random drop | 5 / 10 / 20 % |
| Char-swap | adjacent swap | 1 / 2 / 3 |
| Case + punct. noise | random casing | low / med / high |
| Censoring | `hate → h*te` | low / med / high |
| Keyboard typo | nearest-key | low / med / high |

**E.2 — Image perturbation families (≈ 1417 × 400):**

| Family | Severities |
|---|---|
| Gaussian noise | σ ∈ {0.01, 0.03, 0.05} |
| Gaussian blur | k ∈ {3, 5, 7} |
| JPEG compression | q ∈ {80, 50, 25} |
| Brightness shift | ± {10, 25, 40} % |
| Contrast shift | ± {10, 25, 40} % |
| Translation / crop | {2, 5, 10} % |

**E.3 — Composite cells (≈ 1417 × 300):**
- `composite_2text`, `composite_2image`, `composite_2text_2image`
- Severities: low / med / high / **mixed** (per-component severity sampled per sample — the realistic-user case).
- Verified that mixed ≈ medium for every recipe (`Phase9a_MixedComposites_Report.md`).

**E.4 — Recipes & training (≈ 1417 × 500):**

| Recipe | Loss term added | text-drop p |
|---|---|---|
| `clean` (baseline) | — | 0 |
| `augonly` | — | 0 |
| `kl` | + β · KL(p_c ‖ p_p) | 0 |
| `kldrop-p015` ★ | + β · KL(p_c ‖ p_p) | 0.15 |
| `kldrop-p025` ★ | + β · KL(p_c ‖ p_p) | 0.25 |
| `kldrop-p050` | + β · KL(p_c ‖ p_p) | 0.50 |
| `kldrop` (p = .30) | + β · KL(p_c ‖ p_p) | 0.30 (dominated) |
| `kllowmed` | + β · KL, low+med pert. only | 0 |
| `[[HENRIK]]` adv-trained | + Madry inner-max | – |

`α = 1.0`, `β = 0.5`, 3 seeds. **Evaluation protocol:** decision threshold τ tuned on **dev** only and reused unchanged on `test_{seen,unseen}`; AUROC + macro-F1 + ASR + worst-case-across-families.

**E.5 — Headline numbers (3-seed mean, all 3 splits, single dense table; from `Handoff_2026-05-19-final.md` § 3):**

| Recipe | Clean AUROC dev / seen / unseen | Worst text Δ | Image-only AUROC | Composite_2t mid Δ |
|---|---|---:|---:|---:|
| clean         | 0.742 / 0.747 / 0.738 | 0.115 / 0.129 / 0.110 | 0.593 / 0.582 / 0.602 | 0.090 / 0.087 / 0.073 |
| augonly       | 0.712 / 0.716 / 0.708 | 0.083 / 0.083 / 0.063 | 0.604 / 0.606 / 0.620 | 0.063 / 0.058 / 0.043 |
| kl            | 0.737 / 0.745 / 0.738 | 0.090 / 0.086 / 0.065 | 0.611 / 0.605 / 0.628 | 0.066 / 0.064 / 0.052 |
| **kldrop-p015** ★ | **0.735 / 0.748 / 0.743** | 0.081 / 0.087 / 0.069 | 0.638 / 0.638 / 0.658 | 0.059 / 0.064 / 0.054 |
| **kldrop-p025** ★ | 0.733 / 0.747 / 0.741 | 0.077 / 0.083 / 0.063 | **0.644 / 0.647 / 0.668** | 0.059 / 0.059 / 0.049 |
| kldrop-p050   | 0.700 / 0.717 / 0.712 | **0.065 / 0.068 / 0.049** | 0.641 / 0.651 / 0.666 | **0.046 / 0.046 / 0.035** |
| `[[HENRIK]]` adv | `[[HENRIK]]` | – | – | – |

★ = jointly Pareto-optimal in the (clean, image-only) plane.

---

#### Panel F — *Class-asymmetric mechanism* (mid-right, 1417 × 613)

**Header text:** `6 · The defense doesn't sacrifice hateful-recall — it stops over-flagging`

**Body (≈ 4 lines + small bar plot):**

- Per-example disagreement between `kldrop-p015` and `kl` on test_unseen: **n = 131** flipped predictions out of 2000.
- **97 % of those flips are on label = 0** (bootstrap 95 % CI **94 – 99 %**).
- Mechanism: the image-branch revival cuts false-positive flags on benign memes that the text-only `kl` recipe was over-flagging because the text mentioned a protected group.
- Hateful-class recall is preserved within seed noise on every split.
- **Why this matters for the course:** EE-559's stated objective — *"differentiate harmful hate speech from legitimate critical discourse or satire"* — is **literally** the false-positive-rate problem this finding addresses.

**Visual (~1417 × 320):** a two-bar plot — *"Changed predictions: 97 % label = 0 vs 3 % label = 1, 95 % CI bars."* Source: `failure_analysis.py --split test_unseen`.

---

#### Panel G — *Take-away* (lower-right, 1440 × 686, **highlighted**)

**Header text:** `7 · Take-away — a recipe for a trustworthy hate-meme classifier`

**Body (≈ 5 lines, large type, this is the panel a glancing reader reads):**

> A multimodal hate-meme classifier becomes trustworthy only when it survives **both** the user-obfuscation and the adversary threat models.
>
> **For the user case** (this work): **CLIP fusion + KL-consistency + text-modality dropout p ∈ [0.15, 0.25]**. Image-only AUROC 0.628 → 0.668 on the most naturalistic split, at **zero clean-accuracy cost**. The defense cuts false positives on benign content (97 % label = 0 flips) without sacrificing hateful-class recall.
>
> **For the adversary case** `[[HENRIK]]`: **PGD adversarial training**, closing the ε ≥ 2/255 gap left open by the naturalistic recipe at `[[HENRIK: # pp]]` clean-accuracy cost.
>
> **Stacked recipe** `[[HENRIK]]`: `kldrop-p015 + adv-training` — defends both threats simultaneously.
>
> *Open*: composite high-severity cells (ΔAUROC ≈ 0.14 on the worst attack) and `[[HENRIK: ε ≥ … ?]]` remain partially open.

---

#### References + acknowledgments (bottom-right, 1497 × 403)

**Header text:** `References`

Pack as 2 columns, 8 pt, ~25 numbered refs. Must-include:
- Kiela et al. 2020 — *The Hateful Memes Challenge* (Facebook AI).
- Radford et al. 2021 — *CLIP* / OpenCLIP (`laion2b_s34b_b79k`).
- Madry et al. 2018 — *Towards Deep Learning Models Resistant to Adversarial Attacks* `[[HENRIK]]`.
- Goodfellow et al. 2015 — *Explaining and Harnessing Adversarial Examples* (FGSM) `[[HENRIK]]`.
- Hendrycks & Dietterich 2019 — *ImageNet-C* (severity-based naturalistic robustness benchmark tradition).
- Sohn et al. 2020 — *FixMatch* (consistency-loss lineage).
- Srivastava et al. 2014 — *Dropout* (modality-dropout lineage).
- Lee et al. 2020 — *Multimodal modality dropout* (closest precedent).
- MemeCLIP (Zhao et al. 2024) — closest prior CLIP-based meme detector.
- WOAH 2025 bias-mitigation taxonomy — fairness context for the asymmetric finding.

**Acknowledgments (1 line, italic):** *Compute: EPFL Run:AI (course `course-ee-559-guasch`, 1 × A100 80 GB). Dataset: Hateful Memes via the EPFL group mirror. Course staff: EE-559.*

---

#### Code & data strip (bottom-left, 960 × 84)

Single line, mono-spaced font:

```
Code · github.com/[[ORG]]/robust-meme-hate-detection      Reproduce · scripts/make_poster_figures.py
```

---

### 2.4 Visual style guidance (against the template's palette)

- **Header bars** of each panel: dark navy `#143264` (template-native, e.g. the EPFL-style title bar).
- **Body text**: dark grey `#595959` (template-native body colour).
- **Hero figure box outline**: keep the template's mid-grey `#b7b7b7` as the figure container; place the Pareto plot inside with white background.
- **Highlight panels (A and G)**: keep `#efefef` background — these are the two "first-glance" panels and the off-white tells the eye where to land.
- **Accent colour** for the two recommended recipes (★) on the Pareto plot: one consistent accent (e.g. EPFL red `#FF0000`, kept to ≤ 5 % of poster area).
- **No emojis, no clipart.** One small synthetic example meme inset in Panel A; everything else is data figures.

### 2.5 Reading-path test

A reviewer who reads only the highlighted panels + the hero figure must leave with:

1. *Why it matters* (Panel A): hate-meme classifiers are brittle under both users and adversaries.
2. *Headline result* (hero figure inside Panel C): a 7-point sweep shows a free-lunch Pareto front at p ∈ [0.15, 0.25].
3. *Take-away* (Panel G): `CLIP + KL-consistency + dropout p ∈ [0.15, 0.25]` for users, `+ adv-training` `[[HENRIK]]` for adversaries.

If those three slots transmit that triplet, the poster works.

---

## 3. Henrik's slots (`grep`-able)

| ID | Location | What Henrik fills |
|---|---|---|
| `[[HENRIK: name]]` | Subtitle 2 (banner) | Full author name |
| `[[HENRIK: adversarial training]]` | § 1.1, Panel G | Defense one-liner |
| `[[HENRIK]]` Panel D body | Panel D (bottom-left) | Audit recap + defense recap |
| `[[HENRIK]]` ε-curve figure | Panel D visual | PGD ε-curve plot, baseline vs adv-trained |
| `[[HENRIK]]` Panel E row | Panel E.4 recipes table | Adv-training recipe line |
| `[[HENRIK]]` Panel E.5 cell | Panel E.5 headline table | Adv-trained AUROC numbers |
| `[[HENRIK]]` Panel G text | Panel G take-away | Adversary half of the recipe (`# pp` clean cost) |
| `[[HENRIK]]` refs | References | Madry, Goodfellow (FGSM) |
| `[[HENRIK]]` composition bullet | Panel D / Panel G | Does the stacked recipe trade off? |

Naming convention: every Henrik-owned literal is wrapped in `[[HENRIK: …]]` so it shows up immediately under `grep -n HENRIK project_planning/Project_Framing_and_Poster_Draft.md`.

---

## 4. Open production tasks before printing

1. Vectorise the **architecture diagram** for Panel B from `model_architecture.md` § 1 (Adobe Illustrator / Inkscape; A0 print-ready).
2. Render Panel C **Pareto plot** from `scripts/make_poster_figures.py` → figure 06 at A0-print DPI (≥ 300 DPI).
3. Render Panel F **bar plot** from `failure_analysis.py --split test_unseen`.
4. `[[HENRIK]]` produce Panel D **ε-curve figure**.
5. Compose final SVG: open `EE559_A0_poster_template.svg` in Inkscape; layer panel headers + body text + figures into the regions listed in § 2.1; export to PDF/A.
6. Manual sanity check: at least one person reads only Panels A + hero figure + G and must reproduce the take-away verbatim.

---

## 5. Mapping from poster panels back to the report

For consistency between artifacts, the report should mirror the poster's labelling:

| Poster panel | Report section | Source-of-truth document |
|---|---|---|
| A | §1 Introduction | `Project_Info.md` + this doc § 1 |
| B | §3 Method (architecture + threat models) | `model_architecture.md` |
| E | §3 Method (perturbation suite + recipes) | `Phase3_4_Completion_Report.md`, `Phase5_Completion_Report.md`, `Phase9a_*.md`, `Phase9b_*.md` |
| C | §4.1–4.3 Naturalistic results & defense | `Phase7_Completion_Report.md`, `Phase9b_DropoutSweep_Report.md` |
| F | §4.4 Class-asymmetric mechanism | `Phase6_Completion_Report.md`, `Phase8_TestFailure_Report.md` |
| D | §4.5 Adversarial results & defense | `[[HENRIK]]` |
| G | §5 Discussion + §6 Conclusion | This doc § 1.5, § 2.3 (Panel G) |
| Refs | §7 References | Combined |

That gives a single coherent thread that runs poster → report → committed code.
