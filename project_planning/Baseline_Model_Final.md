# Baseline Multimodal Model: Final Discussion and Decision

**Project:** Adversarial Robustness of Multimodal Meme Hate Detectors (EE-559 mini-project, 6-8 weeks)
**Purpose of this document:** Reconcile the three working drafts (`BaselineModel_Discussion.md`, `Baseline_Model_Discussion.md`, `BaselineModel_Discussion_Claude.md`) into one decision, with corrected facts and a single coherent recommendation.

---

## 1. Executive recommendation

Use a **single pretrained CLIP/OpenCLIP-based dual-encoder model with a small fusion classifier** as the project's baseline. Specifically:

```text
text_emb  = CLIP_TextEncoder(tokens)           # OpenCLIP ViT-B/32, laion2b_s34b_b79k
image_emb = CLIP_ImageEncoder(pixels)
features  = concat([text_emb, image_emb,
                    |text_emb - image_emb|, text_emb * image_emb])
logit     = MLP(LayerNorm(features))
```

This is a *generic pretrained model adapted to Hateful Memes*, not a randomly initialised model. We do **not** train CLIP from scratch; we download the pretrained encoders, freeze them (with optional LoRA/last-block unfreeze if needed), and train only a 2-layer MLP head on the ~8.5k Hateful Memes training examples. The 2020 competition winners are **not** suitable as the main model and should be cited only as historical reference.

---

## 2. Why this baseline (project-specific selection criteria)

The baseline is the *vehicle* for a robustness study, not the contribution. The dominant requirement is not maximum clean AUROC; it is whether the model lets us make clean, controlled claims about robustness under text and image perturbations.

| Criterion | Why it matters here |
|---|---|
| **Single model, not ensemble** | Robustness gap, attack success rate, and consistency-loss training are only interpretable for a single classifier. |
| **End-to-end differentiable from raw pixels and tokens** | White-box image attacks (FGSM/PGD) and consistency regularisation require gradients through the entire image path. |
| **Modular text and image branches** | The roadmap mandates text-only, image-only, and multimodal baselines; dual-encoder design gives these for free. |
| **Reproducible on 2026-native libraries** | A 6-8 week budget cannot absorb 2020-era CUDA/MMF/Detectron2 dependency recovery. |
| **Fine-tunable on one course GPU** | Frozen-encoder training, last-block tuning, or LoRA must be realistic on ~16-24 GB VRAM. |
| **No proprietary external side channels** | Web-entity / face-attribute / OCR-via-API tags are not perturbable, not reproducible, and confound the analysis. |
| **Strong-enough clean baseline (>0.70 AUROC)** | Robustness gaps are only meaningful on a non-trivial classifier. |

---

## 3. The Hateful Memes 2020 winners

The competition was scored by ROC AUC on the unseen Phase 2 test set. The top-five private-test AUROCs are **0.8450, 0.8310, 0.8108, 0.8053, 0.7943**. These numbers establish a useful upper bound for clean performance, but the systems behind them are not appropriate baselines for this project.

### 3.1 1st place - alfred lab (Chu Po-Hsuan) - [HimariO/HatefulMemesChallenge](https://github.com/HimariO/HatefulMemesChallenge)

Ensemble of modified **VL-BERT, UNITER/VILLA, ERNIE-ViL** with extensive external signals: OCR + image inpainting to remove meme text before feature extraction, bottom-up-attention region features, web-entity detection, human-race tagging, and a rule-based racism detector that combines text, race tags, entity tags, and skin tone. Training reportedly used a 16-core, 104 GB RAM, 4xT4 setup; VL-BERT alone needed 4 GPUs to reach a viable batch size.

**Verdict.** Not suitable. Too many side channels (each of which would need its own perturbation analysis), too compute-heavy, and external taggers are non-reproducible. Cite as the leaderboard ceiling.

> Note: the local `MultimodalModels.md` lists "Ron Zhu" as 1st place; the DrivenData winner writeup credits **Chu Po-Hsuan / alfred lab**. The HimariO repository is the corresponding code release.

### 3.2 2nd place - Niklas Muennighoff (vilio) - [Muennighoff/vilio](https://github.com/Muennighoff/vilio)

The cleanest engineering of the top entries. Implements **Oscar, UNITER, VILLA, LXMERT, VisualBERT, ERNIE-ViL** under a unified codebase with consistent training and reproduction scripts.

**Verdict.** Not suitable. All models still consume Faster R-CNN bottom-up-attention region features, which couples image perturbations to a separate detector and breaks pixel-space attack semantics. Vilio's main contribution is *ensembling* heterogeneous V+L backbones; using only one of them throws away its main contribution. Useful as a reference for dataset handling and class-imbalance tricks.

### 3.3 3rd place - HateDetectron (Velioglu & Rose) - [Colab notebook](https://colab.research.google.com/drive/1O0m0j9_NBInzdo3K04jD19IyOhBR1I8i)

**VisualBERT** initialised from COCO captions, fine-tuned on Hateful Memes with hyperparameter search; final score (~0.811 AUROC, 0.765 accuracy) comes from majority-voting an ensemble of 27 selected checkpoints. Image features are again Faster R-CNN ROIs.

**Verdict.** Not suitable. Same region-feature problem; the contribution is HP sweeping + ensembling, not architectural.

### 3.4 4th place - Kingsterdam (Lippe et al.) - [Nithin-Holla/meme_challenge](https://github.com/Nithin-Holla/meme_challenge)

Compared **UNITER, OSCAR, LXMERT**, found UNITER strongest, upsampled text confounders, reweighted hateful examples, and produced final predictions through a 15-fold weighted ensemble over UNITER-base.

**Verdict.** Not suitable. Same region-feature pipeline. Two ideas worth borrowing: explicit attention to Hateful Memes confounders, and reporting across seeds/folds rather than a single run.

### 3.5 5th place - Vlad Sandulescu (burebista) - [vladsandulescu/hatefulmemes](https://github.com/vladsandulescu/hatefulmemes)

**UNITER-large** with image-caption auxiliary text generated by an Im2txt model, ensembled across 12 models, plus Faster R-CNN region features. Multiple subprojects and separate environments for bottom-up attention, captioning, and UNITER.

**Verdict.** Not suitable. Im2txt captions become an additional side channel whose robustness is decoupled from the meme classifier itself.

### 3.6 Common themes (and why they all fail our criteria)

Every top-5 entry shares three properties:

1. **Region-feature image branch** (Faster R-CNN / Detectron2 bottom-up attention). Decouples the image input from the model's actual visual representation in a way that breaks pixel-space perturbation studies.
2. **Late ensembling of multiple V+L transformers.** Inflates leaderboard scores but fragments the model into pieces we cannot uniformly attack or robustify.
3. **2020-era stack.** Old PyTorch, old MMF, old Detectron2; in some cases proprietary APIs. Reproducibility is a project-killing risk on a 6-8 week budget.

These models also predate CLIP. Subsequent CLIP-based work (Hate-CLIPper 2022, ISSUES 2023, MemeCLIP 2024) reaches comparable Hateful Memes AUROC with a fraction of the parameters and a much simpler image path.

---

## 4. Recommended architecture

A single CLIP-based multimodal classifier built on top of frozen (then optionally LoRA-tuned) CLIP encoders.

### 4.1 Backbone and head

- **Backbone:** OpenCLIP `ViT-B/32` (`laion2b_s34b_b79k`). ~150M params total. Fits batch 32-64 on a single 16-24 GB GPU.
- **Stretch backbone:** `ViT-L/14` (laion2b) only as an ablation if compute allows.
- **Fusion:** concat(text, image, |text - image|, text * image), LayerNorm, dropout 0.2.
- **Head:** 2-layer MLP, hidden 512, sigmoid output for binary BCE.

### 4.2 Why this is the right choice

- **End-to-end differentiable from raw pixels and tokens.** FGSM/PGD on the image and gradient-based text attacks both work without surgery.
- **Modular by construction.** Text-only and image-only baselines come from zeroing or removing the corresponding branch.
- **2026-native dependencies.** OpenCLIP, PyTorch 2.x, transformers - all actively maintained.
- **Strong-enough clean performance.** CLIP-based fusion heads (Hate-CLIPper, MemeCLIP, ISSUES) reach ~0.74-0.78 AUROC on Hateful Memes dev-seen, well above the threshold needed for meaningful robustness gaps and within striking distance of the 2020 ensemble winners.
- **Single model, single checkpoint.** Robustness metrics (ASR, robustness gap, worst-case F1) are unambiguous.
- **Compute-friendly.** A full clean training run is hours, not days. The full perturbation grid (≥10 attacks * 3 severities) is tractable.

### 4.3 What we borrow from the winners

- **Cross-modal interaction features** (element-wise product, absolute difference): inductive bias similar to UNITER-style cross-attention without the Faster R-CNN dependency.
- **Confounder-aware evaluation** (from Kingsterdam): pay attention to Hateful Memes' image/text-confounder splits, not just the global score.
- **Threshold tuning on validation, not 0.5** (standard across all top entries; can be worth 1-2 AUROC).
- **Multiple-seed reporting.** The winners obtained this implicitly via ensembling; we do it explicitly with 3 seeds on the main multimodal and robust-multimodal runs.

We explicitly do **not** use:
- Faster R-CNN / Detectron2 region features.
- External entity / face / race attribute taggers or proprietary APIs.
- Heterogeneous transformer ensembles.
- 2020-era V+L checkpoints.

---

## 5. Do we have to train from zero? No.

Three distinct layers of pretraining must be kept separate. Only the smallest is something we train ourselves.

| Layer | What | Source | Our action |
|---|---|---|---|
| **A. Generic image+text encoders** | CLIP ViT-B/32 weights pretrained on ~2B image-caption pairs (LAION-2B). | OpenCLIP, freely downloadable. | **Use as-is.** |
| **B. Hateful-Memes classifier head** | A 2-layer MLP mapping fused CLIP features to a hateful/non-hateful logit. | Not available off-the-shelf for our exact architecture. | **Train ourselves**, hours on one GPU. |
| **C. Optional encoder fine-tuning** | LoRA adapters or last-block unfreeze of CLIP encoders. | Only if Layer B alone underfits. | **Train ourselves if needed.** |

CLIP itself is never retrained. We download a multi-billion-parameter pretrained backbone, freeze it, and train a small head on top. This is what Hate-CLIPper, MemeCLIP, and ISSUES all do.

### 5.1 Existing third-party Hateful-Memes-fine-tuned checkpoints

A 2026 survey of CLIP-based meme-hate work that releases weights:

| Project | Backbone | HMC weights public? | Use to us |
|---|---|---|---|
| **[ISSUES](https://github.com/miccunifi/ISSUES)** (ICCVW 2023) | CLIP + textual inversion + combiner | **Yes** (HMC + HarMeme, GitHub releases). | Reference upper-bound number; not our architecture. |
| **[Hate-CLIPper](https://github.com/gokulkarthik/hateclipper)** (EMNLP-W 2022) | CLIP + Feature Interaction Matrix | Code released, **no clear weights**. | Reference architecture - essentially what we are building. |
| **[MemeCLIP](https://github.com/SiddhantBikram/MemeCLIP)** (EMNLP 2024) | CLIP + lightweight adapters | Weights only for **PrideMM**, not HMC. | Copy the head design; not directly usable. |
| **[RGCL / RA-HMD](https://github.com/JingbiaoMei/RGCL)** (ACL 2024 / EMNLP 2025) | Qwen2.5-VL (7B+) with retrieval | Available via project HF. | Out of scope - 7B+ VLM exceeds our compute and complicates attack pipeline. |
| Community HF unimodal fine-tunes ([ResNet-50](https://huggingface.co/tommilyjones/resnet-50-finetuned-hateful-meme-restructured-balanced), [BERT](https://huggingface.co/limjiayi/bert-hateful-memes-expanded)) | ResNet-50 only / BERT only | Yes | Unimodal sanity baselines, not the multimodal model. |

### 5.2 Why we still train the head ourselves rather than dropping in ISSUES

ISSUES is the most tempting drop-in (CLIP-based, public weights, trained on HMC). For a robustness study it is still the wrong move:

1. **We need a known training distribution.** Clean-trained vs robust-trained must come from the same code path; otherwise the comparison is contaminated by every implementation difference.
2. **ISSUES architecture is not what we proposed.** Textual inversion and the combiner introduce extra components whose perturbation behaviour would have to be characterised separately.
3. **We need full pixel-level gradient access for FGSM/PGD.** Adapting a third-party checkpoint's preprocessing to our attack code is a known source of subtle bugs.
4. **Training the head is cheap.** Frozen ViT-B/32 + 2-layer MLP on 8.5k training images is hours on one GPU. There is no time saved by skipping it; there is time *lost* to debugging dependency mismatches.

### 5.3 What we actually download

Pretrained artifacts that go into the pipeline:

- **OpenCLIP ViT-B/32** (`laion2b_s34b_b79k`). Default.
- *Optional* **OpenCLIP ViT-L/14** for ablation.
- *Optional* **HuggingFace BERT/RoBERTa base** if we want a non-CLIP text-only baseline (CLIP-text keeps the architecture homogeneous and is our default).

Pulled only as **reference numbers**, not as components:

- ISSUES official checkpoint - "best-published CLIP-based number on HMC dev-seen" alongside our own clean baseline.
- Reported AUROCs from alfred lab / Muennighoff (2020 ensembles) - leaderboard ceiling.

---

## 6. Concrete training plan

| Stage | What | Details | Expected outcome |
|---|---|---|---|
| 0 | **No pretraining step.** | CLIP weights are downloaded by `open_clip` on first instantiation. | - |
| 1 | **Frozen encoders, head only.** | 3-5 epochs, batch 32-64, AdamW, lr ~1e-3 on the head, weighted BCE, threshold tuned on validation for macro F1. | ~0.72-0.76 AUROC on dev-seen. Hours on one GPU. |
| 2 | *(optional)* **Last-block unfreeze or LoRA (rank 8-16).** | Only if Stage 1 plateaus < 0.72 AUROC. lr ~1e-5 on encoder params, head lr unchanged. | Modest gain; documented as ablation. |
| 3 | **Robust variant.** | Same as Stage 1/2 but each batch contains a clean and a perturbed view. Loss = BCE(clean) + alpha * BCE(perturbed) + beta * KL(p_clean || p_perturbed). Start alpha=1.0, beta=0.5. | Lower ASR on the perturbation suite at small clean-AUROC cost. |

Compute budget for the full pass (clean + robust, 3 seeds each) is on the order of **a few GPU-days**, well within the project envelope.

---

## 7. Alternatives considered and rejected

| Alternative | Why rejected |
|---|---|
| Reproduce vilio (cleanest of the winners) | Region-feature pipeline; image-perturbation semantics become unclear; 2020 dependencies. |
| VisualBERT via Hugging Face | Same region-feature problem; HF port still expects ROI features. |
| BLIP-2 / LLaVA / Idefics / Qwen2.5-VL as classifier | 7B+ params; fine-tuning out of scope; would dominate the project budget. |
| Train a ViT + BERT cross-attention model from scratch | Reinvents 2020 V+L pretraining without the data; clean accuracy will lose to CLIP. |
| Use ISSUES checkpoint as the primary clean baseline | Different architecture; opaque preprocessing; contaminates clean-vs-robust comparison (see §5.2). |
| Use MemeCLIP weights | Released only for PrideMM, not Facebook Hateful Memes. |
| OpenCLIP ViT-L/14 as default | Heavier than necessary; keep as ablation. |

---

## 8. Final decision

> **Use a single OpenCLIP ViT-B/32 dual-encoder fusion classifier as the baseline multimodal model. Train only the 2-layer MLP head ourselves; freeze the CLIP encoders by default and apply LoRA or last-block unfreezing only if Stage-1 performance plateaus. Treat the 2020 competition winners and the ISSUES checkpoint as published reference numbers, not as components.**

This is what enables the project's actual contribution: a clean, controlled, reproducible robustness benchmark with end-to-end differentiable attacks, modality ablations, and a robustness-aware retraining variant. Picking a competition winner would optimise for the wrong axis (clean leaderboard score) at the cost of the axis that matters for us (analytical clarity under perturbation).

---

## Sources

- [Hateful Memes Challenge - Meta AI](https://ai.meta.com/tools/hatefulmemes/)
- [Hateful Memes Phase 2 leaderboard - DrivenData](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/leaderboard/)
- [HimariO / HatefulMemesChallenge (alfred lab, 1st)](https://github.com/HimariO/HatefulMemesChallenge)
- [Muennighoff / vilio (2nd)](https://github.com/Muennighoff/vilio)
- [HateDetectron Colab (Velioglu & Rose, 3rd)](https://colab.research.google.com/drive/1O0m0j9_NBInzdo3K04jD19IyOhBR1I8i)
- [Nithin-Holla / meme_challenge (Kingsterdam, 4th)](https://github.com/Nithin-Holla/meme_challenge)
- [vladsandulescu / hatefulmemes (5th)](https://github.com/vladsandulescu/hatefulmemes)
- [Hate-CLIPper paper (EMNLP-W 2022)](https://arxiv.org/abs/2210.05916) / [GitHub](https://github.com/gokulkarthik/hateclipper)
- [ISSUES GitHub (ICCVW 2023)](https://github.com/miccunifi/ISSUES)
- [MemeCLIP paper (EMNLP 2024)](https://arxiv.org/html/2409.14703v1) / [GitHub](https://github.com/SiddhantBikram/MemeCLIP)
- [RGCL / RA-HMD GitHub (ACL 2024 / EMNLP 2025)](https://github.com/JingbiaoMei/RGCL)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [OpenAI CLIP ViT-B/32 on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
- [tommilyjones/resnet-50-finetuned-hateful-meme (HF)](https://huggingface.co/tommilyjones/resnet-50-finetuned-hateful-meme-restructured-balanced)
- [limjiayi/bert-hateful-memes-expanded (HF)](https://huggingface.co/limjiayi/bert-hateful-memes-expanded)
