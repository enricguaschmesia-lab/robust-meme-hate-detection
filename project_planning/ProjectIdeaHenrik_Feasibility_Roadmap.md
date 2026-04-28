# Project Planning Note: Adversarial Robustness of Multimodal Meme Hate Detectors

## Executive summary

Henrik's idea is feasible and well aligned with the EE-559 theme, provided it is scoped as a focused robustness study rather than as a new state-of-the-art hate speech detector. The most suitable final project is:

> Train and evaluate a CLIP-based multimodal meme hate classifier, then systematically measure how much its performance degrades under label-preserving text and image perturbations, and whether a simple robustness-aware training strategy improves adversarial robustness without sacrificing too much clean accuracy.

The recommended scope is a 6 to 8 week mini-project centered on the Hateful Memes dataset, with optional secondary evaluation on MultiOFF, MAMI, or HarMeme only if the core pipeline is finished early. This keeps the project at a Master's course level: technically meaningful, measurable, and connected to deep learning concepts from the course, but not thesis-scale.

The main scientific contribution should not be "we built the best meme classifier." Instead, it should be:

1. A controlled robustness benchmark for multimodal hate meme detection.
2. A modality-level analysis of whether attacks on text, image, or both are most damaging.
3. A comparison between clean training and robustness-aware training.
4. A qualitative error analysis explaining why the model fails.

### Planning assumptions

This note assumes:

- The project is a group mini-project, not a thesis.
- The team has access to at least one GPU through local machines or the EPFL cluster.
- The main implementation window is roughly 6 to 8 weeks.
- The final deliverable is a report and poster, so analysis and presentation quality matter as much as raw benchmark performance.
- Dataset access may take time, so the first operational task should be requesting Hateful Memes access and preparing MultiOFF as a fallback.

## Recommended final project definition

### Working title

**Adversarial Robustness of Multimodal Meme Hate Speech Detection**

### Problem statement

Multimodal meme hate detectors rely on both image and text. This creates a specific vulnerability: a meme can be correctly classified when its text and image are clean, but a small change to either modality may break the model without changing the hateful meaning for a human reader. Examples include obfuscated words, inserted punctuation, altered spacing, mild image noise, JPEG compression, blur, or small shifts in meme text.

The project asks:

> How robust are CLIP-based multimodal hate meme classifiers to realistic text and image perturbations, and can robustness-aware training reduce this failure mode?

### Research questions

**RQ1. Clean performance:** How well do text-only, image-only, and multimodal models classify hateful versus non-hateful memes on clean data?

**RQ2. Attack vulnerability:** Which perturbations cause the largest performance drops: text perturbations, image perturbations, or combined multimodal perturbations?

**RQ3. Modality reliance:** Does the multimodal model actually use both modalities, or does it over-rely on one?

**RQ4. Defense:** Does adversarial or corruption-based data augmentation improve robustness while preserving clean performance?

**RQ5. Failure modes:** What kinds of examples remain brittle after robustness-aware training?

## Feasibility assessment

### Overall feasibility

The idea is feasible for a 4 credit Master's course project if the group avoids three forms of scope creep:

1. Training large multimodal foundation models from scratch.
2. Using many datasets with incompatible label definitions.
3. Building a complete adversarial attack library for both text and images.

The feasible version is to use pretrained encoders, a lightweight trainable fusion/classification model, and a carefully defined perturbation suite. This still demonstrates deep learning competence because the project includes multimodal representation learning, fine-tuning, adversarial evaluation, robustness metrics, ablations, and error analysis.

### Expected effort

Assuming a small group and a 6 to 8 week project window, the expected effort is moderate to high but manageable:

| Component | Expected difficulty | Comments |
|---|---:|---|
| Dataset access and preprocessing | Medium | Hateful Memes requires registration and license acceptance. Data format is straightforward once downloaded. |
| Baseline model | Medium | CLIP/OpenCLIP plus a small fusion head is manageable in PyTorch. |
| Text perturbation suite | Low to Medium | Character-level and word-level transformations are easy, but must be label-preserving. |
| Image perturbation suite | Low to Medium | Torchvision/Albumentations can implement most transformations. FGSM/PGD is more technical but bounded. |
| Robustness evaluation | Medium | Need careful metrics, thresholding, and clean-correct subsets. |
| Robust training | Medium | Random perturbation augmentation and consistency regularization are feasible. |
| Cross-dataset evaluation | Medium to High | Useful but can consume time due to label/schema differences. Keep optional. |
| Explainability | Medium to High | Useful for analysis, but should remain lightweight. |

### Practical effort envelope

A realistic implementation budget is:

| Work package | Suggested effort share |
|---|---:|
| Data access, preprocessing, and sanity checks | 15% |
| Clean baselines and training infrastructure | 20% |
| Perturbation suite and attack evaluation | 25% |
| Robustness-aware training | 15% |
| Ablations and qualitative analysis | 15% |
| Report, poster, and reproducibility cleanup | 10% |

This distribution is intentional. The project should spend more effort on rigorous evaluation than on adding many model variants. A clean robustness benchmark with one strong model is better than a shallow comparison of many models.

### Recommended scope for a 4 credit course project

The project should have one primary dataset, one main architecture, one attack suite, and one robustness defense.

**In scope:**

- Binary hateful versus non-hateful meme classification.
- Hateful Memes as the primary dataset.
- CLIP/OpenCLIP-based text and image encoders.
- Text-only, image-only, and multimodal baselines.
- At least 5 text perturbations and 5 image perturbations.
- 3 severity levels for each perturbation where applicable.
- Robustness metrics: clean F1/AUROC, attacked F1/AUROC, attack success rate, robustness gap.
- A simple defense: perturbation data augmentation plus consistency regularization.
- Error analysis with representative examples.

**Optional stretch scope:**

- Evaluate on one additional dataset: MultiOFF, MAMI, or HarMeme.
- Add a white-box image attack such as FGSM or PGD.
- Add OCR-based evaluation if using only raw meme images.
- Add Grad-CAM, occlusion sensitivity, or Integrated Gradients for qualitative explanations.
- Add target-group or offensive-category analysis if the chosen secondary dataset supports it.

**Out of scope:**

- Training a new large vision-language model.
- Full production moderation system.
- Video hate detection.
- Large-scale manual annotation.
- Generating new hateful memes.
- Complex LLM-based reasoning pipelines.
- Human-subject studies.
- Claims that the final model is deployment-ready.

### Feasibility verdict

**Verdict: strong idea if narrowed.**

The idea is course-feasible because it can be built from standard components and evaluated quantitatively. It becomes too broad only if the team tries to combine all proposed datasets, many model families, text/image/video modalities, and multiple defenses. A robust final project should prefer depth over breadth: one carefully evaluated multimodal classifier and a strong robustness analysis.

## Proposed modifications and improvements

### 1. Reframe the project from "attack memes" to "robustness under label-preserving perturbations"

The original idea lists attacks such as censoring words, adding noise, blur, brightness changes, and shifting text placement. These are good, but the project should distinguish between:

- **Naturalistic perturbations:** changes users commonly make online, such as "hate" -> "h@te", extra spaces, punctuation, compression, blur, and brightness changes.
- **Adversarial attacks:** optimization-based attacks designed to fool the model, such as FGSM or PGD on image pixels.

This distinction makes the report more scientifically clean. The minimum project can focus on naturalistic perturbations. If time allows, add FGSM or PGD as the adversarial component.

### 2. Use Hateful Memes as the primary dataset

The original idea lists Hateful Memes, MMHS150K, HarMeme, MAMI, and MultiOFF. These are all relevant, but using all of them as core training data is risky because the label definitions differ:

- Hateful Memes: hateful versus non-hateful, designed for multimodal confounders.
- MultiOFF: offensive versus non-offensive, not exactly hate speech.
- MAMI: misogyny-focused, narrower target category.
- HarMeme: harmful memes and targets.
- MMHS150K: large but noisier social-media multimodal data.

The project should train and evaluate primarily on Hateful Memes. Use one secondary dataset only as an optional generalization test.

### 3. Add unimodal baselines

A multimodal result is not convincing unless compared with text-only and image-only models. These baselines answer whether the model is actually using cross-modal information.

Recommended baselines:

- Majority-class baseline.
- Text-only CLIP text encoder or BERT/RoBERTa classifier.
- Image-only CLIP image encoder classifier.
- Multimodal CLIP fusion classifier.

### 4. Define attacks as label-preserving

A robustness study is only valid if perturbations do not change the ground-truth label. Some transformations can accidentally change meaning. For example, replacing a slur with "[censored]" may reduce the hateful content for a human, while blurring meme text too much may make the meme unreadable.

Recommended rule:

> Perturbations should preserve the human-interpretable label at mild and medium severity. High severity can be included as a stress test, but it must be reported separately.

Add a small manual sanity check of 50 to 100 attacked examples to verify that the perturbation suite is not destroying meaning.

### 5. Evaluate both random and worst-case perturbations

Do not only report average performance under random corruptions. Also report the worst perturbation per example or per attack family.

Useful metrics:

- Average attacked performance.
- Worst-case attacked performance across all attacks.
- Worst-case attacked performance within each modality.
- Attack success rate on examples that were clean-correct.

### 6. Include a simple defense

The project is stronger if it does not only show that the model is brittle. Add one practical robustness method:

- Random text and image perturbation augmentation during training.
- Consistency regularization between clean and perturbed predictions.
- Optional modality dropout to reduce over-reliance on a single modality.

This gives a natural before/after result.

### 7. Keep explainability lightweight

The original idea asks to study how text and image alone affect the model. This is important, but a full explainability project would be too large. Use simple analysis:

- Compare full model, text-only model, and image-only model.
- Evaluate attacks applied only to text versus only to image.
- Mask text or replace image with a blank image to measure modality dependence.
- Include a small qualitative gallery of failure cases.

If time remains, add occlusion sensitivity or Grad-CAM for the image branch.

## Proposed quantifiable objectives

The project should be judged by measurable objectives rather than vague goals.

### Minimum objectives

1. **Build a reproducible clean-data baseline.**
   - Train at least 3 baselines: text-only, image-only, multimodal.
   - Report accuracy, macro F1, AUROC, precision, recall, and confusion matrix.
   - Tune classification threshold on validation data rather than assuming 0.5.

2. **Implement a controlled perturbation benchmark.**
   - Implement at least 5 text perturbations.
   - Implement at least 5 image perturbations.
   - Use at least 3 severity levels where applicable.
   - Save attacked examples and metadata so results are reproducible.

3. **Measure robustness degradation.**
   - For each attack and severity, report attacked accuracy, macro F1, AUROC, and robustness gap.
   - Report attack success rate:

```text
ASR = (# examples clean-correct but attacked-wrong) / (# examples clean-correct)
```

   - Report worst-case robustness across attack families.

4. **Analyze modality sensitivity.**
   - Compare clean and attacked performance for text-only, image-only, and multimodal models.
   - Report whether text perturbations or image perturbations are more harmful.
   - Identify at least 20 representative failure cases.

5. **Train one robustness-aware variant.**
   - Train a multimodal model with perturbation augmentation and/or consistency regularization.
   - Compare clean and attacked performance against the clean-trained multimodal baseline.
   - Report whether robustness improves without a large clean-performance drop.

6. **Produce final scientific artifacts.**
   - A final report with methods, metrics, results, ablations, and limitations.
   - A poster-ready summary table and figure.
   - Reproducible code/configuration for training and evaluation.

### Suggested success criteria

These criteria are not guaranteed performance targets, but they define what would count as a strong course result:

| Objective | Strong outcome |
|---|---|
| Clean multimodal baseline | Beats both text-only and image-only baselines on macro F1 or AUROC. |
| Robustness benchmark | Covers at least 10 attack types total with severity curves. |
| Attack analysis | Identifies at least 3 attack families that significantly degrade performance. |
| Defense | Reduces ASR on at least 3 attack families while losing no more than 3 percentage points clean macro F1 or AUROC. |
| Modality analysis | Shows whether the model depends more on text, image, or their interaction. |
| Error analysis | Provides clear qualitative explanations for at least 20 model failures. |

### Stretch objectives

Only attempt these after the minimum objectives are complete:

- Add FGSM or PGD image attacks.
- Add a combined attack that perturbs both text and image.
- Evaluate transfer robustness on MultiOFF, MAMI, or HarMeme.
- Compare CLIP ViT-B/32 versus CLIP ViT-B/16 or OpenCLIP variants.
- Add adversarial fine-tuning with generated attacks rather than random augmentations.
- Add calibration metrics such as ECE and reliability diagrams.

## Proposed architecture

### System overview

```text
                  Clean meme dataset
                          |
                  Preprocessing layer
              image tensor + meme text
                          |
             +------------+-------------+
             |                          |
       Text perturbations         Image perturbations
             |                          |
             +------------+-------------+
                          |
                    Model pipeline
                          |
       +------------------+------------------+
       |                                     |
 CLIP text encoder                    CLIP image encoder
       |                                     |
 text embedding                       image embedding
       |                                     |
       +------------------+------------------+
                          |
                  Fusion module
                          |
                 Classification head
                          |
               hateful / non-hateful
                          |
          Clean and attacked evaluation
```

### Data flow

Each example should contain:

- `image_id`
- `image_path`
- `text`
- `label`
- `split`
- optional `attack_name`
- optional `attack_severity`
- optional `attack_modality`

For clean training, `attack_name = none`. For attacked evaluation, create transformed views without changing the original label.

### Model variants

#### Baseline 0: Majority class

This gives a sanity-check lower bound.

#### Baseline 1: Text-only model

Options:

- CLIP text encoder plus MLP classifier.
- RoBERTa/BERT plus MLP classifier.

The CLIP text encoder keeps the architecture consistent with the image branch.

#### Baseline 2: Image-only model

Options:

- CLIP image encoder plus MLP classifier.
- Frozen CLIP image encoder initially, optional fine-tuning later.

This tests whether visual content alone is predictive.

#### Baseline 3: Multimodal CLIP fusion model

Recommended main model:

```text
text = CLIP_TextEncoder(tokens)
image = CLIP_ImageEncoder(image)

features = concat(
    text,
    image,
    abs(text - image),
    text * image
)

fused = MLP(LayerNorm(features))
logit = Linear(fused)
prediction = sigmoid(logit)
```

This is simple, stable, and strong enough for a course project. The `abs(text - image)` and `text * image` terms help the classifier capture interactions rather than only concatenating modalities.

#### Robust model

Use the same architecture as the multimodal baseline, but train with clean and perturbed views.

Recommended objective:

```text
L_total = CE(y, p_clean)
        + alpha * CE(y, p_perturbed)
        + beta * KL(p_clean || p_perturbed)
```

Where:

- `CE` is binary cross-entropy.
- `KL` encourages prediction consistency between clean and perturbed views.
- `alpha` controls supervised loss on attacked examples.
- `beta` controls consistency strength.

Start with `alpha = 1.0` and `beta = 0.5`, then tune lightly.

### Training strategy

Recommended training stages:

1. Freeze CLIP encoders and train only the classifier/fusion head.
2. If the baseline underfits, unfreeze the last CLIP transformer block or use LoRA/adapters.
3. Train robust variant using random perturbation augmentation.
4. Keep model capacity modest to avoid spending the project on compute.

Recommended loss:

- Weighted binary cross-entropy if the class distribution is imbalanced.
- Optional focal loss only if false negatives dominate and weighted BCE is insufficient.

Recommended thresholding:

- Select the decision threshold on validation data to maximize macro F1 or balanced accuracy.
- Reuse the same threshold when evaluating attacked versions.

### Perturbation suite

#### Text perturbations

Use deterministic and reproducible transformations. Examples:

| Attack | Description | Severity examples |
|---|---|---|
| Leetspeak | Replace letters with visually similar symbols. | `a -> @`, `i -> 1`, `o -> 0` |
| Character insertion | Insert punctuation or spaces inside words. | `hate -> h.a.t.e`, `hate -> h a t e` |
| Character deletion | Remove a small fraction of characters. | 5%, 10%, 20% |
| Character swap | Swap adjacent characters. | 1, 2, 3 swaps |
| Case and punctuation noise | Random casing and punctuation changes. | low, medium, high |
| Censoring | Replace internal letters with `*`. | `hate -> h*te`, `idiot -> id***` |
| Keyboard typo | Replace characters with nearby keyboard keys. | low, medium, high |

Important: do not apply transformations only to offensive words unless the offensive vocabulary is carefully defined. A general perturbation function avoids hardcoding slurs in code and reduces ethical issues.

#### Image perturbations

Use standard image corruptions:

| Attack | Description | Severity examples |
|---|---|---|
| Gaussian noise | Add random pixel noise. | sigma 0.01, 0.03, 0.05 |
| Gaussian blur | Blur the image. | kernel 3, 5, 7 |
| JPEG compression | Reduce image quality. | quality 80, 50, 25 |
| Brightness shift | Change brightness. | +/-10%, +/-25%, +/-40% |
| Contrast shift | Change contrast. | +/-10%, +/-25%, +/-40% |
| Translation/crop | Slightly shift or crop image. | 2%, 5%, 10% |
| Text-region occlusion | Mask a small region likely to contain text. | optional, only if OCR/bounding boxes are available |

#### White-box image attack option

If time permits, add FGSM or PGD:

```text
x_adv = x + epsilon * sign(grad_x CE(y, f(x, text)))
```

Recommended epsilons should be small and reported in normalized image scale. Keep this as a stretch goal because it requires care with image normalization, clipping, and differentiability through the image encoder.

#### Combined attacks

After single-modality attacks work, evaluate combined attacks:

- One text perturbation plus one image perturbation.
- Worst text perturbation plus worst image perturbation.
- Random text/image perturbation pair.

Combined attacks are useful because real adversaries can edit both meme text and image appearance.

### Evaluation metrics

Report clean and attacked metrics using the same split.

Core metrics:

- Accuracy.
- Macro F1.
- AUROC.
- Precision.
- Recall.
- False positive rate.
- False negative rate.

Robustness metrics:

- **Robustness gap:** `clean_metric - attacked_metric`.
- **Attack success rate:** fraction of clean-correct examples flipped by attack.
- **Worst-case accuracy/F1:** performance under the strongest attack per example or per attack family.
- **Area under robustness curve:** average performance over severity levels.

Modality metrics:

- Text attack degradation versus image attack degradation.
- Full model versus text-only versus image-only.
- Prediction flip rate when only one modality is perturbed.

Calibration metrics, optional:

- Expected calibration error.
- Reliability diagram.
- Confidence shift under attack.

### Result tables to include in the final report

Recommended final tables:

1. Clean baseline comparison.
2. Text attack results by attack and severity.
3. Image attack results by attack and severity.
4. Combined attack results.
5. Robust training before/after comparison.
6. Ablation table: text-only, image-only, multimodal, robust multimodal.

Recommended figures:

1. Robustness curve across severity levels.
2. Bar chart of attack success rate by attack family.
3. Clean versus attacked confusion matrices.
4. Qualitative examples of model failures.

### Core experiment matrix

The minimum experiment matrix should be small enough to finish but complete enough to support the conclusions.

| Model | Clean eval | Text attacks | Image attacks | Combined attacks | Robust training |
|---|---:|---:|---:|---:|---:|
| Majority baseline | yes | no | no | no | no |
| Text-only CLIP/BERT | yes | yes | no | no | no |
| Image-only CLIP | yes | no | yes | no | no |
| Multimodal CLIP fusion | yes | yes | yes | yes | no |
| Robust multimodal CLIP fusion | yes | yes | yes | yes | yes |

Recommended minimum number of reported runs:

- 3 random seeds for the main multimodal baseline if compute allows.
- 1 to 3 random seeds for unimodal baselines.
- 1 to 3 random seeds for the robust model.
- All attacks evaluated deterministically once per trained checkpoint, or with fixed perturbation seeds if attacks are stochastic.

If compute is limited, prioritize multiple seeds for the main clean and robust multimodal models rather than for every baseline.

## Dataset plan

### Primary dataset: Hateful Memes

Use Hateful Memes as the primary dataset because it was designed for exactly this kind of multimodal reasoning. It includes image-text pairs where the image or text alone can be misleading.

Recommended usage:

- Train on the official training split.
- Use a validation split for threshold selection and early stopping.
- Use the official development split as final evaluation if test labels are not available.
- Keep the hidden test set untouched if labels are unavailable.

### Optional secondary datasets

Use only one secondary dataset if time remains.

| Dataset | Best use | Risk |
|---|---|---|
| MultiOFF | External offensive meme robustness test. | "Offensive" is not identical to "hateful." |
| MAMI | Robustness on misogynistic memes. | Narrower label definition and access request needed. |
| HarMeme | Target-aware harmful meme analysis. | Different label taxonomy. |
| MMHS150K | Larger noisy multimodal data. | More preprocessing and possible missing images. |

Recommendation:

- First choice: **MultiOFF** if easy to access, because it is lightweight and meme-focused.
- Second choice: **MAMI** if the team wants a gender-hate focus and can get access early.
- Avoid making MMHS150K core unless the group has extra time for data cleaning.

## Implementation roadmap

### Phase 0: Project setup and ethical handling

Deliverables:

- Repository structure.
- Dataset access checklist.
- Short ethics note.
- Experiment tracking setup.

Tasks:

1. Create project repository directories:
   - `data/`
   - `src/`
   - `configs/`
   - `experiments/`
   - `notebooks/`
   - `reports/`
2. Add a README describing how to reproduce experiments.
3. Add `.gitignore` entries for datasets, model checkpoints, and generated attacked images.
4. Document dataset license restrictions and do not redistribute raw hateful content.
5. Decide how hateful/offensive examples will be shown in the final report. Prefer blurred or paraphrased examples when possible.

### Phase 1: Dataset acquisition and preprocessing

Deliverables:

- Working Hateful Memes data loader.
- Clean train/validation/evaluation split.
- Dataset statistics table.

Tasks:

1. Register for Hateful Memes and download the dataset.
2. Parse JSONL metadata into a standard dataframe.
3. Load images and text with consistent transforms.
4. Compute basic statistics:
   - number of examples per split;
   - class balance;
   - text length distribution;
   - image size distribution.
5. Build deterministic train/validation split if needed.
6. Add seed control for reproducibility.

Definition of done:

- A script can load a batch of images, texts, and labels.
- A small sanity training run works on 100 examples.

### Phase 2: Clean baselines

Deliverables:

- Majority, text-only, image-only, and multimodal baseline results.
- Saved model checkpoints.
- Clean result table.

Tasks:

1. Implement majority baseline.
2. Implement text-only CLIP baseline.
3. Implement image-only CLIP baseline.
4. Implement multimodal CLIP fusion baseline.
5. Train each model with early stopping.
6. Tune classification threshold on validation data.
7. Evaluate all clean baselines on the same evaluation split.

Definition of done:

- The multimodal baseline trains reliably.
- The result table includes accuracy, macro F1, AUROC, precision, recall, FPR, and FNR.
- The multimodal model is compared against unimodal baselines.

### Phase 3: Perturbation benchmark

Deliverables:

- Reproducible text and image attack functions.
- Attack metadata format.
- Visual/textual sanity-check samples.

Tasks:

1. Implement text perturbations:
   - leetspeak;
   - spacing/punctuation insertion;
   - character deletion;
   - character swap;
   - casing and punctuation noise;
   - censoring;
   - keyboard typo.
2. Implement image perturbations:
   - Gaussian noise;
   - blur;
   - JPEG compression;
   - brightness;
   - contrast;
   - translation/crop.
3. Define severity levels.
4. Add deterministic seeds for perturbations.
5. Generate a small inspection set of perturbed examples.
6. Manually check 50 to 100 examples for label preservation.

Definition of done:

- Each attack can be applied independently.
- Perturbed examples preserve labels at low and medium severity.
- The benchmark records attack name, modality, severity, and random seed.

### Phase 4: Robustness evaluation

Deliverables:

- Full attack result tables.
- Robustness curves.
- Attack success rate analysis.

Tasks:

1. Run every attack against the trained text-only, image-only, and multimodal models.
2. Evaluate each attack at each severity level.
3. Compute attack success rate only on clean-correct examples.
4. Compute robustness gap for macro F1 and AUROC.
5. Compute worst-case performance across text attacks, image attacks, and all attacks.
6. Compare which modality is more fragile.

Definition of done:

- The report can answer which attack family is most damaging.
- The results separate text-only, image-only, and multimodal vulnerabilities.

### Phase 5: Robustness-aware training

Deliverables:

- Robust multimodal model.
- Before/after comparison.
- Ablation on augmentation or consistency loss if time permits.

Tasks:

1. Add random perturbation augmentation during training.
2. For each clean training example, sample either:
   - clean view only;
   - text-perturbed view;
   - image-perturbed view;
   - combined perturbed view.
3. Train with cross-entropy on clean and perturbed views.
4. Add consistency regularization between clean and perturbed predictions.
5. Evaluate the robust model using the same attack benchmark.
6. Compare:
   - clean performance drop;
   - attacked performance gain;
   - ASR reduction;
   - worst-case improvement.

Definition of done:

- There is a clear table comparing clean-trained and robust-trained multimodal models.
- The team can say whether robustness improved and what it cost.

### Phase 6: Ablations and analysis

Deliverables:

- Ablation table.
- Qualitative failure analysis.
- Modality-dependence analysis.

Tasks:

1. Compare fusion variants:
   - concatenation only;
   - concatenation plus absolute difference/product;
   - optional gated fusion.
2. Compare training variants:
   - clean only;
   - augmentation only;
   - augmentation plus consistency.
3. Analyze false positives and false negatives.
4. Select at least 20 representative examples:
   - clean-correct but attacked-wrong;
   - clean-wrong;
   - robust model fixed;
   - robust model still failed.
5. For each example, record likely reason:
   - text obfuscation;
   - image corruption;
   - multimodal reasoning failure;
   - over-reliance on protected-group mention;
   - threshold/calibration issue.

Definition of done:

- The final report has more than metrics: it explains failure patterns.

### Phase 7: Optional secondary dataset evaluation

Deliverables:

- External robustness table.
- Discussion of label mismatch limitations.

Tasks:

1. Choose one secondary dataset only.
2. Map labels to a binary harmful/offensive class carefully.
3. Evaluate zero-shot or fine-tuned transfer.
4. Run a reduced perturbation suite.
5. Report that label definitions differ and avoid overclaiming.

Definition of done:

- Secondary dataset results are clearly marked as external generalization, not the main benchmark.

### Phase 8: Final report and poster

Deliverables:

- Final paper.
- Poster.
- Reproducibility appendix.

Tasks:

1. Write the introduction around robustness and safer online spaces.
2. Explain why multimodal meme detection is vulnerable.
3. Present model architecture and perturbation benchmark.
4. Report clean and attacked results.
5. Discuss defense results.
6. Include qualitative examples.
7. State limitations and ethical considerations.
8. Prepare poster figures:
   - architecture diagram;
   - attack success rate chart;
   - robust training comparison;
   - example failure cases.

Definition of done:

- A reader can understand the model, reproduce the evaluation, and see the robustness tradeoffs.

## Suggested timeline

| Week | Main goal | Deliverable |
|---:|---|---|
| 1 | Setup and data access | Data loader, stats, ethics note |
| 2 | Clean baselines | Text-only, image-only, multimodal results |
| 3 | Perturbation suite | Text/image attacks with severity levels |
| 4 | Robustness benchmark | Full attacked evaluation tables |
| 5 | Robust training | Augmentation/consistency model |
| 6 | Ablations | Fusion/training ablations and modality analysis |
| 7 | Optional external eval and qualitative analysis | Secondary dataset or deeper failure study |
| 8 | Final writing | Report, poster, reproducibility appendix |

If the actual project window is shorter, remove Week 7 and keep only the core Hateful Memes experiments.

## Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Dataset access delay | Hateful Memes requires registration. | Request access immediately; use MultiOFF as fallback. |
| Too many datasets | Label definitions differ and preprocessing grows. | Keep Hateful Memes primary; one optional secondary dataset only. |
| Perturbations change labels | Invalidates robustness conclusions. | Manual sanity check and severity control. |
| Model learns text shortcuts | Multimodal claim becomes weak. | Include unimodal baselines and modality ablations. |
| Clean performance is low | Robustness results become less meaningful. | Start with frozen CLIP, then fine-tune last layers if needed. |
| Defense hurts clean performance | Robustness/accuracy tradeoff may be poor. | Report the tradeoff honestly; tune augmentation severity. |
| Attack suite is too broad | Implementation consumes the project. | Implement simple deterministic attacks first, add PGD only as stretch. |
| Ethical handling of hateful content | Reports may expose harmful text/images. | Avoid unnecessary reproduction of hateful content; blur or paraphrase examples. |

## Ethical and safety considerations

This project handles hateful and offensive content. The final report should explicitly state:

- The model is a research prototype, not a deployable moderation system.
- Dataset licenses and access restrictions are respected.
- Raw hateful memes should not be redistributed.
- Examples in the report should be minimized, blurred, or paraphrased where possible.
- The system may encode dataset biases and should not be used for real moderation decisions without fairness evaluation and human oversight.
- Robustness to obfuscation is dual-use: it can help moderation systems, but it also studies adversarial behavior. Avoid publishing attack code in a way that targets real platforms.
- Follow the EPFL data planning guidance linked in `Project_Info.md`.
- The proposed project does not require collecting new human-subject data. If the team decides to run a human annotation or user study beyond a small internal sanity check, consult the course staff and the EPFL ethics guidance before starting.

## Final recommended project specification

### Final title

**Robustness of Multimodal Hate Meme Detection Under Text and Image Perturbations**

### Final objective

Develop and evaluate a CLIP-based multimodal hate meme classifier, quantify its vulnerability to realistic and adversarial perturbations in text and image modalities, and test whether robustness-aware training improves performance under attack.

### Final deliverables

1. A reproducible CLIP-based multimodal classification pipeline.
2. Text-only, image-only, and multimodal clean baselines.
3. A perturbation benchmark with at least 10 attack types across text and image.
4. Robustness metrics and severity curves.
5. A robust-trained model using augmentation and consistency regularization.
6. Ablation and modality-dependence analysis.
7. Qualitative failure analysis.
8. Final report and poster.

### Minimum viable contribution

If time becomes tight, the project is still complete if it delivers:

- Hateful Memes only.
- Frozen CLIP encoders.
- Text-only, image-only, and multimodal baselines.
- At least 5 text attacks and 5 image corruptions.
- Clean versus attacked robustness table.
- One robust training method.
- A clear analysis of which modality is most vulnerable.

### Best-case contribution

If everything goes smoothly, the project can additionally deliver:

- FGSM or PGD white-box image attack.
- Combined text-image attacks.
- Secondary dataset transfer evaluation.
- Calibration analysis.
- Lightweight visual explanation via occlusion sensitivity.

## Why this fits the course goals

The project directly supports the course theme, "Deep learning to foster safer online spaces," by studying whether hate detection systems remain reliable when harmful content is obfuscated. It also addresses the course objective of context comprehension: the model must reason over image and text together, and the evaluation tests whether that reasoning is robust.

The project has enough deep learning substance:

- pretrained multimodal encoders;
- fusion architecture design;
- adversarial/robustness evaluation;
- robustness-aware loss design;
- model ablations;
- error analysis.

At the same time, it avoids thesis-level scale by not attempting to solve all of multimodal hate speech detection. The result should be a strong mini-project: bounded, measurable, technically meaningful, and aligned with safer online interaction.
