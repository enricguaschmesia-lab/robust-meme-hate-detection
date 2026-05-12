# Baseline Model Architecture and Setup

**Project:** Adversarial Robustness of Multimodal Meme Hate Detectors (EE-559).
**Purpose of this document:** complete, implementation-level specification of the baseline model and the steps required to take a fresh clone of the repository to a trained, evaluation-ready, attack-compatible classifier. This is the blueprint for Phase 1-2 of the roadmap.

---

## 1. Architecture overview

A dual-encoder CLIP-based multimodal classifier with a lightweight fusion head. Fully differentiable from raw pixels and tokens, so white-box image attacks (FGSM/PGD) and consistency-loss training operate on the same forward graph used for clean prediction.

```text
                                meme = (image, text)
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │                                                       │
       Text branch                                              Image branch
              │                                                       │
   tokenize(text) -> ids               resize 224x224 -> ToTensor[0,1] -> normalize(μ,σ)
              │                                                       │
   CLIP_TextEncoder (ViT-B/32)                            CLIP_ImageEncoder (ViT-B/32)
              │                                                       │
       t ∈ ℝ^{512}                                              v ∈ ℝ^{512}
              │                                                       │
              └───────────────────────┬───────────────────────────────┘
                                      ▼
                  Fusion: concat(t, v, |t - v|, t ⊙ v)  ∈ ℝ^{2048}
                                      ▼
                  LayerNorm -> Dropout(0.2) -> Linear(2048→512) -> GELU
                                      ▼
                                Dropout(0.2) -> Linear(512→1)
                                      ▼
                          logit  (BCEWithLogits during training)
                                      ▼
                       sigmoid(logit) -> p(hateful)
```

Key properties:

- **Single model, single checkpoint** - no ensembling at any stage.
- **Encoders frozen by default** in Stage 1 of training. Only the fusion head trains.
- **Optional LoRA / last-block unfreeze** in Stage 2 if Stage 1 plateaus.
- **Image normalization is a Module inside the model**, not in the dataloader. This makes the input to `model.forward` a tensor in `[0, 1]` that can have `requires_grad=True`, which is what FGSM/PGD need.

---

## 2. Component specification

### 2.1 Image encoder

- **Model:** OpenCLIP `ViT-B/32`, pretrained `laion2b_s34b_b79k`.
- **Input:** float tensor of shape `(B, 3, 224, 224)` in range `[0, 1]`.
- **Output:** float tensor of shape `(B, 512)`. Not L2-normalized (we use raw features for the classifier).
- **Frozen:** all parameters by default (`param.requires_grad = False`).
- **First ablation (not stretch):** ViT-L/14 (output dim 768). With the cluster's A100 80 GB this fits trivially even with both encoders held in memory. Run it as a parallel headline result alongside ViT-B/32. ViT-B/32 stays the *primary* only because we want fast iteration over the perturbation grid (10+ attacks × 3 severities × dev set, repeated across seeds and the robust variant).

### 2.2 Text encoder

- **Model:** OpenCLIP ViT-B/32 text transformer.
- **Input:** integer tensor of shape `(B, 77)` from `open_clip.get_tokenizer('ViT-B-32')`.
- **Output:** float tensor of shape `(B, 512)`. Not L2-normalized.
- **Frozen:** by default.

### 2.3 Image normalization module

Wrapped inside the model so it sits on the autograd graph.

- **Constants:** OpenAI-CLIP statistics
  - `mean = (0.48145466, 0.4578275, 0.40821073)`
  - `std  = (0.26862954, 0.26130258, 0.27577711)`
- **Computation:** `x_norm = (x - mean) / std`, broadcast over the channel dimension.

### 2.4 Fusion layer

Element-wise interaction features capture cross-modal compositionality without cross-attention.

- **Input:** `t ∈ ℝ^{B×512}`, `v ∈ ℝ^{B×512}`.
- **Output:** `f = [t, v, |t - v|, t ⊙ v] ∈ ℝ^{B×2048}`.

### 2.5 Classification head

- **LayerNorm(2048)** - stabilises across the four sub-blocks.
- **Dropout(p=0.2)**.
- **Linear(2048 → 512)** + **GELU**.
- **Dropout(p=0.2)**.
- **Linear(512 → 1)** - raw logit; sigmoid is applied in the loss (`BCEWithLogitsLoss`) and in inference.

### 2.6 Parameter count (defaults)

| Module | Params (frozen + trainable) |
|---|---:|
| CLIP image encoder ViT-B/32 | ~88 M (frozen) |
| CLIP text encoder ViT-B/32 | ~63 M (frozen) |
| Fusion + head | ~1.3 M (trainable) |
| **Total** | **~152 M, of which ~1.3 M trainable in Stage 1** |

---

## 3. Model pseudocode (PyTorch)

```python
# src/robust_meme_hate_detection/models/clip_fusion.py
import torch, torch.nn as nn, torch.nn.functional as F
import open_clip


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1))

    def forward(self, x):                          # x in [0, 1]
        return (x - self.mean) / self.std


class FusionHead(nn.Module):
    def __init__(self, dim=512, hidden=512, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, v):
        f = torch.cat([t, v, (t - v).abs(), t * v], dim=-1)
        return self.mlp(self.norm(f)).squeeze(-1)  # (B,) raw logit


class CLIPHateMemeClassifier(nn.Module):
    def __init__(
        self,
        clip_arch="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
        freeze_encoders=True,
        head_hidden=512,
        head_dropout=0.2,
    ):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_arch, pretrained=clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_arch)

        # Detach the visual / text submodules from the CLIP wrapper so we
        # can call them directly and apply our own normalisation outside.
        self.visual = clip_model.visual
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask

        self.normalize = Normalize(CLIP_MEAN, CLIP_STD)
        self.head = FusionHead(dim=self.visual.output_dim, hidden=head_hidden,
                               dropout=head_dropout)

        if freeze_encoders:
            for p in self.visual.parameters():        p.requires_grad = False
            for p in self.transformer.parameters():   p.requires_grad = False
            self.token_embedding.weight.requires_grad = False
            self.positional_embedding.requires_grad   = False
            for p in self.ln_final.parameters():      p.requires_grad = False
            self.text_projection.requires_grad        = False

    # ---------------- encoding helpers ----------------
    def encode_image(self, images01):                # images01: (B,3,224,224) in [0,1]
        x = self.normalize(images01)
        return self.visual(x)                        # (B, 512)

    def encode_text(self, token_ids):                # (B, 77) longs
        x = self.token_embedding(token_ids)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)                       # (S, B, D)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        # CLIP text feature = features at the [EOS] / argmax position
        eot = token_ids.argmax(dim=-1)
        x = x[torch.arange(x.size(0)), eot] @ self.text_projection
        return x                                     # (B, 512)

    # ---------------- forward ----------------
    def forward(self, images01, token_ids):
        v = self.encode_image(images01)
        t = self.encode_text(token_ids)
        return self.head(t, v)                       # (B,) raw logit

    # ---------------- ablation modes ----------------
    def forward_text_only(self, token_ids):
        t = self.encode_text(token_ids)
        v = torch.zeros_like(t)
        return self.head(t, v)

    def forward_image_only(self, images01):
        v = self.encode_image(images01)
        t = torch.zeros_like(v)
        return self.head(t, v)
```

Notes:
- `encode_image` accepts a `[0, 1]` tensor. This is the contract that keeps the model attack-ready.
- Text-only / image-only forwards zero-out the missing branch. This satisfies the unimodal-baseline requirement of the roadmap without training three separate models. (We may still train dedicated unimodal heads for stronger numbers; both options are supported by this class.)

---

## 4. Data pipeline

### 4.1 Hateful Memes layout (cluster mirror)

The dataset is **already staged on the EPFL group scratch** and visible from inside Run:AI jobs at `/scratch/datasets/hate_meta`. No registration, no download, no rsync.

```text
/scratch/datasets/hate_meta/
├── img/              # 10,000 .png files
├── train.jsonl       # 8,500 rows, all labeled  (3,050 hateful / 5,450 non-hateful)
├── dev.jsonl         #   500 rows, all labeled  (250 / 250, balanced)
├── test.jsonl        # 1,000 rows, UNLABELED
├── README.md
└── LICENSE.txt
```

Each line: `{"id": 42953, "img": "img/42953.png", "label": 0|1, "text": "..."}`.

This is the public mirror layout, not the original Meta release. Two consequences:

1. There is **one** `dev.jsonl`, not separate `dev_seen` / `dev_unseen`. References elsewhere in this document use `dev.jsonl`.
2. `test.jsonl` has **no `label` field**. It cannot be used for supervised evaluation; it is only useful for inference-time submissions or qualitative inspection. See §4.5 for the held-out-from-train policy that replaces a labeled test set.

The existing loader at `src/robust_meme_hate_detection/data/hateful_memes.py` already handles this layout (it accepts both `dev` and `dev_seen` aliases).

### 4.2 Image preprocessing (deliberately differentiable-friendly)

We **do not** use OpenCLIP's default `preprocess_val` for training/eval. That pipeline goes PIL→tensor and includes a non-differentiable resize. Instead:

1. **In the dataloader:** PIL resize to 224 (bicubic, antialias) → CenterCrop(224) → `ToTensor()` (this yields a float tensor in `[0, 1]`).
2. **In the model:** Normalize(mean, std) as a `nn.Module` (differentiable).
3. **Train-time augmentation only** (Stage 3 robust variant): apply photometric/geometric augmentations *before* `ToTensor`, leaving the `[0, 1]` invariant.

This way `model(images01, token_ids)` is fully differentiable w.r.t. `images01`, so PGD just does:

```python
images01.requires_grad_(True)
logit = model(images01, token_ids)
loss = F.binary_cross_entropy_with_logits(logit, labels)
grad = torch.autograd.grad(loss, images01)[0]
images01_adv = (images01 + epsilon * grad.sign()).clamp(0, 1)
```

### 4.3 Text preprocessing

- **Tokenizer:** `open_clip.get_tokenizer('ViT-B-32')`, returning `(B, 77)` longs.
- Truncation/padding handled by the tokenizer.
- Text perturbations are applied at the **string level** *before* tokenization (leetspeak, character swaps, etc.). Text attacks remain discrete and do not require backprop through the embedding lookup for the scope of this project.

### 4.4 Dataset class

The repo already has `src/robust_meme_hate_detection/data/hateful_memes.py` with a richer `HatefulMemesDataset` and a `build_hateful_memes_dataloader` factory. It accepts arbitrary `image_transform` and `text_transform` callables. We do **not** rewrite it; we configure it with two callables that produce the exact contract this model expects:

- `image_transform`: PIL → resize 224 → CenterCrop → ToTensor → `(3, 224, 224)` tensor in `[0, 1]`. Note: **no Normalize here** — that lives inside the model (§2.3) so attacks see raw pixels.
- `text_transform`: str → `open_clip` tokenizer → `(77,)` long tensor.

For documentation and tests, the contract a batch must satisfy is what the following minimal class shows:

```python
# (contract reference; the actual loader is the existing one)
import json, pathlib
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class HatefulMemesDataset(Dataset):
    """Returns (image01, token_ids, label, meta).
    image01: float tensor (3, 224, 224) in [0, 1]
    token_ids: long tensor (77,)
    label: float tensor scalar
    meta: dict with 'id', 'orig_text', and optional perturbation tags
    """

    def __init__(self, jsonl_path, image_dir, tokenizer,
                 text_perturb=None, image_perturb=None, train=False):
        self.records   = [json.loads(l) for l in open(jsonl_path)]
        self.image_dir = pathlib.Path(image_dir)
        self.tokenizer = tokenizer
        self.text_perturb  = text_perturb
        self.image_perturb = image_perturb

        ops = [T.Resize(224, interpolation=T.InterpolationMode.BICUBIC,
                        antialias=True),
               T.CenterCrop(224)]
        if train:
            ops += [T.RandomHorizontalFlip(p=0.0)]   # off by default; memes have text
        ops += [T.ToTensor()]                         # -> [0, 1]
        self.image_tfm = T.Compose(ops)

    def __len__(self):  return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.image_dir / r["img"].replace("img/", "")).convert("RGB")
        text = r["text"]
        if self.image_perturb is not None: img  = self.image_perturb(img)
        if self.text_perturb  is not None: text = self.text_perturb(text)
        image01 = self.image_tfm(img)                                # (3,224,224)
        token_ids = self.tokenizer([text])[0]                        # (77,)
        label = torch.tensor(float(r.get("label", 0)))
        meta = {"id": r["id"], "orig_text": r["text"]}
        return image01, token_ids, label, meta
```

### 4.5 Split policy (handling the unlabeled test set)

Because `test.jsonl` is unlabeled in the public mirror, we cannot do supervised evaluation on it. Recommended split policy:

| Logical split | File / source | Size | Purpose |
|---|---|---:|---|
| `train` | first 8 000 rows of `train.jsonl` (deterministic) | 8 000 | gradient steps |
| `train_held_out` | last 500 rows of `train.jsonl` (stratified, fixed seed) | 500 | **final clean and attacked numbers reported in the paper** |
| `dev` | `dev.jsonl` | 500 | early-stopping, threshold tuning, attack-grid sweeps during development |
| `test` | `test.jsonl` (unlabeled) | 1 000 | inference-only, kept untouched for any future external benchmark |

The 8 000 / 500 split of training data is computed once with a fixed seed and saved to `data/processed/splits/train_split.json` so that all seeds and Stage 1/2/3 runs use exactly the same partition. Class balance is preserved (~36% positive in both halves).

Reporting convention:

- **Development numbers** (the ones the team uses to iterate) come from `dev.jsonl`.
- **Final reported numbers** (clean, per-attack, per-severity, robust-vs-clean delta) come from `train_held_out`.
- Never tune anything against `train_held_out`. It is touched exactly once per checkpoint.

---

## 5. Training pipeline

### 5.1 Loss, optimizer, scheduler

- **Loss:** `BCEWithLogitsLoss(pos_weight=w)` where `w = N_neg / N_pos` on the training split. With the cluster mirror that gives `w = 5450 / 3050 ≈ 1.79` (~36% positive).
- **Optimizer:** AdamW. Learning rates per stage below.
- **Scheduler:** cosine schedule with 10% linear warmup, restarts disabled.
- **Batch size:** **128 by default** on the A100 80 GB (was 64 in the earlier draft; revised after the cluster check confirmed an A100 80 GB). 256 is also feasible if we want to test it; if so, scale head LR linearly (2e-3). Stay at 64 if profiling something memory-intensive.
- **Epochs:** 10 with early stopping on `dev` macro F1, patience 3.
- **Mixed precision:** `torch.cuda.amp.autocast(dtype=torch.bfloat16)`. The cluster's A100 fully supports bf16 with torch 2.10 + CUDA 12.8.
- **Reproducibility:** seed PyTorch, NumPy, Python `random`. Run 3 seeds for the main multimodal and robust-multimodal models.

### 5.2 Stage 1 - frozen encoders, train head only

- Trainable: fusion head only (~1.3 M params).
- Head LR: `1e-3` at batch 128. Weight decay: `1e-2`.
- Expected outcome: ~0.72-0.76 AUROC on `dev`.
- **Wall-clock on A100 80 GB:** under 30 minutes for the full 10-epoch run (frozen encoders, ~63 steps/epoch at batch 128). The earlier "roughly an hour on a single modern GPU" estimate was conservative for laptop-class GPUs; revised after the cluster check.
- **Total compute envelope** for the full pass (clean + robust × 3 seeds, both backbones): roughly **one A100-day**, not "a few GPU-days".

```python
def train_stage1(model, train_loader, val_loader, epochs=10, lr=1e-3):
    params = [p for p in model.head.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = cosine_with_warmup(optim, total_steps=epochs * len(train_loader),
                               warmup_steps=int(0.1 * epochs * len(train_loader)))
    pos_weight = compute_pos_weight(train_loader.dataset)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = -1
    for epoch in range(epochs):
        model.train()
        for image01, tokens, label, _ in train_loader:
            image01, tokens, label = to_device(image01, tokens, label)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logit = model(image01, tokens)
                loss  = loss_fn(logit, label)
            optim.zero_grad(); loss.backward(); optim.step(); sched.step()

        f1 = evaluate(model, val_loader)["macro_f1"]
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), "ckpt/stage1_best.pt")
```

### 5.3 Stage 2 (optional) - last-block unfreeze (LoRA only as ablation)

Trigger only if Stage 1 plateaus below ~0.72 AUROC.

**Default in this revision: last-block unfreeze.** With 80 GB of VRAM, full backprop through the last 2 residual blocks of each encoder is essentially free, so there is no memory-driven reason to add LoRA's complexity. Set `requires_grad=True` for:

- the last 2 residual blocks of `self.visual.transformer`,
- the last 2 residual blocks of `self.transformer`,
- `self.ln_final` and `self.text_projection`.

Hyperparameters: encoder LR `1e-5`, head LR unchanged (`1e-3`), weight decay `1e-2` (raise to `5e-2` if validation curves show overfitting on the small 8000-sample train split).

**LoRA, kept as a regularisation ablation.** With only 8 000 training examples, LoRA may still help by limiting effective parameter count. If we run it, apply rank-8 LoRA (`alpha=16`, `lora_dropout=0.1`) to `q_proj`/`k_proj`/`v_proj` of the last 2 transformer blocks of each encoder, encoder-LoRA LR `5e-5`.

> Implementation note: `peft` does not plug directly into `open_clip` because the attention naming differs from HuggingFace CLIP. The cleanest path is a small custom `LoRALinear` wrapping the relevant `Linear` modules, attached at model construction time. Switching to `transformers.CLIPModel` is no longer needed since LoRA is now optional rather than the default.

### 5.4 Stage 3 - robust variant

Each minibatch produces both a clean view and a perturbed view; we sum supervised losses and add a KL-consistency term.

- **Per-example sampling:** with prob 1/3 each, apply text-only, image-only, or combined perturbations.
- **Loss:**
  ```text
  L = BCE(y, p_clean) + α · BCE(y, p_perturbed) + β · KL(p_clean || p_perturbed)
  ```
  Defaults: `α = 1.0`, `β = 0.5`. Tune lightly.
- **Detached target for consistency:** `p_clean` is detached on the KL side so the consistency gradient flows only through the perturbed branch (standard for FixMatch-style consistency).

```python
def robust_step(model, batch, perturb, loss_fn, alpha=1.0, beta=0.5):
    image01, tokens, label, _ = batch
    image01p, tokensp = perturb(image01, tokens)         # see §6.2
    logit_c = model(image01,  tokens)
    logit_p = model(image01p, tokensp)
    p_c = torch.sigmoid(logit_c)
    p_p = torch.sigmoid(logit_p)
    bce_c = loss_fn(logit_c, label)
    bce_p = loss_fn(logit_p, label)
    kl = F.binary_cross_entropy(p_p, p_c.detach(), reduction="mean")
    return bce_c + alpha * bce_p + beta * kl
```

### 5.5 Threshold tuning

After training, sweep thresholds `τ ∈ {0.30, 0.31, ..., 0.70}` on the validation split and pick the one that maximises macro F1. Reuse the same `τ` for all attacked evaluations of that checkpoint.

---

## 6. Inference and evaluation

### 6.1 Clean inference

```python
@torch.no_grad()
def predict(model, image01, tokens):
    logit = model(image01, tokens)
    return torch.sigmoid(logit)
```

### 6.2 Attack-ready inference (gradient access)

```python
def fgsm_image(model, image01, tokens, label, epsilon, loss_fn):
    image01 = image01.clone().detach().requires_grad_(True)
    logit = model(image01, tokens)
    loss = loss_fn(logit, label)
    grad = torch.autograd.grad(loss, image01)[0]
    return (image01 + epsilon * grad.sign()).clamp(0, 1).detach()


def pgd_image(model, image01, tokens, label, epsilon, alpha, steps, loss_fn):
    delta = torch.zeros_like(image01).uniform_(-epsilon, epsilon)
    delta = delta.detach().requires_grad_(True)
    for _ in range(steps):
        x_adv = (image01 + delta).clamp(0, 1)
        loss = loss_fn(model(x_adv, tokens), label)
        g = torch.autograd.grad(loss, delta)[0]
        delta = (delta + alpha * g.sign()).clamp(-epsilon, epsilon).detach().requires_grad_(True)
    return (image01 + delta).clamp(0, 1).detach()
```

Text attacks remain string-level and therefore go through the dataset's `text_perturb` callable; no gradient is needed.

### 6.3 Metrics

For each (model, attack, severity) cell of the experiment matrix:

- Accuracy, macro F1, AUROC, precision, recall, FPR, FNR.
- Robustness gap: `clean_metric - attacked_metric`.
- Attack success rate: `# clean-correct ∧ attacked-wrong / # clean-correct`.
- Worst-case F1 across attack families.

---

## 7. Repository structure

The repo uses an `src/`-layout package called `robust_meme_hate_detection`. Some directories already exist (marked ✅); others are added by this plan (marked ➕).

```text
robust-meme-hate-detection/
├── configs/
│   ├── data/
│   │   └── hateful_memes.yaml         ✅  exists; will be updated to point to /scratch/datasets/hate_meta
│   ├── stage1.yaml                    ➕  frozen-encoder head training
│   ├── stage2_unfreeze.yaml           ➕  last-2-block unfreeze
│   ├── stage2_lora.yaml               ➕  optional LoRA ablation
│   ├── stage3_robust.yaml             ➕  robust training
│   └── eval.yaml                      ➕  attack grid
├── src/robust_meme_hate_detection/
│   ├── __init__.py                    ✅
│   ├── data/
│   │   ├── __init__.py                ✅
│   │   └── hateful_memes.py           ✅  HatefulMemesDataset + dataloader factory
│   ├── models/
│   │   └── clip_fusion.py             ➕  §3 implementation
│   ├── perturb/
│   │   ├── text.py                    ➕  leetspeak, swap, censor, typos
│   │   └── image.py                   ➕  blur, noise, JPEG, brightness, contrast
│   ├── attacks/
│   │   └── pgd.py                     ➕  FGSM / PGD on image01
│   ├── train/
│   │   ├── stage1.py                  ➕
│   │   ├── stage2_unfreeze.py         ➕
│   │   ├── stage2_lora.py             ➕
│   │   └── stage3_robust.py           ➕
│   ├── eval/
│   │   ├── metrics.py                 ➕
│   │   └── run_attack_grid.py         ➕
│   └── utils/
│       ├── seeding.py                 ➕
│       ├── threshold.py               ➕
│       └── logging.py                 ➕
├── scripts/
│   ├── inspect_hateful_memes.py       ✅  already used by previous work
│   ├── smoke_dataloader.py            ✅
│   └── make_train_split.py            ➕  one-shot, writes data/processed/splits/train_split.json
├── tests/
│   └── test_hateful_memes_data.py     ✅
├── cluster/                           ➕  project-side cluster automation
│   ├── config.env                     ➕  paths, PVC names, image refs (this project's values)
│   ├── state.local.env                ➕  active image tag
│   └── README.md                      ➕
├── Dockerfile                         ➕  custom image with open_clip etc. (see §8.5)
├── data/
│   ├── processed/splits/              ➕  train_split.json (deterministic 8000/500 partition)
│   ├── raw/                           ✅  empty on cluster nodes; dataset is on /scratch
│   └── external/                      ✅
├── docs/
│   ├── CLUSTER_DATA.md                ✅  to be updated to reflect /scratch/datasets/hate_meta
│   └── DATA.md                        ✅
├── experiments/                       ✅  generated outputs, ckpts, logs (gitignored)
├── notebooks/                         ➕  one EDA, one results notebook
├── reports/                           ➕  paper/poster drafts
├── project_planning/                  ✅
├── pyproject.toml                     ✅
├── requirements.txt                   ✅
└── README.md                          ✅
```

### 7.1 Cluster artifact layout

Where things live during and after a cluster run:

| Path | Where | Purpose | Lifetime |
|---|---|---|---|
| `/scratch/datasets/hate_meta` | inside Run:AI job | read-only dataset (course-staged) | persistent |
| `/scratch/robust-meme-hate-detection/experiments/<job>/` | inside Run:AI job | checkpoints, metrics.json, logs | persistent on group scratch |
| `/home/guasch/robust-meme-hate-detection/` | jumphost + inside job | synced project code | persistent |
| `cluster-results/<job>/` | local WSL | pulled artifacts (logs, metrics, plots) | local working copy |
| `cluster-checkpoints/<job>/` | local WSL | pulled checkpoints only | local working copy |

`scripts/cluster.sh` (project-side) will know these paths via `cluster/config.env`. The pull-artifacts step is the only thing that copies bytes back to the laptop.

---

## 8. Environment setup

### 8.1 Hardware (EPFL Run:AI cluster, verified 2026-05-02)

This project does not need a local GPU. Real training runs on the cluster.

| Property | Value |
|---|---|
| Allocation | 1× NVIDIA A100-SXM4-80GB per Run:AI job |
| Driver / CUDA | 580.82.07 / CUDA 13.0 (image: CUDA 12.8 toolkit) |
| VRAM | 80 GB (no memory pressure on either backbone) |
| Project | `course-ee-559-guasch` |
| Mount pattern | `--existing-pvc claimname=course-ee-559-scratch-g49,path=/scratch` and `claimname=home,path=/home/guasch` |
| Run-as | `--run-as-uid 316498` |
| Dataset path inside job | `/scratch/datasets/hate_meta` |
| Experiment outputs path | `/scratch/robust-meme-hate-detection/experiments/<job>/` |
| Job submission | `runai submit ...` from the jumphost (or via `cluster.sh`); **not** SLURM |

A laptop GPU is not required for the project. The only local GPU/CPU work is the synthetic-tensor smoke pass (§9 Track A).

### 8.2 Python environment

The cluster image targets **Python 3.12.3** with PyTorch 2.10.0 + CUDA 12.8. Locally we mirror this with `uv` only for the synthetic-tensor smoke pass and CI; for real runs we always use the cluster image.

```bash
# Local-only environment (Track A smoke pass; no GPU required)
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . -r requirements.txt
```

### 8.3 Dependencies (`requirements.txt`)

```text
torch                       # version controlled by cluster image; not pinned here
torchvision
open_clip_torch>=2.24
transformers>=4.41          # for tokenisers
peft>=0.11                  # only if the LoRA ablation in §5.3 is run
pillow>=10
numpy>=1.26
scikit-learn>=1.4           # metrics, threshold sweep
albumentations>=1.4         # image perturbations
nlpaug>=1.1                 # text perturbations
pyyaml>=6.0
tqdm>=4.66
wandb>=0.17                 # optional
matplotlib>=3.8
pandas>=2.2
huggingface_hub             # already in pyproject.toml
```

torch is intentionally unpinned because the cluster image controls it; for the local synthetic smoke pass `uv` will pull a CPU build, which is fine.

### 8.4 Hateful Memes dataset

**No local download needed.** The dataset is staged on the EPFL group scratch and visible from inside Run:AI jobs at `/scratch/datasets/hate_meta` (8 500 train labeled, 500 dev labeled, 1 000 test unlabeled, 10 000 PNGs; verified end-to-end on 2026-05-02). See §4.1.

Update needed in the existing repo before the first real run:
- `configs/data/hateful_memes.yaml` → `cluster_root: /scratch/datasets/hate_meta`
- `docs/CLUSTER_DATA.md` → drop the registration / rsync-upload sections; document the actual staged path.

### 8.5 Custom Docker image

The course base image `registry.rcp.epfl.ch/ee559/environment-with-packages:latest` ships PyTorch but **not** `open_clip_torch`. We build a project-specific image on top:

```dockerfile
# Dockerfile
FROM registry.rcp.epfl.ch/ee559/environment-with-packages:latest

USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-bake CLIP weights so compute pods without outbound internet still work.
RUN python - <<'PY'
import open_clip
open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
PY

USER guasch
```

Build and push (using the cluster automation skill the team already has):

```bash
./cluster/cluster.sh build-image v0.1
./cluster/cluster.sh push-image v0.1
```

This writes the active image into `cluster/state.local.env`, and subsequent job submissions use it.

---

## 9. End-to-end setup checklist

The work splits into two tracks. Track A is fast, runs on a laptop, and proves the code is correct. Track B runs on the cluster and produces the actual baseline checkpoint.

### Track A — local, code-only (no GPU, no dataset)

Goal: verify the model class and the loader contract on synthetic tensors. Anyone with a checkout can do this in minutes.

#### A.1 Repository bootstrap

```bash
git clone <repo>
cd robust-meme-hate-detection
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . -r requirements.txt
```

#### A.2 Make the deterministic train split

```bash
python scripts/make_train_split.py \
    --train-jsonl /scratch/datasets/hate_meta/train.jsonl \
    --out data/processed/splits/train_split.json \
    --held-out 500 --seed 0
```

(Run this with the *cluster* dataset path the first time, then commit the resulting JSON ID list. Track A then only needs the IDs, not the data itself.)

#### A.3 Synthetic-tensor smoke

```bash
python -m robust_meme_hate_detection.tests.smoke_synthetic
```

This script (a) instantiates `CLIPHateMemeClassifier` with random weights to avoid the OpenCLIP download, (b) makes a fake batch of `images01` and `token_ids`, (c) runs forward and backward, (d) runs a 2-step PGD on `images01` and asserts the gradient is non-zero. Expected exit code 0 in under a minute.

#### A.4 Loader unit tests

```bash
pytest tests/test_hateful_memes_data.py
```

Already in the repo and passing. Re-run after any change to the data path or the loader contract.

### Track B — Run:AI cluster, real training

Goal: a Stage-1 checkpoint with dev-AUROC ≥ 0.72 and a verified attack pipeline.

#### B.1 Unlock SSH and validate the workflow

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/epfl-ssh-key
./cluster/cluster.sh doctor
./cluster/cluster.sh remote-check
```

Both must be all-green before any submission.

#### B.2 Build and push the custom image

Once per change to `Dockerfile` / `requirements.txt`:

```bash
./cluster/cluster.sh build-image v0.1
./cluster/cluster.sh push-image v0.1
```

#### B.3 Sync project code to jumphost

```bash
./cluster/cluster.sh sync-code
```

#### B.4 Smoke job (real GPU, real dataset, no training)

A short Run:AI job that imports the model, builds the dataset against `/scratch/datasets/hate_meta`, runs forward + backward on one batch, and writes a tiny metrics file.

```bash
./cluster/cluster.sh submit-smoke smoke-stage1
./cluster/cluster.sh wait-job smoke-stage1
./cluster/cluster.sh pull-artifacts smoke-stage1
```

Expected in the logs: `nvidia-smi` shows A100, batch forward succeeds at the configured size, no OOM, no missing-image errors, AUROC near 0.5 (random head).

#### B.5 Stage-1 training

```bash
./cluster/cluster.sh run-train stage1-seed0
```

`run-train` submits, monitors, waits for completion, pulls artifacts and checkpoints, and prints a summary. Expected outputs in `cluster-results/stage1-seed0/`:

- `ckpt/best.pt`
- `metrics.json` (accuracy, macro F1, AUROC, precision, recall, FPR, FNR on `dev`)
- Validation AUROC in **0.72-0.76** on `dev`
- Wall-clock under 30 minutes on A100 80 GB

Repeat with `stage1-seed{1,2}` for the 3-seed reporting.

#### B.6 Threshold sweep

```bash
python -m robust_meme_hate_detection.utils.threshold \
    --ckpt cluster-results/stage1-seed0/ckpt/best.pt \
    --jsonl /scratch/datasets/hate_meta/dev.jsonl
```

Persists `cluster-results/stage1-seed0/threshold.json` (or runs as a follow-on Run:AI job).

#### B.7 Attack-readiness check

```bash
./cluster/cluster.sh submit-smoke pgd-check \
    --ckpt cluster-results/stage1-seed0/ckpt/best.pt \
    --epsilon 4/255 --steps 5 --max-batches 4
```

Expected: AUROC drops, no `element 0 of tensors does not require grad`. Confirms the model is end-to-end differentiable through the actual cluster runtime.

After B.7 the project is ready to enter Phase 3 of the roadmap (perturbation benchmark) and populate the experiment matrix in §6.

---

## 10. Configuration sketch (`configs/stage1.yaml`)

```yaml
seed: 0

model:
  arch: ViT-B-32
  pretrained: laion2b_s34b_b79k
  freeze_encoders: true
  head_hidden: 512
  head_dropout: 0.2

data:
  dataset_root: /scratch/datasets/hate_meta        # in-job path; loader resolves img/<id>.png from here
  train_split_file: data/processed/splits/train_split.json   # 8000 train / 500 train_held_out IDs
  val_split: dev                                   # uses dev.jsonl for early stopping
  num_workers: 8                                   # A100 host can sustain this comfortably

train:
  batch_size: 128                                  # was 64; revised for A100 80 GB
  epochs: 10
  lr_head: 1.0e-3                                  # scale linearly if batch_size changes
  weight_decay: 1.0e-2
  warmup_ratio: 0.1
  amp_dtype: bfloat16
  early_stop_patience: 3
  early_stop_metric: macro_f1
  pos_weight_auto: true                            # computes N_neg/N_pos on the train split

logging:
  out_dir: /scratch/robust-meme-hate-detection/experiments/stage1_seed0
  wandb_project: robust-meme-hate
  wandb_offline: true                              # default; flip to false if API key is set
```

Stage-2 (`stage2_unfreeze.yaml`, `stage2_lora.yaml`) and Stage-3 (`stage3_robust.yaml`) configs reuse the same shape with extra `unfreeze` / `lora` / `perturb` sections.

---

## 11. Known risks at the architecture level

| Risk | Symptom | Mitigation |
|---|---|---|
| Stage 1 plateaus below 0.72 AUROC | head underfits ViT-B/32 features | Stage 2 last-2-block unfreeze (default, §5.3); LoRA only as a regularisation ablation. |
| Image attacks have no effect | model lacks gradient through image path | Verify `Normalize` is *inside* the model, not the dataloader; smoke-test B.7. |
| Class imbalance hurts recall | low recall on hateful class | `pos_weight ≈ 1.79` in BCE (auto-computed); threshold tuning on F1. |
| Text-only or image-only baseline beats multimodal | model fails to use cross-modal signal | inspect interaction features; lower head LR or widen the hidden. |
| OpenCLIP cache miss inside Run:AI pod | first run hangs or fails on no outbound internet | weights are baked into the custom image at build time (§8.5). Verify with `ls /home/guasch/.cache/huggingface/hub/` inside a smoke job; if absent, rebuild the image. |
| LoRA does not attach cleanly to OpenCLIP | `peft` errors on module name lookup | LoRA is no longer the default (§5.3). If we run it, use a small custom `LoRALinear` wrapping the relevant attention `Linear` modules. |
| Train/dev contamination | train_held_out IDs leak into the training subset | `make_train_split.py` is run once with seed 0 and the resulting JSON is committed; all stages load IDs from that file, never resampled. |
| 8000-example train set overfits | dev metric peaks early then degrades | tighter early-stop patience (3), higher weight decay (5e-2), or stay frozen and skip Stage 2. |
| Image permission-quirk on `/scratch/datasets/hate_meta` | `PermissionError` on image read | listed perms are `drwx------` but Ceph ACLs grant the course group access. Verified working in the diagnostic job; if a job fails on this path, re-check `id` shows `rcp-caas-ee-559-g49_AppGrpU`. |

---

## 12. Definition of done for this document

The architecture is "ready for the next project steps" when, on a fresh clone:

1. **Track A** (A.1–A.4) succeeds without intervention on a laptop with no GPU.
2. **Track B.4** (cluster smoke job) prints `nvidia-smi` showing the A100, builds the dataset against `/scratch/datasets/hate_meta`, and completes a forward + backward pass.
3. **Track B.5** produces a checkpoint with `dev` AUROC ≥ 0.72 on at least one seed, in under 30 minutes wall-clock.
4. **Track B.7** confirms gradient flow to the input image (PGD reduces AUROC on a clean-correct subset, no `requires_grad` errors).

At that point the team can move to Phase 3 of the roadmap (implement the perturbation suite and run the attack grid against the Stage-1 checkpoint), then Phase 5 (Stage-3 robust training), without revisiting model-architecture decisions.

---

## Revision history

- **2026-05-02 — initial draft.** Laptop-class assumptions: 12-24 GB GPU, batch 64, ~1 hour per Stage-1 run, dataset to be downloaded locally, full Meta release with five split files.
- **2026-05-02 — cluster-aware revision.** A100 80 GB confirmed; dataset already staged at `/scratch/datasets/hate_meta` (3 splits, 10 000 images, test unlabeled). Batch 128 default, ViT-L/14 promoted to first ablation, last-2-block unfreeze replaces LoRA as Stage 2 default, setup checklist split into local Track A and cluster Track B, custom image build documented, train_held_out split policy added (§4.5), repo structure aligned with actual `src/robust_meme_hate_detection/` layout.
