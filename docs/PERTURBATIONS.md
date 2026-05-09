"""Perturbation System for Adversarial Robustness

This document explains how to use the text and image perturbation system for
evaluating adversarial robustness of the Hateful Memes classifier.

## Quick Start

```python
from robust_meme_hate_detection.perturbations import TextPerturbation, ImagePerturbation, ComposePerturbation
from robust_meme_hate_detection.data.hateful_memes import build_hateful_memes_dataloader

# Create perturbation pipelines
text_pipeline = ComposePerturbation([
    TextPerturbation(mode='leetspeak', probability=0.3, severity=0.5),
    TextPerturbation(mode='char_deletion', probability=0.2, severity=0.1),
])

image_pipeline = ComposePerturbation([
    ImagePerturbation(mode='gaussian_noise', probability=0.4, severity=0.05),
    ImagePerturbation(mode='blur', probability=0.3, severity=0.3),
])

# Load data with perturbations
loader = build_hateful_memes_dataloader(
    dataset_root='data/raw/data',
    split='dev',
    batch_size=4,
    apply_perturbations=True,
    text_perturbation=text_pipeline,
    image_perturbation=image_pipeline,
)
```

## Available Text Perturbations

### TextPerturbation(mode, probability, severity)

- **mode**: Type of text perturbation
- **probability**: Chance (0-1) of applying this perturbation
- **severity**: Strength (0-1) of the perturbation

#### Modes:

1. `'leetspeak'`: Replace letters with numbers/symbols (a→@, e→3, etc.)
   - severity: fraction of applicable characters to substitute

2. `'char_deletion'`: Remove random characters
   - severity: fraction of characters to delete

3. `'char_swap'`: Swap adjacent characters
   - severity: fraction of adjacent pairs to swap

4. `'spacing'`: Insert spaces between characters
   - severity: fraction of positions to insert spaces

5. `'punctuation'`: Insert random punctuation (!?.,'.)
   - severity: fraction of positions to insert punctuation

6. `'case_noise'`: Randomly flip letter case
   - severity: fraction of letters to flip

7. `'censoring'`: Replace characters with asterisks (*)
   - severity: fraction of characters to censor

## Available Image Perturbations

### ImagePerturbation(mode, probability, severity)

- **mode**: Type of image perturbation
- **probability**: Chance (0-1) of applying this perturbation
- **severity**: Strength (interpretation depends on mode)

#### Modes:

1. `'gaussian_noise'`: Add Gaussian noise
   - severity: std dev as fraction of [0, 255] range

2. `'blur'`: Apply Gaussian blur
   - severity: blur radius as fraction (0-1 maps to 0-10 pixels)

3. `'compression'`: Apply JPEG compression
   - severity: quality level (0.1 → quality 90, 1.0 → quality 10)

4. `'brightness'`: Adjust brightness
   - severity: scale factor (0.1 → 0.7x, 1.0 → 1.3x brightness)

5. `'contrast'`: Adjust contrast
   - severity: scale factor (0.1 → 0.7x, 1.0 → 1.3x contrast)

6. `'translation'`: Shift pixels with gray padding
   - severity: max shift as fraction of image size

7. `'crop'`: Crop and zoom
   - severity: crop size as fraction of image

8. `'occlusion'`: Add random gray masks
   - severity: mask size as fraction of image (creates 1-3 masks)

## Stacking Perturbations

Use `ComposePerturbation` to apply multiple transformations to the same sample:

```python
# Multiple text perturbations, each with independent probability
text_pipeline = ComposePerturbation([
    TextPerturbation(mode='leetspeak', probability=0.3, severity=0.5),
    TextPerturbation(mode='char_deletion', probability=0.2, severity=0.1),
    TextPerturbation(mode='spacing', probability=0.2, severity=0.3),
    TextPerturbation(mode='censoring', probability=0.1, severity=0.05),
])

# Multiple image perturbations
image_pipeline = ComposePerturbation([
    ImagePerturbation(mode='gaussian_noise', probability=0.4, severity=0.05),
    ImagePerturbation(mode='blur', probability=0.3, severity=0.3),
    ImagePerturbation(mode='compression', probability=0.2, severity=0.2),
    ImagePerturbation(mode='brightness', probability=0.2, severity=0.2),
])
```

Each perturbation is applied sequentially and independently. A sample can receive
0, 1, 2, or more perturbations depending on each perturbation's probability.

**Example distribution for 4 perturbations at 0.1 probability each:**
- P(0 perturbations) ≈ 65.6%
- P(1 perturbation) ≈ 29.2%
- P(2 perturbations) ≈ 4.9%
- P(3+ perturbations) < 1%

## Using with HatefulMemesDataset

The perturbation system integrates with the dataset loader. Set `apply_perturbations=True`
and pass your perturbation pipelines:

```python
# No perturbations (clean data)
clean_loader = build_hateful_memes_dataloader(
    dataset_root='data/raw/data',
    split='dev',
    apply_perturbations=False,
)

# With text and image perturbations (multimodal attacks)
perturbed_loader = build_hateful_memes_dataloader(
    dataset_root='data/raw/data',
    split='dev',
    apply_perturbations=True,
    text_perturbation=text_pipeline,
    image_perturbation=image_pipeline,
)

# Text-only attacks (no image perturbation)
text_only_loader = build_hateful_memes_dataloader(
    dataset_root='data/raw/data',
    split='dev',
    apply_perturbations=True,
    text_perturbation=text_pipeline,
    image_perturbation=None,
)

# Image-only attacks (no text perturbation)
image_only_loader = build_hateful_memes_dataloader(
    dataset_root='data/raw/data',
    split='dev',
    apply_perturbations=True,
    text_perturbation=None,
    image_perturbation=image_pipeline,
)
```

## Implementation Details

- Perturbations are applied **before** standard transforms
- This allows you to perturb raw data and then apply model-specific preprocessing
- All perturbations are stateless (no side effects across samples)
"""

