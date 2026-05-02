# Data Layout

The current local Hateful Memes dataset is stored at:

```text
data/raw/data
```

It contains:

```text
data/raw/data/
  train.jsonl
  dev.jsonl
  test.jsonl
  img/
  README.md
  LICENSE.txt
```

The setup scripts and config now use this directory as the default data root.

## Verified Counts

The loader was validated against the current local files:

| Split | Examples | Labels | Missing Images |
|---|---:|---:|---:|
| train | 8500 | 8500 | 0 |
| dev | 500 | 500 | 0 |
| test | 1000 | 0 | 0 |

The test split is unlabeled, so it should be treated as an inference-only split
unless labels are provided separately.

## Should We Reorganize It?

For now, keep `data/raw/data` because it is the exact extracted dataset folder
and the code points to it through config. Before syncing to the cluster, one
cleanup pass is recommended:

- Keep `data/raw/data` as the canonical raw dataset root.
- Keep `data/raw/data.zip` locally only if you want a backup of the original zip.
- Remove `data/raw/hateful_memes` if it is still the partial interrupted mirror
  download and not needed.
- Do not commit any raw data, zip files, checkpoints, or generated attacked
  examples.

Longer term, `data/raw/data` could be renamed to `data/raw/hateful_memes` for
clarity. That is mostly cosmetic and should only be done before cluster jobs
start depending on a path. If we rename it, update:

- `configs/data/hateful_memes.yaml`
- `README.md`
- `docs/CLUSTER_DATA.md`
- cluster sync commands

## Processing Policy

Do not preprocess the raw images into a new permanent image folder yet. For
Phase 1 and the first baselines, load raw PNGs and apply transforms at runtime.

Create processed artifacts only when they save repeated work or preserve a
reproducible split/attack state, for example:

- `data/processed/splits/` for deterministic train/dev metadata exports;
- `data/processed/features/` for cached CLIP embeddings after the baseline
  feature pipeline is finalized;
- `data/processed/attacks/` for generated attacked examples or attack metadata.

Runtime transforms should remain in code/config for now, because we will need
different clean, perturbed, and CLIP-specific transforms in later phases.
