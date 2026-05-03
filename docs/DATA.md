# Data Layout

The dataset lives on the EPFL group scratch and is **not** stored locally on
laptops. See `CLUSTER_DATA.md` for the cluster paths and access details.

## Logical splits used by the project

| Logical split | Source | Examples | Labeled? | Purpose |
|---|---|---:|:-:|---|
| `train` | first 8,000 IDs of `train.jsonl` | 8,000 | yes | gradient steps |
| `train_held_out` | last 500 IDs of `train.jsonl` (stratified, seed 0) | 500 | yes | final reported numbers (clean & attacked) |
| `dev` | all rows of `dev.jsonl` | 500 | yes | early-stopping, threshold tuning, dev attack sweeps |
| `test` | all rows of `test.jsonl` | 1,000 | no | inference-only |

Class balance: ~36% positive. `pos_weight ≈ 1.79` is auto-computed from the
8,000-example training subset.

The 8,000/500 partition is materialized by:

```bash
python scripts/make_train_split.py \
    --train-jsonl /scratch/datasets/hate_meta/train.jsonl \
    --out data/processed/splits/train_split.json \
    --held-out 500 --seed 0
```

The resulting JSON is committed to git so all stages and seeds share one
partition.

## Loader

`src/robust_meme_hate_detection/data/hateful_memes.py` accepts either of:

- a logical split name (`train`, `dev`, `test`, `dev_seen`, etc.) when reading
  raw JSONL files, or
- a `train_split_file` containing the precomputed ID partition.

Image preprocessing is **deliberately split** between the dataloader and the
model:

- **Dataloader:** PIL resize → CenterCrop → ToTensor → `(3,224,224)` in `[0,1]`.
- **Model:** Normalize(mean, std) inside `nn.Module`, so attacks see raw pixels.

This contract is what keeps FGSM/PGD operating on `model.forward(images01,
token_ids)` directly. See `model_architecture.md` §4.2.

## Don't

- Don't commit raw images, JSONL, checkpoints, or generated attacked examples.
- Don't preprocess the dataset into a static cached folder yet — Phase 1/2
  loads raw PNGs at runtime so we can introduce different clean / perturbed /
  attacked transforms without re-extracting features.
- Don't tune anything on `train_held_out`. It is touched exactly once per
  checkpoint, for the final reported numbers.
