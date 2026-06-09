# Data

The **Hateful Memes** dataset is not redistributed here. To train or evaluate,
place the dataset under a directory and point `data.dataset_root` (in the YAML
configs) or `--dataset-root` (for `evaluate.py`) at it. The directory must
contain the split JSONL files and the `img/` folder:

```
<dataset_root>/
  train.jsonl          # {"id", "img", "text", "label"} per line
  dev.jsonl
  test_seen.jsonl      # (optional) labelled held-out splits used in the report
  test_unseen.jsonl
  img/00001.png ...
```

The dataset is available from the Hateful Memes Challenge
(Kiela et al., 2020); public mirrors such as `neuralcatcher/hateful_memes`
provide the labelled `test_seen` / `test_unseen` splits used for held-out
evaluation.

## `processed/splits/train_split.json`

The committed deterministic 8000 / 500 train / held-out partition of
`train.jsonl` (stratified, seed 0). Training reads this file so every run uses
the same split. It lists example ids only — no dataset content.

The **interactive demo does not need the dataset**: it ships with 10 sample
memes under `demo/samples/`.
