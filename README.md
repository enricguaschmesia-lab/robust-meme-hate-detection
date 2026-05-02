# Robust Meme Hate Detection

Course project repository for studying adversarial robustness in multimodal
hateful meme detection.

The Phase 1 goal is to acquire the Hateful Memes dataset, set up reproducible
data loading, and make the local and EPFL cluster paths explicit.

## Repository layout

```text
configs/      Experiment and data configuration files
data/         Local data root; raw data is git-ignored
docs/         Project and cluster workflow notes
experiments/  Local experiment outputs; git-ignored
notebooks/    Exploratory notebooks
reports/      Report/poster artifacts
scripts/      Command-line utilities
src/          Python package source
tests/        Tests and smoke checks
```

## Phase 1 dataset setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the recommended Hateful Memes mirror:

```bash
python scripts/download_hateful_memes.py \
  --source cs5242 \
  --output-dir data/raw/data
```

Inspect local files and split statistics:

```bash
python scripts/inspect_hateful_memes.py \
  --data-root data/raw/data \
  --check-images
```

Build a dataloader and print one batch:

```bash
python scripts/smoke_dataloader.py \
  --data-root data/raw/data \
  --split train \
  --batch-size 4
```

Run the data tests:

```bash
$env:PYTHONPATH="src"; python -m unittest discover -s tests -v
```

On Linux/macOS/cluster shells:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The local Windows Python environment used during setup did not have PyTorch
installed, so the metadata/PIL dataset tests pass locally and the actual
`torch.utils.data.DataLoader` batch test is skipped. It will run once PyTorch is
available, for example in the EPFL course image or a project virtual
environment.

For EPFL cluster data placement, see [docs/CLUSTER_DATA.md](docs/CLUSTER_DATA.md).
