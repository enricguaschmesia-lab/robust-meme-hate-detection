# Cluster Data Workflow

The EPFL Run:AI jobs should load large datasets from the group scratch volume,
not from a local laptop path.

For group 49, the course tutorial and prior setup notes indicate:

- remote group scratch path on the jumphost:
  `/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch`
- scratch PVC name:
  `course-ee-559-scratch-g49`
- path inside a Run:AI job after mounting the PVC:
  `/scratch`

Recommended project dataset location:

```text
/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch/robust-meme-hate-detection/data/raw/data
```

Inside a job, the same directory is visible as:

```text
/scratch/robust-meme-hate-detection/data/raw/data
```

Use this path in training commands:

```bash
python3 scripts/smoke_dataloader.py \
  --data-root /scratch/robust-meme-hate-detection/data/raw/data \
  --split train
```

## Recommended acquisition strategy

The user-suggested `emily49/hateful-memes` Hugging Face repository exposes the
JSONL metadata files, but the Hugging Face API reports only:

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `README.md`
- `LICENSE.txt`

It does not expose the meme image files, so it is not sufficient for a
multimodal project by itself.

Use the fuller Hugging Face mirror instead:

```text
cs5242-hateful-memes/hateful-memes-data
```

This mirror includes JSONL split files and the `img/` directory. Downloading it
means accepting the original Meta Hateful Memes research license terms.

## Local download

From the repo root:

```bash
pip install -r requirements.txt
python scripts/download_hateful_memes.py \
  --source cs5242 \
  --output-dir data/raw/data
python scripts/inspect_hateful_memes.py \
  --data-root data/raw/data \
  --check-images
```

## Upload to group scratch

From a Linux/WSL shell, prefer `rsync`:

```bash
rsync -av --info=progress2 \
  data/raw/data/ \
  epfl-jumphost:/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch/robust-meme-hate-detection/data/raw/data/
```

From native Windows PowerShell, `scp -r` also works but is less efficient for
resuming interrupted transfers:

```powershell
scp -r data\raw\data guasch@jumphost.rcp.epfl.ch:/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch/robust-meme-hate-detection/data/raw/
```

## Download directly on the cluster

If internet access from the jumphost is reliable, downloading directly to
scratch avoids copying several GB from the laptop:

```bash
ssh epfl-jumphost
cd /mnt/course-ee-559/rcp-caas-ee-559-g49/scratch
git clone <PRIVATE_REPO_URL> robust-meme-hate-detection
cd robust-meme-hate-detection
python3 -m pip install --user huggingface_hub
python3 scripts/download_hateful_memes.py \
  --source cs5242 \
  --output-dir /mnt/course-ee-559/rcp-caas-ee-559-g49/scratch/robust-meme-hate-detection/data/raw/data
```

If this fails due to missing packages or blocked networking on the jumphost,
download locally and upload with `rsync`.

## Run:AI mount reminder

Mount scratch when submitting jobs:

```bash
--existing-pvc claimname=course-ee-559-scratch-g49,path=/scratch
```

Then pass the job-visible path:

```bash
--data-root /scratch/robust-meme-hate-detection/data/raw/data
```

Keep raw data, checkpoints, and generated attacked examples out of Git.
