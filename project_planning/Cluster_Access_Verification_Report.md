# Cluster Access & Hard-Blocker Verification Report

**Date:** 2026-05-02
**Verified by:** end-to-end Run:AI smoke job `hatecheck-20260502-185828` (status: **Succeeded**, then deleted).

---

## TL;DR

> **Both hard blockers are resolved.** A submitted Run:AI job ran on an NVIDIA **A100 80 GB**, mounted the group scratch volume, listed the Hateful Memes dataset, parsed the JSONL files, and reported `BLOCKERS_RESOLVED`. The path the project configs currently point to is wrong; the dataset actually lives at `/scratch/datasets/hate_meta` inside jobs. Two minor housekeeping items remain (add `open_clip` to the image, update the project config path).

| Blocker | Before | After | Evidence |
|---|---|---|---|
| **GPU access** | Open | ✅ **Resolved** | `nvidia-smi` inside Run:AI job showed 1× A100-SXM4-80GB; `torch.cuda.is_available() == True`. |
| **Hateful Memes dataset** | Open (local copy missing) | ✅ **Resolved** | `/scratch/datasets/hate_meta/` reachable from inside a job. 8500 train / 500 dev / 1000 test rows + 10 000 PNGs. JSONL schema matches the loader. |
| `open_clip_torch` available in image | Not part of original blockers | ⚠️ **Soft blocker** | Base image lacks `open_clip` (`ModuleNotFoundError`). Fix: add to Dockerfile, rebuild, push. |
| Project config dataset path | Pointed at non-existent path | ⚠️ **Needs update** | Configs say `/scratch/robust-meme-hate-detection/data/raw/data`; actual path is `/scratch/datasets/hate_meta`. |

---

## What was tested

### 1. Local prerequisites and SSH
- VPN/DNS to `jumphost.rcp.epfl.ch`: reachable.
- SSH key at `~/.ssh/epfl-ssh-key` with passphrase; loaded into `ssh-agent` (PID 12806) by user.
- Direct SSH to `epfl-jumphost`: works, lands on `haas001` as user `guasch`.

### 2. `./scripts/cluster.sh doctor`
All checks green:

```
ssh                ok
rsync              ok
docker             ok
git                ok
codex              ok
docker daemon      ok
local dataset      ok (cluster-access-tutorial/fruit_dataset)   # tutorial dataset, unrelated
config             ok
sync excludes      ok
ssh batch mode     ok
remote runai       ok (course-ee-559-guasch)
remote scratch     ok (/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch-g49)
active image       registry.rcp.epfl.ch/ee-559-guasch/cluster-access-tutorial:v0.5
```

### 3. Dataset discovery on cluster scratch

The project doc `docs/CLUSTER_DATA.md` says the dataset should be at
`/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch/robust-meme-hate-detection/data/raw/data`.
That path does **not** exist. The correct path is:

```
Jumphost view: /mnt/course-ee-559/rcp-caas-ee-559-g49/scratch-g49/datasets/hate_meta/
Inside job:    /scratch/datasets/hate_meta/
```

Contents (verified by SSH and re-verified inside the Run:AI job):

| File / dir | Size / count | Notes |
|---|---|---|
| `train.jsonl` | 8 500 rows | 3 050 hateful + 5 450 non-hateful (35.9% positive) |
| `dev.jsonl` | 500 rows | 250 / 250 (balanced) |
| `test.jsonl` | 1 000 rows | **unlabeled** (no `label` key) |
| `img/` | 10 000 PNGs | sample: 550×366 8-bit RGB |
| `README.md`, `LICENSE.txt` | present | original Meta licence text |

Schema matches the loader at `src/robust_meme_hate_detection/data/hateful_memes.py`:
```json
{"id": 42953, "img": "img/42953.png", "label": 0, "text": "..."}
```

The data permissions on the directory are nominally `drwx------` for uid 20084650, but Ceph ACLs grant the course group access — the listing, JSON reads, and image reads all succeeded as user `guasch`.

### 4. Run:AI diagnostic job

Submitted with the canonical mount pattern:
```
runai submit hatecheck-20260502-185828 \
  --image registry.rcp.epfl.ch/ee559/environment-with-packages:latest \
  --gpu 1 --run-as-uid 316498 \
  --existing-pvc claimname=course-ee-559-scratch-g49,path=/scratch \
  --existing-pvc claimname=home,path=/home/guasch \
  --command -- bash /home/guasch/hate_meme_blocker_check.sh
```

Lifecycle: `Pending → Running → Succeeded` in ~30 seconds. Selected log lines:

```
===== nvidia-smi =====
NVIDIA-SMI 580.82.07   Driver 580.82.07   CUDA 13.0
0  NVIDIA A100-SXM4-80GB    81 920 MiB total / 6 MiB used

===== Python / Torch =====
Python 3.12.3
torch 2.10.0+cu128  cuda True  ndev 1  name NVIDIA A100-SXM4-80GB

===== End-to-end mini test =====
DATASET_OK train=8500 dev=500 test=1000 images=10000
GPU_OK cuda=True ndev=1
BLOCKERS_RESOLVED
```

The job was deleted after verification.

---

## Findings

### Hard blockers — both resolved

1. **GPU.** A single A100 80 GB is allocatable to a Run:AI job under project `course-ee-559-guasch`. CUDA-enabled PyTorch is preinstalled in the base image. This is more headroom than the architecture document assumed (it targeted 16-24 GB); we can comfortably use ViT-B/32 at large batch and even consider ViT-L/14 if useful.
2. **Dataset.** The Hateful Memes dataset is staged on the group scratch and visible from inside Run:AI jobs at `/scratch/datasets/hate_meta`. The schema, split sizes, and image count match the loader's contract. Because the dataset lives on the shared scratch (read-only ACLs for our user), we do **not** need to upload anything from the laptop, contrary to what `docs/CLUSTER_DATA.md` currently describes.

### Items that surfaced during verification (next-step backlog)

3. **`open_clip_torch` is not in the base image.** Submitting a CLIP-based job today would `ModuleNotFoundError`. Two options:
   - **Recommended:** add `open_clip_torch>=2.24` and the rest of `requirements.txt` to a custom Dockerfile, run `cluster.sh build-image && cluster.sh push-image`. The cluster workflow is set up for exactly this.
   - **Quick alternative:** prepend `pip install --user open_clip_torch` to the job command. Fine for one-off tests, not for long training runs.
4. **Project config path is stale.**
   - `configs/data/hateful_memes.yaml`: `cluster_root: /scratch/robust-meme-hate-detection/data/raw/data` → should be `/scratch/datasets/hate_meta`.
   - `docs/CLUSTER_DATA.md`: drop the rsync/upload section and document the actual path. Several "Recommended acquisition strategy" steps no longer apply.
5. **Path naming inconsistency in cluster docs.** `cluster/config.env` uses `/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch-g49`. Some project docs use `/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch` (no `-g49`). The `-g49` form is correct on the jumphost; from inside a job both views collapse to `/scratch`.
6. **`/shared-ro` and `/shared-rw` mounts.** The cluster skill mentions these PVCs, but they are not currently needed — the dataset is on the group scratch under `/scratch`. I did not mount them in the diagnostic job; mount them only when course staff actually publish read-only material we want to consume.

### Things that are non-issues
- The `home_PVC=home` claimname looked unusual but it works: the diagnostic script (written to `/home/guasch/` on the jumphost) was visible at `/home/guasch/` inside the job and executed cleanly.
- Job UID/GID handling worked without `--run-as-gid`. The default `getpwuid()` issue mentioned in the skill failure-handling notes did not appear.

---

## What this changes for the project plan

### `model_architecture.md` updates
- §8.1 Hardware: replace "≥12 GB VRAM minimum / ≥16 GB comfortable" with "EPFL Run:AI: A100 80 GB by default; ViT-B/32 main run, ViT-L/14 viable as ablation."
- §8.4 Hateful Memes dataset: replace the local-download instructions with: "Already staged at `/scratch/datasets/hate_meta` inside Run:AI jobs. No download needed."
- §9.1-9.5 Setup checklist: adjust for cluster-first workflow. The `python -c "import torch; print(torch.cuda.is_available())"` smoke test should be run inside a Run:AI job, not on the WSL laptop.

### `Baseline_Model_Final.md` updates
- §6 (training plan): "Total compute budget for a complete training pass (clean + robust, 3 seeds each) is on the order of a few GPU-days at most" — confirmed feasible. Stage 1 on A100 80 GB at batch 128 should finish well under an hour.

### Immediate next steps (in order)
1. **Update the dataset path** in `configs/data/hateful_memes.yaml` to `cluster_root: /scratch/datasets/hate_meta`. Trivial PR.
2. **Build the custom image** with the Hateful-Memes project's `requirements.txt` baked in (`open_clip_torch`, `transformers`, `albumentations`, `nlpaug`). Push to `registry.rcp.epfl.ch/ee-559-guasch/...`.
3. **Adapt `cluster.sh`** for this project (or copy the script and re-point its config). The skill explicitly says reuse is just config + dataset path + entrypoint.
4. **Run a Stage-1 smoke job** that loads ~100 examples through the existing `HatefulMemesDataset`, builds a `CLIPHateMemeClassifier`, does one forward and one backward pass on the A100. Closes the "ready for training" gate from `model_architecture.md` §12.

### Authoritative answers I can now give

- **"Will I have to train the model from 0?"** No. CLIP encoders download from OpenCLIP at first use; only the fusion head trains.
- **"Do we have the dataset?"** Yes, on cluster scratch at `/scratch/datasets/hate_meta`. No upload needed.
- **"Do we have GPUs?"** Yes, A100 80 GB on demand under project `course-ee-559-guasch`.
- **"Are the hard blockers from my earlier assessment closed?"** Yes — both. The remaining work is software-side (custom image, config fix, project-side cluster wrapper) and is straightforward.
