# Cluster Data Workflow

The Hateful Memes dataset is **already staged on the EPFL group scratch volume**
by the course staff. Training jobs read it directly; no laptop download or
rsync is required.

Verified on 2026-05-02 by an end-to-end Run:AI diagnostic job
(`hatecheck-20260502-185828`).

## Paths

| View | Path | Notes |
|---|---|---|
| Inside a Run:AI job | `/scratch/datasets/hate_meta` | mount: `--existing-pvc claimname=course-ee-559-scratch-g49,path=/scratch` |
| Jumphost view | `/mnt/course-ee-559/rcp-caas-ee-559-g49/scratch-g49/datasets/hate_meta` | for `ls` / inspection |

The directory permissions show `drwx------` for the staff uid, but Ceph ACLs
grant the course group (`rcp-caas-ee-559-g49_AppGrpU`) read access. Reads
succeed under user `guasch` (uid 316498) both on the jumphost and inside
Run:AI jobs.

## Layout

```
/scratch/datasets/hate_meta/
├── img/              # 10,000 .png files
├── train.jsonl       # 8,500 rows, labeled (3,050 hateful / 5,450 non-hateful)
├── dev.jsonl         #   500 rows, labeled (250 / 250, balanced)
├── test.jsonl        # 1,000 rows, UNLABELED in this mirror
├── README.md
└── LICENSE.txt
```

JSONL schema:

```json
{"id": 42953, "img": "img/42953.png", "label": 0, "text": "..."}
```

## Implications for evaluation

`test.jsonl` has no `label` field, so it cannot be used for supervised
evaluation. The project carves a deterministic held-out slice from
`train.jsonl` for final reporting; see `model_architecture.md` §4.5.

| Logical split | Source | Size | Use |
|---|---|---:|---|
| `train` | first 8,000 rows of `train.jsonl` | 8,000 | gradient steps |
| `train_held_out` | last 500 rows of `train.jsonl` (stratified) | 500 | final reported numbers |
| `dev` | `dev.jsonl` | 500 | early-stopping, threshold tuning |
| `test` | `test.jsonl` | 1,000 | inference-only, untouched |

Run `scripts/make_train_split.py` once with seed 0 to materialize the partition
into `data/processed/splits/train_split.json`. The JSON ID list is committed
to git so every seed and stage uses the same partition.

## Inside-job usage

Run:AI jobs should pass the in-job path:

```bash
--data-root /scratch/datasets/hate_meta
```

The default mounts in `cluster/cluster.sh` are:

```text
--existing-pvc claimname=course-ee-559-scratch-g49,path=/scratch
--existing-pvc claimname=home,path=/home/guasch
```

`/shared-ro` and `/shared-rw` are not currently mounted because no course
material lives there for this project; add them to the mount list only if a
specific need arises.

## Don't

- Do **not** redistribute `/scratch/datasets/hate_meta` outside the cluster
  (license terms; raw hateful content).
- Do **not** commit raw images, JSONL, or generated attacked examples to git.
- Do **not** point training scripts at a local mirror of the dataset; use the
  in-job `/scratch/datasets/hate_meta` path so all runs are reproducible.
