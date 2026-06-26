# 02_train — model training

> **Advanced / reproduction-only.** You don't need this to *run* Shorkie — download the released
> weights with `data/download.sh --models all` and use `examples/` + `minimal_example/`. This stage
> documents how the released models were trained (and lets you retrain on your own data).

Three model variants, all built on the same `baskerville-yeast` + `westminster`
forks (pinned submodules under `external/`) and the same `unet_small_bert_drop`
architecture. They are deliberately minimal variations of one another:

| Variant | Dir | Driver | Init | Output |
|---|---|---|---|---|
| **Shorkie_LM** | `shorkie_lm/` | `hound_train.py` (single fold) | random | 1× `model_best.h5` → `gs://seqnn-share/shorkie_lm/` |
| **Shorkie_finetuned** | `shorkie_finetuned/` | `westminster_train_folds.py --restore <LM>` | **from Shorkie_LM** | 8× `model_best.h5` → `gs://seqnn-share/shorkie/f0..f7/` |
| **Shorkie_scratch** | `shorkie_scratch/` | `westminster_train_folds.py` (no `--restore`) | random | 8× `model_best.h5` (ablation; not publicly released) |

`Shorkie_LM` is a masked DNA language model pretrained on the 165_Saccharomycetales
corpus; its `model_best.h5` is the `--restore` target the supervised models start from.
`Shorkie_finetuned` and `Shorkie_scratch` train the **same** 8-fold supervised set
(5215 ChIP-exo/MNase/RNA-seq tracks, R64, 16 bp) — see `data/manifest.json` and
`scripts/01_data_build/`.

## Shorkie_finetuned vs Shorkie_scratch — the exact diff

These two are identical except for **one launch flag** and **two `params.json`
`train` fields**. Everything else — the full `model` block, the folds, the data,
`warmup_steps`/`adam_beta1`/`adam_beta2`, and every other
`westminster_train_folds.py` flag — is byte-for-byte the same.

| | Shorkie_finetuned | Shorkie_scratch |
|---|---|---|
| `make_model.sh` | **has** `--restore <Shorkie_LM .h5>` | **no** `--restore` |
| `params.json` → `task` | `fine-tune` | `supervised` |
| `params.json` → `learning_rate` | `2e-5` | `1e-4` |
| `params.json` → `model` block | — identical — | — identical — |

So the ablation cleanly isolates the effect of LM pretraining: drop the `--restore`
of the pretrained weights and raise the learning rate (random init tolerates / needs
a larger step), holding architecture, data, warmup, and optimizer betas fixed.

> **Provenance / fidelity.** The committed `shorkie_lm/params.json` and
> `shorkie_finetuned/params.json` are **byte-identical to the released configs** on
> `gs://seqnn-share/{shorkie_lm,shorkie}/params.json` (verified), so this repo's
> training commands match the published models exactly. `shorkie_scratch/params.json`
> is the work-dir random-init config (no released counterpart). A separate work-dir
> fine-tuning run (`self_supervised_unet_small_bert_drop`) used different
> `warmup_steps`/Adam betas and produced different weights — it is **not** the
> released model and is not what these scripts reproduce.

## The restore mechanism

`westminster_train_folds.py` (which fans out to `hound_train.py` per fold) decides
how each fold's weights are initialized:

- **`task: fine-tune` + `--restore <Shorkie_LM model_best.h5>`** → the pretrained LM
  trunk is loaded, then fine-tuned with the supervised head. This is `Shorkie_finetuned`.
- **`task: supervised` + no `--restore`** → weights start from random `lecun_normal`
  init (the `kernel_initializer` in the shared `model` block). This is `Shorkie_scratch`.

The restore checkpoint is resolved from config (`models.shorkie_lm_checkpoint`), which
points at the `Shorkie_LM` `train/model_best.h5`.

## Paths & portability

All three scripts source `scripts/common/env.sh` and resolve every path through
`config/paths.yaml` (a `cfg()` helper reads dotted keys). No absolute paths are baked
in. The data/checkpoint locations come from:

- `datasets.lm_train_dir` — 165_Saccharomycetales corpus (LM training data)
- `datasets.supervised_data` — 8-fold supervised TFRecords (the `../data` arg)
- `datasets.lm_corpus_split_root` — `--eval_dir`
- `models.shorkie_lm_checkpoint` — the `--restore` target

Each script accepts `--dry-run`, which prints the fully-resolved `hound_train.py` /
`westminster_train_folds.py` command without launching it — use this to inspect what
will run.

The `#SBATCH` headers target this paper's cluster (a100 / `ssalzbe1_gpu`). To run on a
different scheduler — or with none at all — submit through the portable wrapper, which
reads `config/slurm.yaml` and falls back to a plain local run:

```bash
# portable submit (reads config/slurm.yaml profiles)
scripts/common/submit.sh --profile gpu scripts/02_train/shorkie_finetuned/make_model.sh

# scheduler-free local run (e.g. inside the container)
SHORKIE_LOCAL=1 scripts/common/submit.sh --profile gpu scripts/02_train/shorkie_scratch/make_model.sh
```

GPU is required for training. The released weights can be downloaded instead of
retrained — see `data/download.sh`.
