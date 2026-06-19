# 00_setup — one-time setup

Stage 00 bootstraps a fresh checkout: pin the external code, build the conda
environment, write the config files, and pull the released artifacts. Run it
once before any other stage; the later stages assume the `yeast_ml` env is
active and `config/paths.yaml` exists.

| Step | What |
|------|------|
| `init_submodules.sh` | Init the pinned forks `external/baskerville-yeast` and `external/westminster`, print their pins, then echo the follow-up install/config commands. |

## Order of operations

```bash
# 1. pin + fetch the external model code
bash scripts/00_setup/init_submodules.sh

# 2. create the environment and install editable packages
conda env create -f environment.yml && conda activate yeast_ml
pip install -e external/baskerville-yeast -e external/westminster -e .

# 3. copy and edit the config templates (paths are read from here, not hardcoded)
cp config/paths.example.yaml config/paths.yaml
cp config/slurm.example.yaml config/slurm.yaml

# 4. fetch released artifacts (manifest-driven; see data/README.md)
data/download.sh --minimal                       # 8 fine-tuned folds -> ./my_shorkie
data/download.sh --models all                    # LM + fine-tuned weights
data/download.sh --lm-corpus <tier|all> -u PROJECT   # requester-pays datasets
data/download.sh --supervised [bigwigs|tfrecords|all] -u PROJECT
```

## Config & data

- **Paths/SLURM config:** every downstream script reads paths from
  `config/paths.yaml` via the installed `shorkie` package (keys such as
  `work_root`, `data_root`, `release_root`, `corpus_build_data_root`,
  `motif_db_dir`, and external-tool paths). `source scripts/common/env.sh`
  exports these as shell variables (`${DATA_ROOT}`, `${BASKERVILLE_SCRIPTS}`, …).
- **Data download** is not duplicated here — it is `data/download.sh`, driven by
  `data/manifest.json` (model weights from the public `gs://seqnn-share`;
  datasets from the requester-pays `gs://shorkie-paper`, which needs `gsutil`
  and a billing project via `-u`). See [`../../data/README.md`](../../data/README.md).

## Tools

- `git` (submodules), `conda`/`pip`, and `gsutil` (or `wget`/`curl` fallback for
  the public model bucket) for `data/download.sh`.
