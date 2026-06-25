# 00_setup — one-time setup

Stage 00 bootstraps a fresh checkout: pin the external code, build the conda
environment, write the config files, and pull the released artifacts. Run it
once before any other stage; the later stages assume the `yeast_ml` env is
active and `config/paths.yaml` exists.

| Step | What |
|------|------|
| `init_submodules.sh` | Init the pinned forks `external/baskerville-yeast` and `external/westminster`, print their pins, then echo the follow-up install/config commands. |
| `verify_install.sh` | After `pip install -e .`: `import shorkie`, resolve the released model keys, run `pytest tests/`. |
| `upload_release.sh` | **Maintainer-only.** Publish the catalogued-but-not-yet-uploaded artifacts (`shorkie_random_init` model → `gs://seqnn-share`; `eqtl`/`mpra` scores → `gs://shorkie-paper`). Idempotent (`gsutil cp -n`). Needs write access to both buckets. `--dry-run` prints the commands. |
| `verify_release.py` | Audit `manifest.json` vs the buckets (`gsutil stat`/`ls`): size+md5 for models, non-empty prefixes for datasets. Run after `upload_release.sh` to confirm completeness. |

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

# 4. (optional) sanity-check the install
bash scripts/00_setup/verify_install.sh

# 5. fetch released artifacts (manifest-driven; see data/README.md)
data/download.sh --minimal                       # 8 fine-tuned folds -> ./my_shorkie
data/download.sh --models all                    # LM + fine-tuned weights (live now)
data/download.sh --lm-corpus <tier|all> -u PROJECT   # requester-pays datasets
data/download.sh --supervised [bigwigs|tfrecords|all] -u PROJECT
data/download.sh --eqtl -u PROJECT               # Figure-7 eQTL scores (after release upload)
data/download.sh --mpra all -u PROJECT           # Figure-6 MPRA data (after release upload)
```

### Maintainer: publishing the release

`shorkie_random_init` + the `eqtl`/`mpra` scores are catalogued in `manifest.json` with
`pending_upload: true` but not yet on the buckets. Publish them (needs write access to
`gs://seqnn-share` + `gs://shorkie-paper`), then verify:

```bash
bash scripts/00_setup/upload_release.sh --all -u PROJECT      # gsutil cp -n (idempotent); --dry-run to preview
python scripts/00_setup/verify_release.py -u PROJECT          # all green when complete
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
