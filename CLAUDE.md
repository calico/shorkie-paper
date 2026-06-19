# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the **analysis / reproducibility release for the Shorkie paper**: the scripts, an installable helper package, and figure-reproduction notebooks that rebuild the datasets, train the three model variants, and regenerate the paper's benchmarks and figures. Remote: `github.com/calico/shorkie-paper`.

Unlike a typical analysis dump, this repo **is set up to be reproducible by an outside user**:
- **Installable package** — `pyproject.toml` makes `src/shorkie/` importable (`pip install -e .`); there is a real `environment.yml` (conda env `yeast_ml`). (Earlier revisions had neither — do not trust older notes that say "no pyproject/requirements".)
- **No hardcoded paths** — every filesystem path resolves through `config/paths.yaml` via `shorkie.config` (Python) or `scripts/common/env.sh` / a local `cfg()` helper (shell). The single machine-specific value is `work_root`.
- **Pinned external code** — the model framework is **not** vendored; it lives in two git submodules under `external/` (see below).
- **Catalogued data** — `data/manifest.json` lists every released artifact (gs:// URI, size, MD5) and `data/download.sh` fetches + verifies them.

The actual model code (the `SeqNN` architecture, training loop, ISM/variant scoring, data writers) lives in the **external Calico repos** this repo only *invokes*:
- **`external/baskerville-yeast`** — `hound_*.py` scripts (`hound_train.py`, `hound_data.py`, `hound_ism_*.py`, `yeast_test_genes.py`, …) and the importable `baskerville` package (`from baskerville import seqnn, dna, gene`). Upstream: https://github.com/calico/baskerville-yeast
- **`external/westminster`** — multi-fold orchestration, primarily `westminster_train_folds.py`. Upstream: https://github.com/calico/westminster

When a script's behavior depends on `hound_*`, `westminster_*`, or `baskerville.*`, the source of truth is the submodule, **not** this repo. Init them with `git submodule update --init` (pinned commits) and `pip install -e external/baskerville-yeast -e external/westminster`. Config keys `external.baskerville_scripts` / `external.westminster_scripts` point shell scripts at the `hound_*`/`westminster_*` entrypoints.

## The three model variants

The model is two-stage: **Shorkie LM** (a masked DNA language model pretrained on fungal genomes) is fine-tuned with supervised epigenomic/transcriptomic tracks to produce **Shorkie**. The repo covers three variants end-to-end:

| Variant | What | Train command | Differs by |
|---|---|---|---|
| **shorkie_lm** | masked DNA LM, multi-species fungal corpus (released tier = 165 Saccharomycetales), seq_len 16384 | `hound_train.py` (`loss=mlm`, `use_bert=true`, `unet_small_bert_drop`) | — (base) |
| **shorkie_finetuned** | supervised, 5215 ChIP-exo/MNase/RNA-seq tracks, R64, 8-fold CV | `westminster_train_folds.py … --restore <LM .h5>` | `task=fine-tune`, `lr=2e-5` |
| **shorkie_scratch** | **identical** supervised set, random init (ablation) | same command **without `--restore`** | `task=supervised`, `lr=1e-4` |

The only mechanistic difference between finetuned and scratch is the `--restore` flag + learning rate (verified: the two `params.json` differ in exactly those two `train` fields; model blocks are byte-identical). `scripts/02_train/README.md` documents the comparison.

## Execution model — read this before running or editing scripts

**Everything assumes the JHU Salzberg HPC cluster (SLURM) + the `yeast_ml` conda env.** GPU (`a100`/`ica100`, `-A ssalzbe1_gpu`) is only needed for model training/inference and a few notebooks (fig06/08/10); most data-wrangling/plotting is CPU. SLURM client binaries may be at `/cm/shared/apps/slurm/current/bin`.

- **Conda env:** `yeast_ml` (canonical interpreter `/home/kchao10/miniconda3/envs/yeast_ml/bin/python3`). Most scripts assume it is active; `minimal_example` uses the full path explicitly.
- **Paths are config-driven, not hardcoded.** Copy `config/paths.example.yaml` → `config/paths.yaml` (git-ignored) and set `work_root` (default `/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML` = `/scratch4/ssalzbe1/khchao/Yeast_ML`). Other roots interpolate from it. A few legacy build/eval roots have their own keys (`corpus_build_*_root`, `ssm_results_root`, `yeast_seqnn_eval_root`, `tools.*`) — override them too if you rerun the corpus build / genome eval. `src/shorkie/config.py` loads `$SHORKIE_CONFIG` → `config/paths.yaml` → `config/paths.example.yaml` and supports `${...}` interpolation.
- **SLURM scripts** carry `#SBATCH` headers; run with `sbatch script.sh`. `config/slurm.example.yaml` holds partition/account profiles. Large data/weights are on GCS — fetch via `data/download.sh` (see `data/manifest.json`); they are not in the repo.

## Directory map

- **`src/shorkie/`** — the installable package (single source of shared code; replaces the old copy-pasted helpers):
  - `config.py` — path resolution (`load`, `get`, `path`, `repo_root`).
  - `helpers/yeast_helpers.py` — sequence/coverage/ISM/plotting helpers (`process_sequence`, `predict_tracks`, `plot_coverage_track_bins`, `make_seq_1hot`, …).
  - `models/ensemble.py` — 8-fold ensemble loader + scoring (`load_ensemble`, `make_input`, `ensemble_predict`, `predict`, `logSED`).
  - `data/{util,bed_helper}.py`, `viz/load_cov.py` (`CovFace`, `read_coverage`, `seq_norm`).
- **`scripts/`** — all pipelines, staged `00 → 04` (see `scripts/README.md`):
  - `00_setup/` — submodule init, env checks, download wrappers.
  - `01_data_build/{lm_corpus,supervised_tracks}/` — genome-corpus pipeline (download → RepeatMasker/DUST mask → paralog/repeat filter → TFRecords) and FASTQ→BAM→BigWig→peaks→`hound_data.py`.
  - `02_train/{shorkie_lm,shorkie_finetuned,shorkie_scratch}/` — the three train drivers (each `params.json` + a config-parameterized `*.sh` with `--dry-run`).
  - `03_eval/{lm,supervised}/` — LM perplexity / genome eval / arch comparison, and track-prediction metrics.
  - `04_analysis/{shorkie_lm,shorkie,shorkie_scratch,others}/` — eQTL, MPRA, motif/MoDISco, ISM, attention, UMAP, SMT3, dependency maps, phylogeny, arch viz, ablations.
  - `common/` — `env.sh` (exports config roots), `submit.sh`/`slurm_header.sh` (portable `#SBATCH` + local/container fallback).
- **`notebooks/`** — 14 figure-reproduction notebooks (`figNN_<topic>.ipynb`), each importing from `shorkie`, pinned to the `yeast_ml` kernel; `notebooks/README.md` is the figure→notebook→upstream-stage→artifact index.
- **`config/`** — `paths.example.yaml`, `slurm.example.yaml` (templates; copy to the `.yaml` form).
- **`data/`** — small committed reference files (`R64_annotations/`, `species_lists/`) + `manifest.json` + `download.sh`. Large data is **not** committed.
- **`external/`** — the two pinned submodules. **`minimal_example/`** — self-contained logSED variant scorer (best place to learn model load + scoring). **`containers/`** — Dockerfile + Apptainer def (scheduler-free path).

## Conventions when editing or adding scripts

- **Numbered pipeline steps** (`0_`, `1_`, `2_`, sometimes `0a_`/`s5_`) encode order within a directory; preserve it when adding steps. Subdirectories nest the same way.
- **`.py` + `.sh` pairing:** a `foo.py` doing the work has a sibling `foo.sh` SLURM wrapper supplying paths/args. Follow this when adding a compute step.
- **Paths via config, never hardcoded.** Python: `from shorkie import config` then `config.path('<dotted.key>')` / module-level `ROOT = str(config.path('work_root'))` + f-strings. Shell: `source scripts/common/env.sh` for `${WORK_ROOT}`/`${DATA_ROOT}`/… or the self-contained one-liner used throughout `04_analysis`:
  `cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }` then `VAR="$(cfg dotted.key)"`. No `--mail-user=` literals (use `config/slurm.yaml`).
- **Shared code is imported, not copied.** Use `from shorkie.helpers.yeast_helpers import …`, `from shorkie.models.ensemble import …`, etc. Do not reintroduce the old per-directory `yeast_helpers.py`/`util.py`/`bed_helper.py`/`load_cov.py` copies.
- **Model layout:** trained models are an **8-fold ensemble** at `<model_dir>/train/f{0..7}c0/train/model_best.h5`, scored by averaging fold predictions. Accompanied by `params.json` (two keys: `train`, `model`) and a TSV targets sheet (`cleaned_sheet.txt`, ~5215 tracks).
- **Model input encoding:** Shorkie takes `(16384, 170)` tensors — channels 0–3 are DNA one-hot (A/C/G/T), 4–169 are species identity (column 114 = *S. cerevisiae*). See `minimal_example/run_shorkie_variant.py` and `shorkie.models.ensemble.make_input`.
- **logSED** = `log2(Σ_alt_bins + 1) − log2(Σ_ref_bins + 1)` over gene-body output bins; the core variant-effect metric throughout the eQTL and MPRA analyses.
- **Chromosome naming gotcha:** the Ensembl GTF uses bare Roman (`I`..`XVI`); the cleaned FASTA + released bigwigs use `chrI`..`chrXVI`. When fetching by gene, remap (see fig06/fig08 `to_fasta_chrom`).

## Heavy external tools

Invoked from shell scripts (must be on `PATH` / in the env, or pointed to via `tools.*` config keys): `modisco`/`tfmodisco` (most used), `bedtools`, `samtools`, `STAR`, `macs3`, `RepeatMasker`/`RepeatModeler`, `minimap2`, `nucmer`/`mummerplot`/`show-coords`, `busco`, `dashing2`, `mash`, `meme`/`fimo`. Most are in `environment.yml`; `dashing2` is not on bioconda (build from source). GPU training also needs `tensorrt` + a CUDA TensorFlow build (baked into `containers/`).
