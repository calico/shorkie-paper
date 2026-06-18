# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is (and is not)

This is the **analysis/reproducibility repo for the Shorkie paper** — a collection of ~206 Python scripts, ~123 shell scripts, and a few notebooks that reproduce the figures and benchmarks in the paper. **It is not an installable package and has no build, lint, or test system.** There are no `setup.py`, `pyproject.toml`, `requirements.txt`, CI config, or tests; do not look for or invent them.

The actual model code (the `SeqNN` architecture, training loop, ISM/variant scoring, data writers) lives in two **external Calico repos** that this repo only *invokes*:
- **baskerville-yeast** — `hound_*.py` scripts (e.g. `hound_train.py`, `hound_ism_bed.py`, `hound_ism_snp.py`, `hound_MPRA_folds.py`, `hound_model_viz.py`) and the importable `baskerville` package (`from baskerville import seqnn, dna, gene`). https://github.com/calico/baskerville-yeast
- **westminster** — multi-fold orchestration, primarily `westminster_train_folds.py`. https://github.com/calico/westminster

When a script's behavior depends on `hound_*`, `westminster_*`, or `baskerville.*`, the source of truth is the external repo, **not** this one. On this cluster they are checked out under the `Yeast_ML` root described below.

## Execution model — read this before running or editing scripts

**Everything assumes the JHU Salzberg HPC cluster (SLURM) and a specific filesystem layout.** Scripts are not portable as-is.

- **Conda env:** `yeast_ml` (canonical interpreter `/home/kchao10/miniconda3/envs/yeast_ml/bin/python3`). Most scripts assume it is already activated; the `minimal_example` uses the full path explicitly.
- **SLURM:** 76 of the shell scripts are `sbatch` batch scripts with `#SBATCH` headers; run them with `sbatch script.sh`, not `bash`. Partitions/accounts seen in the repo: `bigmem` (`-A ssalzbe1_bigmem`, CPU-heavy preprocessing), `parallel`, and `a100` (`-A ssalzbe1_gpu`, GPU training/inference). GPU is only needed for model training and inference; most data-wrangling/plotting steps are CPU.
- **Hardcoded paths everywhere (~409 occurrences):** scripts reference the absolute root `/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML`. `scr4_ssalzbe1` is a symlink to `/scratch4/ssalzbe1`, so this resolves to `/scratch4/ssalzbe1/khchao/Yeast_ML` — a **sibling of this repo**. That `Yeast_ML/` tree holds the genomes, TFRecords, BigWigs, trained model `.h5` files, and the `baskerville-yeast` checkout (westminster lives under `Yeast_ML/seq_experiment/westminster`). **To run anything outside this exact user/cluster, you must edit these paths.** When adapting a script, rewriting the hardcoded root is the first step, not an afterthought.

## Directory map / architecture

The repo is organized around the paper's two models and one ablation study. **The model is two-stage: `Shorkie LM` (a masked DNA language model pretrained on fungal genomes) is fine-tuned with supervised epigenomic/transcriptomic tracks to produce `Shorkie`.** The directory split mirrors this.

- **`model/`** — the published training commands (just the launch snippets; real logic is in baskerville/westminster):
  - `model/shorkie_lm/` — LM pretraining via `hound_train.py` (`train.sh`).
  - `model/shorkie/` — supervised fine-tuning from the LM checkpoint, 8 folds, via `westminster_train_folds.py` (`make_model.sh`), restoring `LM_Johannes/.../model_best.h5`.
  Each has its own `params.json` (two top-level keys: `train` and `model`).

- **`analysis/shorkie_lm/`** — analyses of the language model. Notably `data_preprocessing/` is the full genome-corpus pipeline (download FASTA/GTF → repeat masking with RepeatMasker/RepeatModeler/DUST → paralog/repeat filtering with minimap2 & nucmer/mummer → genome-similarity with dashing2/mash → TFRecord generation). Also `genome_evaluation/` (BUSCO, window stats), `lm_model_eval/`, `motif_analysis/` (heavy tfmodisco use), `attention_map/`, `umap_cluster_promoter/`, `lm_SMT3_viz/`.

- **`analysis/shorkie/`** — analyses of the fine-tuned model: `track_data_preprocess/` (ChIP/MNase/RNA-seq → BAM → BigWig coverage via STAR/samtools/macs3/bedtools), `track_prediction_eval/`, `eqtl/` (cis-eQTL benchmarking — see its own `README.md`), `eqtl_data/`, `mpra/` (DREAM Challenge MPRA benchmark, logSED scoring), `ism_motif/` (in-silico mutagenesis + modisco motif discovery over RP/TSS genes and induction time-series).

- **`analysis/shorkie_Random_Init/`** — ablation baselines trained from random init (no LM pretraining): `1_architecture_search/`, `2_lr_search/`, `3_finetuning_datasets/`, `ism_motif/`.

- **`analysis/others/`** — `phylogenetic_tree/` (species tree from NCBI taxonomy) and `viz_shorkie_lm_arch/`.

- **`minimal_example/`** — the one self-contained, documented entry point: `run_shorkie_variant.py` computes a logSED variant-effect score for a single SNP from a trained ensemble. Best place to learn how the model is loaded and scored without the cluster pipeline. `run_example.sh` is its SLURM wrapper.

- **`data/R64_annotations/`** — small committed reference files (`chrom.sizes`, repeat/gap BEDs) for the *S. cerevisiae* R64 genome. Large data (genomes, TFRecords, BigWigs, weights) is **not** in the repo — see the `gs://shorkie-paper/...` and `gs://seqnn-share/...` URLs in `README.md`.

## Conventions to follow when editing or adding scripts

- **Numbered pipeline steps:** filenames are prefixed `0_`, `1_`, `2_`, … (sometimes `0a_`) to encode execution order within a directory. Subdirectories nest the same way (`1_data_download/`, `2_repeat_region_masking/`, …). Preserve this ordering when adding steps.
- **`.py` + `.sh` pairing:** a `foo.py` doing the work commonly has a sibling `foo.sh` that is the SLURM wrapper supplying paths/args. When adding a compute step, follow this pattern.
- **Duplicated helpers, not shared imports:** `yeast_helpers.py` (sequence/coverage/plotting helpers) is **copied** into ~9 analysis subdirectories rather than imported from a package; `util.py` and `bed_helper.py` are likewise duplicated. There is no `__init__.py`/package root. If you fix a bug in one `yeast_helpers.py`, check whether sibling copies need the same fix.
- **Model layout:** trained models are an **8-fold ensemble** at `<model_dir>/train/f{0..7}c0/train/model_best.h5`, scored by averaging fold predictions. Accompanied by `params.json` (architecture) and a tab-separated targets sheet (`sheet.txt` / `cleaned_sheet.txt`, ~5215 ChIP/RNA-seq tracks).
- **Model input encoding:** Shorkie takes `(16384, 170)` tensors — channels 0–3 are DNA one-hot (A/C/G/T), channels 4–169 are species identity (column 114 = *S. cerevisiae*). See `minimal_example/run_shorkie_variant.py:make_input`.
- **logSED** = `log2(Σ_alt_bins + 1) − log2(Σ_ref_bins + 1)` over gene-body output bins; the core variant-effect metric, computed throughout the eQTL and MPRA analyses.
- **Heavy external tools** invoked from shell scripts (must be on `PATH`/in the env): `modisco`/`tfmodisco` (motif discovery — by far the most used), `bedtools`, `samtools`, `STAR`, `macs3`, `RepeatMasker`/`RepeatModeler`, `minimap2`, `nucmer`/`mummerplot`/`show-coords`, `busco`, `dashing2`, `mash`.
