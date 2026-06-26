# 04_analysis/shorkie_scratch — random-init (no-LM) ablation analyses

> **Advanced / reproduction-only.** You don't need this to *run* Shorkie. These ablation analyses
> (random-init vs LM-pretrained) are reported in the paper; to use Shorkie, see `examples/` +
> `minimal_example/` on the released artifacts (`data/download.sh`).

Stage-04 ablation analyses for the **random-init / from-scratch supervised
baseline** (`02_train/shorkie_scratch/`, no MLM pretraining). These compare it
against the LM-pretrained `Shorkie` to quantify the impact of language-model
pretraining: architecture choice, learning rate, finetuning-corpus choice, and
in-silico-mutagenesis motif recovery. Each working `*.py` keeps its paired
`*.sh` SLURM runner; scripts read paths from `config.path("work_root")` (via
`shorkie.config`) and source `scripts/common/env.sh` (`WORK_ROOT`,
`BASKERVILLE_SCRIPTS`).

| Step | Directory / file | Purpose |
|------|------------------|---------|
| 1 | `1_architecture_search/` | Cross-fold train/valid loss & Pearson-r curves comparing model architectures (CovNet/U-Net smaller, U-Net small scratch vs. `Shorkie`) |
| 2 | `2_lr_search/` | Same curves swept across learning-rate variants (`learning_rate_*`), with averaged, subplot, and bar summaries |
| 3 | `3_finetuning_datasets/` | Curves comparing finetuning corpora / MLM pretraining sources (R64 yeast, 80-strain, 1341-fungal, Saccharomycetales) and best-valid-r vs. MLM-perplexity scatter |
| — | `ism_motif/motif_shorkie_random_init__RP_TSS/` | ISM saliency over RP/TSS gene windows for the scratch model (see below) |

## Per-step files

- **`1_architecture_search/`** — `1_compare_variants_avg.py` (averaged curves),
  `2_compare_variants_subplot.py` (per-metric panels), `3_merge_pngs.py` /
  `3_merge_pngs_AB.py` (figure assembly).
- **`2_lr_search/`** — `1_compare_lr_variants_avg.py`,
  `2_compare_lr_variants_subplot.py`, `3_compare_lr_variants_bar.py`,
  `4_merge_plots.py`.
- **`3_finetuning_datasets/`** —
  `1_compare_cross_valid_curves_avg_various_mlm.py`,
  `2_compare_cross_valid_curves_subplot_various_mlm.py`,
  `plot_best_valid_r_vs_perplexity.py`, `merge_plots.py` / `merge_plots_BD.py`.

All three search dirs parse per-fold `train/f{0..7}c0/train.out` logs and emit
plots under `./results/` (or `--out_dir`); they are CPU-only matplotlib jobs.

## ism_motif (in-silico mutagenesis)

`ism_motif/motif_shorkie_random_init__RP_TSS/` runs ISM on the scratch model and
plots DNA-logo saliency, in numbered order:

| Step | File | Purpose |
|------|------|---------|
| run | `ism_run/*.sh` | GPU ISM via baskerville `hound_ism_bed.py` over RP / TSS gene windows (`--stats SUM,logSUM,logSED`) |
| 1 | `1_saliency_score_genic_intergenic.py` (+`.sh`) | logSED saliency split by genic / intergenic region |
| 2 | `2_save_tensor.py` (+`.sh`) | serialize per-window saliency tensors |
| 3 | `3_plot_dna_logo.py` (+`.sh`) | render DNA sequence-logo plots |

## External tools / requirements

- **baskerville-yeast** `hound_ism_bed.py` (`$BASKERVILLE_SCRIPTS`) — GPU,
  needed only for the `ism_motif` ISM run; the search/plotting steps are
  CPU-only.
- Inputs are the trained 8-fold models, `params.json`, the R64 GTF/FASTA, and
  RP/TSS window BEDs under `$WORK_ROOT` (no modisco/meme used in this stage).
