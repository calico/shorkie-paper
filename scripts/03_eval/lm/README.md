# 03_eval/lm — Shorkie LM (masked language model) evaluation

Evaluation of the masked-LM pretraining stage: held-out perplexity/loss across
corpus tiers and architectures, characterization of the genome corpus itself,
and a gene-level comparison between the self-supervised (big-U-Net) LM and its
supervised counterpart. Most working `*.py` keep a paired `*.sh` runner, and all
scripts read paths from `config/paths.yaml` via `shorkie.config` (shell scripts
source `scripts/common/env.sh`) — no hardcoded filesystem paths.

| Subdir | Purpose |
|--------|---------|
| `lm_model_eval/` | Parse held-out test logs and plot MLM loss / perplexity comparisons across architectures (`Conv_Small/Big`, `U-Net_Small/Big`) and corpus tiers (`r64`, `strains`, `saccharomycetales`, `fungi_1385`). |
| `genome_evaluation/` | Characterize the LM genome corpus: gene-density windows, BUSCO completeness, genome-distance, and repeat/sequence-length distributions (see steps below). |
| `model_evaluation/` | Gene-level prediction comparison of the self-supervised vs. supervised big-U-Net via baskerville `yeast_test_genes.py`, then visualize per-gene metrics. |

### `lm_model_eval/`

| Step | File | Purpose |
|------|------|---------|
| 1 | `1_model_arch_comparison_test_eval.py` | Grouped bar charts of overall test loss per architecture × corpus tier. |
| 2 | `2_model_arch_comparison_test_eval_loss_perplexity.py` | Parse overall + region-specific loss/perplexity from test reports. |
| 3 | `3_dataset_comparison.py` | Train/valid loss curves across corpus tiers (CLI: smoothing window, steps, dpi). |
| 4 | `4_model_arch_comparison_all.py` | Combined per-dataset train/valid curves with minimum markers. |

### `genome_evaluation/`

| Step | Dir | Purpose |
|------|-----|---------|
| 1 | `1_window_eval/` | Count genes / coding vs. noncoding per genomic window (overall, per-chromosome) and summary stats. |
| 2 | `2_annotation_eval/busco/` | Download fungi `odb10` DB, extract proteins, run BUSCO completeness, and plot results. |
| 3 | `3_genome_dist/` | Pairwise genome distance to R64: `nucmer`/`show-coords`/`mummerplot` alignment, plus `dashing2/` and `mash/` sketch-distance + viz. |
| 4 | `4_repeat_eval/` | Venn diagram of soft-masked (lowercase) repeat regions across masking tools. |
| 5 | `5_repeat_masked_region/` | Plot per-genome soft-masked repeat ratios over sequence windows. |
| 6 | `6_plot_fasta_seq_len_dist/` | Sequence-length distribution of corpus FASTAs by database/type. |

### `model_evaluation/`

| Step | File | Purpose |
|------|------|---------|
| 1 | `1_self_supervised_model_evaluation.sh`, `1_supervised_model_evaluation.sh` | Run gene-level test eval over 8 folds for the self-supervised and supervised big-U-Net (baskerville `yeast_test_genes.py`). |
| 2 | `2_viz.py` | Collate per-fold `acc.txt` (Pearson r, R²) into self- vs. supervised comparison plots. |
| 3 | `3_viz_all_gene.py`, `3_viz_all_gene_split_by_type.py` | Per-gene metric distributions, overall and split by gene type. |

## External tools

`busco` (with the fungi `odb10` lineage DB), `dashing2`, `mash`, and MUMmer
(`nucmer`, `show-coords`, `mummerplot`) must be on `PATH`/in the env; `dashing2`
is located via `config` key `tools.dashing2_bin`. The genome-distance and BUSCO
runners also take a `data_type` argument (corpus tier, e.g. `strains`) as `$1`.

## Config keys read

`work_root`, `corpus_build_data_root`, `corpus_build_results_root`,
`yeast_seqnn_eval_root`, `tools.dashing2_bin`, `tools.repeatmasker_lib`,
`tools.rmrb_lib`. The `model_evaluation/` runners also use `BASKERVILLE_SCRIPTS`
(exported by `scripts/common/env.sh`) to locate `yeast_test_genes.py`.
