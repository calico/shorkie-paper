# 04_analysis/shorkie — fine-tuned (supervised) model analyses

Downstream analyses of the fine-tuned 8-fold Shorkie ensemble: cis-eQTL
benchmarking, the DREAM-Challenge MPRA benchmark, and in-silico mutagenesis
(ISM) + MoDISco motif discovery over ribosomal-protein (RP) / TSS genes and
induction time-series. Scripts read paths from `config.path("work_root")` (and
`motif_db_dir`) via `shorkie.config` / `scripts/common/env.sh`; each working
`*.py` keeps a paired `*.sh` SLURM runner, and numbered prefixes encode order.

| Sub-stage | Directory | Purpose |
|-----------|-----------|---------|
| eQTL | [`eqtl/`](eqtl/README.md) | cis-eQTL benchmark across models/datasets (GWAS preprocessing → neg-set generation → variant scoring → figures). **See its own README.** |
| MPRA | `mpra/` | DREAM-Challenge MPRA benchmark via logSED variant scoring (`log2(ALT+1) − log2(REF+1)`) |
| ISM/motif | `ism_motif/` | ISM attribution + MoDISco motif discovery over RP/TSS genes and induction time-series |

## `mpra/` — DREAM-Challenge MPRA benchmark

| Step | Directory | Purpose |
|------|-----------|---------|
| 1 | `1_data_preprocessing/` | build single / dual-index MPRA TSV input sets |
| 2 | `2_hound_mpra_run/` | run MPRA inference, write `sed.h5`, verify logSED (`eval_MPRA_h5_logSED.py`; `run_MPRA_{pos,neg}*.sh`) |
| 3 | `3_process_hdf5_logsed/` | extract T0 RNA-seq index; analyze per-gene logSED (single / dual indices) |
| 4 | `4_mpra_high_low_seq/` | high-vs-low expression sequence viz, MPRA classifier (AUROC/AUPRC) + aggregation |
| 5 | `5_mpra_viz/` | reference-model track viz and predicted-vs-observed scatter/regression figures |

## `ism_motif/` — ISM + MoDISco motif discovery

| Directory | Purpose |
|-----------|---------|
| `motif_shorkie__RP_TSS/` | core ISM pipeline: `ism_run/` (`hound_ism_bed.py` over RP/TSS/Proteasome/splice-site windows) → `1_create_gene_bed/` (gene/TSS BEDs) → DNA-logo & saliency plots → `2_modisco_analysis/` (MoDISco + report) and `3_timepoint_analysis/` (per-timepoint score viz + diff MoDISco) |
| `motif_shorkie_ism__snp/` | SNP-centered ISM (`hound_ism_snp.py`) with score-HDF5 checks and DNA-logo plots |
| `motif_shorkie__time_series/` | induction time-series target-vs-prediction comparison and per-track metrics |

## External tools

- **MoDISco** (`modisco motifs` / `modisco report`) for motif discovery and
  HTML reports; `modisco report` matches against the MEME files under
  `motif_db_dir` (`merged_meme.meme`, `merged_meme_high_conf.meme`).
- **`hound_*`** scripts from the `baskerville-yeast` checkout —
  `hound_ism_bed.py`, `hound_ism_snp.py` (ISM scoring) and the MPRA / SED
  scoring entry points — invoked by the `*.sh` runners (`work_root` resolved
  via `shorkie.config`).
