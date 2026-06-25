# Figure 3 — Shorkie architecture and RNA-seq prediction performance

> *"Shorkie architecture and RNA-seq prediction performance across multiple scales."*

Reproduction package for **main-text Figure 3**. Published reference: [`../../paper/Figures/Figure_3.pdf`](../../paper/Figures/Figure_3.pdf) (`published/Figure_3_full.png`).

- **Reproduce:** [`fig03_supervised_rnaseq_prediction.ipynb`](../../notebooks/fig03_supervised_rnaseq_prediction.ipynb) (executed in tmux, 9/9 cells, 0 errors, 5 figures).
- **Verify:** [Verification](#phase-3--verification) + `reproduced/verify_fig03.csv`.

---

## Phase 1 — Discovery

Figure 3 establishes that LM-pretrained **Shorkie** beats the from-scratch **Shorkie_Random_Init** at predicting RNA-seq/ChIP coverage. 10 panels.

| Panel | Claim | Type | Generating script | Input + config | Notebook |
|---|---|---|---|---|---|
| **A** | Shorkie architecture (U-Net + 8 transformer blocks + task heads) | schem | — (params) | `02_train/shorkie_finetuned/params.json` → `models.shorkie_finetuned` | none |
| **B** | β-estradiol induction protocol (→ time-point RNA-seq tracks) | schem | — | `datasets.bigwigs` | none |
| **C** | bin-level Pearson R distribution by track type — **split violin** | comp | `…/1_bin_level_freq_viz.py::plot_box_violin` | `train/f{i}c0/eval/acc.txt` (median-per-id) | none |
| **D** | bin-level R scatter (Random_Init x vs Shorkie y) | comp | `…/1_bin_level_freq_viz.py::scatter_all_groups_scatter` | top-level `acc.txt` (mean-per-id) | none |
| **E** | gene-level R scatter (`pearsonr`) | comp | `…/3_gene_level_score_dist_viz.py` **`level="track"`** | per-dt `gene_level_eval_rc/…/{dt}/acc.txt` | `fig03` |
| **F** | quantile-normalized gene-level R scatter (`pearsonr_norm`) | comp | `…/3_gene_level_score_dist_viz.py` **`level="track"`** | per-dt `acc.txt` | `fig03` |
| **G** | within-gene track-level R (`pearsonr_gene`) | comp | `…/3_gene_level_score_dist_viz.py` **`level="gene"`** | per-dt `gene_acc.txt` (drop bottom-10% cov) | none |
| **H–J** | coverage at RPL7A / RPS16B-RPL13A / EFM5 (obs vs Shorkie vs Random_Init) | gpu | `…/3_viz_rnaseq_tracks/2_yeast_rna_seq_models.py` (+ `3_gene_annotation_viz.py`) | ensemble + bigwig + GTF | `fig03` |

> **Note on the panel E/F/G titles:** the released `3_gene_level_score_dist_viz.py` has an *inverted*
> `level`↔title naming — panels **E/F** (titled "Gene Level") are emitted at `level="track"` (reading the
> per-data-type `acc.txt`), and panel **G** (titled "Track Level") at `level="gene"` (reading `gene_acc.txt`).

**Models:** Shorkie = `self_supervised_unet_small_bert_drop`; Shorkie_Random_Init = `supervised_unet_small_bert_drop_variants/learning_rate_0.0005` (the script's optimized-LR from-scratch baseline) — both 8-fold under `datasets.supervised_root`. Env `yeast_ml` (CPU for C–G; GPU for H–J). Eval-table columns: bin `index,pearsonr,r2,identifier,description`; gene `gene_id,pearsonr,pearsonr_norm,pearsonr_gene,…`. Track types parsed from `description`: RNA-Seq (3053), ChIP-exo (1128), 1000-strain RNA-Seq (1014), ChIP-MNase (20).

### Reproduction scope
- **Fully reproduced (CPU), exact-match to the published figure:** 3C (split violin), 3D (bin scatter), 3E/3F/3G (gene-level scatters). 3A schematic (programmatic block-stack from `params.json`, track-count box ChIP-exo 1128 / Histone 20 / RNA-Seq 3053 / 1000-strain 1014); 3B documented schematic.
- **GPU (cached):** 3H–J coverage at the 3 loci × {Shorkie, Random_Init}, one model per process via `sbatch reproduction/figure_03/panels/run_coverage.sbatch`; re-rendered (gene-annotation track + published colors) by `panels/plot_coverage.py`. No GPU rerun needed — the coverage NPZ is cached in `reproduced/coverage/`.

---

## Phase 3 — Verification

**`reproduced/verify_fig03.csv`: 33/33 PASS** — checked against the values **printed on the published
figure** (recomputed with the source-of-truth recipes; see `recheck/DISCREPANCIES.md`).
- **3C** — 8 split-violin medians, e.g. RNA-Seq Shorkie **0.776** / Random_Init **0.703**; 1000-strain 0.629/0.579; ChIP-MNase 0.446/0.424; ChIP-exo 0.356/0.315. All match.
- **3D/3E/3F/3G** — every group mean point matches the published printed value (3D RNA-Seq 0.71/0.78; 3E 0.80/0.88; 3F 0.36/0.38; 3G 0.61/0.73; + the 1000-strain points).
- **3H/3I/3J** — coverage Pearson R vs observed 0.96–0.99 (Shorkie), 0.85–0.97 (Random_Init).

### Discrepancy log (honest) — see [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md) for full detail
| Item | Status | Root cause |
|---|---|---|
| 3C plot type | **fixed** | published is a **split violin** (`plot_box_violin`); the reproduction had rendered the KDE branch of the same script |
| 3C Random_Init median | **fixed** | the figure plots the **lr=5e-4 variant** (RNA-Seq median **0.703**); the prior verify checked **0.67** = the manuscript-*text* plain-supervised baseline (a different model) |
| 3E/3F/3G gene-level means | **fixed** | the prior reproduction pooled `gene_acc.txt` across all tracks (→0.9178); the correct recipe reads per-data-type `acc.txt` (E/F, `level="track"`) and `gene_acc.txt` (G, `level="gene"`, drop bottom-10% `coverage_norm_self`) and reproduces 0.80/0.88, 0.36/0.38, 0.61/0.73 **exactly** |
| 3H/I/J styling | **fixed** | fine-tuned row recoloured red→**orange**; gene-annotation track + intron dashed lines added (GTF) |
| **3G "87.8% of genes"** | **residual (documented)** | the manuscript-text fraction does **not** reproduce — pearsonr_gene RNA-Seq fraction-above-diagonal is robustly **~75%** across every metric/level/filter/pooling variant; the panel-G scatter and its means (0.61/0.73) reproduce exactly, so only the headline text fraction differs (likely an earlier checkpoint / gene-filter at writing time) |

> **Recheck update (2026-06-22):** full figure-exactness pass. All four data panels (C/D/E/F/G) and the
> coverage panels (H/I/J) now match the published figure in plot type, styling, colours, and printed
> numbers (33/33). Builders live in `recheck/` (`build_3C_violin.py`, `build_3DEFG_scatter.py`,
> `build_verify_fig03.py`, `make_sidebyside_fig03.py`); `panels/plot_coverage.py` gained the
> gene-annotation track. Side-by-sides in `recheck/Figure_3_published_vs_reproduced.png`.

**Changes to legacy scripts:** none edited; the released figure scripts' recipes are reproduced faithfully.
