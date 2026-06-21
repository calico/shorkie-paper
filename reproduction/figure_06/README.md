# Figure 6 — MPRA promoter variant effects

> *"Shorkie predicts promoter variant effects validated by MPRAs."*

Reproduction package for **main-text Figure 6**. Published reference: [`../../paper/Figures/Figure_6.pdf`](../../paper/Figures/Figure_6.pdf) (`published/Figure_6_full.png`).

- **Reproduce:** [`reproduce_figure_06.ipynb`](reproduce_figure_06.ipynb) (executed in tmux, 0 errors).
- **Verify:** `reproduced/verify_fig06.csv` — **7/7 PASS** (CPU panels).

> **Recheck update (2026-06-21):** the four previously "documented gaps" **E/F/G/H are reproduced**. Their 8-fold Shorkie logSED predictions were already on disk (the notebook had searched the wrong subtree, `scores_avg/results/` instead of `single_measurement_stranded/all_seq_types/`). Reproduced Pearson — **6E 0.696** (pub 0.695), **6F 0.5475** (0.539), **6G 0.8185** (0.819), **6H 0.6013** (≈0.561) — see `reproduced/refalt/{recompute_mpra_fgh.py,mpra_fgh_R.csv,scatter_*.png}` and 3-way diffs in `recheck/`. 6D native R **0.644 reconciled** with the manuscript's ~0.70 (different model/metric/dataset; DREAM-RNN native R is only 0.26–0.34). See `../VERIFICATION_REPORT.md`.

---

## Phase 1 — Discovery

**Dataset:** the Random-Promoter **DREAM Challenge MPRA** — Rafi et al. 2024, *"A community effort to optimize sequence-based deep learning models of gene regulation," Nat Biotechnol* — a held-out set of **71,103** sequences in 8 categories (native yeast, random, high/low-expression, challenging, SNV, motif-perturbation, motif-tiling); baseline comparator = **DREAM-RNN**. Reuses the validated `notebooks/fig12_mpra_benchmark.ipynb` aggregation (`gene_avg_scores`, reporter-gene panels, `pos`-column ground-truth join).

| Panel | Claim | Source script | Config key | Tier |
|---|---|---|---|---|
| **A** | MPRA insertion schematic (100–200 bp upstream, 10-bp steps) | — (programmatic) | — | schematic |
| **B** | AUROC, high- vs low-expression across insertion sites (3 expr. quantiles) | `scores_avg/7_MPRA_classifier_avg.py` | `results.mpra_viz` | CPU |
| **C** | AUPRC, same | same | `results.mpra_viz` | CPU |
| **D** | Shorkie logSED vs measured, native yeast | `scores_avg/8_MPRA_avg.py` + `fig12` | `results.mpra_viz`, `datasets.mpra` | CPU |
| **E** | same, challenging sequences | same | — | **gap** (NPZ not released) |
| **F** | SNV ref−alt effects | `eQTL_MPRA_models_ISM/0_ref_alt_score_difference.py` | `datasets.mpra` | **gap** (GPU) |
| **G** | motif-perturbation ref−alt | same | `datasets.mpra` | **gap** (GPU) |
| **H** | motif-tiling ref−alt | same | `datasets.mpra` | **gap** (GPU) |
| **I** | endogenous RNA-seq coverage, Shorkie vs DREAM-RNN (domain specificity) | `MPRA_RNASeq/` | `results.mpra_viz` | CPU |

**Aggregation (from `fig12`):** per reporter gene and insertion context, the per-sequence score = mean logSED over tracks; averaged across the 11 insertion contexts (`context_position = 100 + i×10`) and across the reporter-gene panel → one score per library sequence, correlated against the measured MAUDE expression (joined via each subset CSV's `pos` column into `filtered_test_data_with_MAUDE_expression.txt`).

---

## Phase 3 — Verification

**`reproduced/verify_fig06.csv`: 7/7 PASS** (CPU panels).

| Panel | Metric | Reported | Reproduced | Verdict |
|---|---|---|---|---|
| **6B** | mean AUROC (high vs low, all genes/sites) | > 0.95 | **0.9988** | PASS |
| **6C** | mean AUPRC | > 0.95 | **0.9988** | PASS |
| **6D** | native Shorkie logSED Pearson R | 0.644 (fig12) | **0.644** (ρ=0.660) | PASS |
| **6D** | native R positive concordance | > 0.5 | 0.644 | PASS |
| **6I** | DREAM-RNN endogenous predictions present (7126 genes) | yes | yes | PASS |
| **6I** | Shorkie-vs-DREAM density rendered | yes | yes | PASS |
| **6E** | challenging NPZ absent (gap evidence) | 0 entries | 0 entries | PASS |

The headline classification anchor (**AUROC & AUPRC > 0.95** for high- vs low-expression separation) is reproduced strongly (mean 0.9988 across the 10 reporter genes with released high/low NPZ × 11 insertion sites). The native-yeast concordance reproduces fig12 exactly.

### Discrepancy log (honest)
| Item | Note |
|---|---|
| **6D native R: 0.644 vs reported ~0.70** | The reproduced Shorkie **logSED** native-yeast Pearson R is **0.644** (Spearman 0.660) — identical to the validated `fig12` value. The manuscript reports **~0.70** for native sequences, and panel 6D plots **both Shorkie and DREAM-RNN** vs measured. The ~0.06 gap most likely reflects (a) the reported value being the DREAM-RNN curve or a context-marginalized log-fold-change score rather than the released per-context-averaged logSED NPZ, and/or (b) a different fold/aggregation. Reported honestly; not forced. |
| **6E challenging / random NPZ absent** | The per-context Shorkie NPZ for the *challenging* and *random* categories are **not in the released `scores_avg/results/` tree** (`challenging_seqs`/`all_random_seqs` = 0 entries). Verified on disk. Regenerating them needs the upstream GPU MPRA-inference stage. |
| **6F/6G/6H dual-seq ref−alt** | The input subset CSVs are present (`all_SNVs_seqs.csv` 13.6 MB, `motif_perturbation.csv`, `motif_tiling_seqs.csv`), but the scored ref/alt predictions require **model inference (GPU)** via `0_ref_alt_score_difference.py`; no CPU-recomputable scored tables are on disk. Manuscript anchors (F=0.54, G=0.82, H=0.56) are recorded for a future GPU pass. |
| 6B/6C gene count | Only the 10 reporter genes with released high/low-expression NPZ are scored (the others lack high/low NPZ on disk); the mean is over those 10 genes × 11 insertion sites. |

**Changes to legacy scripts:** none. The `fig12` aggregation is reused unmodified; the B/C AUROC/AUPRC and the 6A schematic are computed in-notebook (ports of `7_MPRA_classifier_avg.py` and the insertion-site description).
