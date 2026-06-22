# Figure 6 — deep recheck: discrepancies, root causes, and resolutions

Figure 6 = *"Shorkie predicts promoter variant effects validated by MPRAs"* (Random-Promoter
DREAM Challenge MPRA, Rafi et al. 2024; 71,103 test sequences; baseline = **DREAM-RNN**).

This recheck reproduced **every panel A–I exactly from cached data (no GPU)** and found that the
previous reproduction (which reported "7/7 PASS") was **substantially incomplete and partly
mislabeled**. All issues below are now fixed; the reproduced Pearson/Spearman match the values
printed on the published figure to **|Δ| ≤ 0.001** for all of D–I (both models).

## Reproduced vs published (printed values; Pearson, Spearman)

| Panel | Set | Shorkie (pub) | Shorkie (repro) | DREAM-RNN (pub) | DREAM-RNN (repro) |
|---|---|---|---|---|---|
| 6D | Yeast (single) | 0.695, 0.718 | **0.695, 0.718** | 0.891, 0.891 | **0.891, 0.891** |
| 6E | Random (single) | 0.744, 0.758 | **0.744, 0.758** | 0.981, 0.983 | **0.981, 0.983** |
| 6F | SNV (dual Δ) | 0.539, 0.420 | **0.539, 0.420** | 0.866, 0.709 | **0.866, 0.709** |
| 6G | Motif Perturbation (dual Δ) | 0.819, 0.745 | **0.819, 0.745** | 0.983, 0.970 | **0.983, 0.970** |
| 6H | Motif Tiling (dual Δ) | 0.561, 0.546 | **0.561, 0.546** | 0.943, 0.944 | **0.943, 0.944** |
| 6I | RNA-seq coverage density | r=0.895, ρ=0.837 | **0.895, 0.837** | r=0.249, ρ=0.261 | **0.249, 0.261** |
| 6B | AUROC vs insertion position | >0.95 (3 quantile aggs) | **mean 0.995, aggs >0.95** | — | — |
| 6C | AUPRC vs insertion position | >0.95 | **mean 0.996, aggs >0.95** | — | — |

## Defects found in the previous reproduction, with root cause + fix

### 1. Missing DREAM-RNN (green) subpanels — D, E, F, G, H (and the DREAM half of I)
**What:** each published panel D–I is a *two-scatter composite* — a **Shorkie (blue)** scatter
and a **DREAM-RNN (green)** scatter side-by-side. The previous reproduction rendered only the
Shorkie halves; all five green subpanels and the DREAM half of I were absent.
**Root cause:** the notebook reused only the single-model `fig12` aggregation and never loaded the
DREAM-RNN comparison predictions.
**Fix:** the DREAM-RNN per-sequence predictions are cached on disk
(`data/random-promoter-dream-challenge-2022/data/DREAM-RNN_output.txt`, 71,103 rows). The recheck
builders compute the green subpanels exactly as the source scripts
(`MPRA_ref_model_viz_single_index.py`, `MPRA_ref_model_viz_dual_indices.py`). Reproduced DREAM
Pearson **and** Spearman match the figure to ≤0.001 for all five panels.

### 2. Panel E mislabeled "challenging" instead of "Random"
**What:** the previous reproduction treated panel 6E as the *challenging* sequences (Pearson
≈0.696) and reported the *random* category as an "absent NPZ" gap.
**Root cause:** the README/notebook panel map was wrong; *challenging* is a **supplementary**
panel (Fig S25), not main Fig 6. Published main-figure 6E = **Random Sequences**.
**Fix:** 6E is now the Random panel — Shorkie 0.744, DREAM 0.981 (both exact). The random NPZ are
present in the complete stranded tree (they were only absent from the `scores_avg/results` subtree
the notebook had searched).

### 3. Panel D value off (0.644 → 0.695) — wrong NPZ subtree
**What:** the previous 6D reported Shorkie Pearson 0.644 and documented a ~0.06 "residual" vs the
manuscript's ~0.70.
**Root cause (two compounding issues):**
 (a) it read the `MPRA_promoter_seqs/scores_avg/results/yeast_seqs` subtree instead of the
     **stranded** tree `MPRA_promoter_seqs/results/single_measurement_stranded/all_seq_types/`
     used by the figure; and
 (b) it averaged the Shorkie logSED over **all 11 insertion contexts** rather than the **180 bp**
     context the figure uses.
**Fix:** using the stranded tree + the 180 bp context, 6D Shorkie = **0.695** (exact). The
"residual" was an artifact of the wrong subtree/aggregation, not a real model gap.

### 4. The "180 bp context" recipe (explains the D/E/F/G/H Shorkie residuals)
**What:** panels D–H titles read *"Aggregated across all genes, 180 bp"*, and the manuscript states
*"The position 180 bp upstream was selected for subsequent analyses."*
**Root cause:** the published Shorkie scores use **only the 180 bp insertion context** (file
`*_ctx8_*` / `*_context_8_*`), not the mean over all contexts. Averaging over all contexts shifts
the correlations (6D −0.016, 6E −0.024, 6F +0.008, 6H +0.040 vs published).
**Fix:** `mpra_common.SITE = 180`. Evidence (recheck diagnostic):

| Panel | all-context | **180 bp only** | published |
|---|---|---|---|
| 6D | 0.679 | **0.695** | 0.695 |
| 6E | 0.720 | **0.744** | 0.744 |
| 6F | 0.547 | **0.539** | 0.539 |
| 6G | 0.819 | **0.819** | 0.819 |
| 6H | 0.601 | **0.561** | 0.561 |

This single recipe correction resolves the prior 6H "+0.04 residual" (it was the all-context
average over the 12-gene motif-tiling set; at 180 bp the 12 genes give 0.561 exactly) and the 6D/6E
gaps simultaneously.

### 5. Panel I never numerically reproduced
**What:** the previous reproduction only checked "DREAM predictions present" and displayed a
pre-rendered PNG.
**Fix:** ported `MPRA_RNASeq/9_combined_density_subplot.py` — gaussian-kde 2D density of
log2(prediction+1) vs log2(Mean T0 RNA-Seq coverage+1). Shorkie (8 folds, valid+test;
`Shorkie_all_gene_eval_rc/f{0..7}c0/RNA-Seq/*/gene_{preds,targets}_stats.tsv`) gives r=0.895,
ρ=0.837; DREAM-RNN (180 bp upstream predictions, all splits;
`MPRA_RNASeq/predictions/upstream_180bp_predictions.tsv`) gives r=0.249, ρ=0.261 — both exact.

## Non-issues / notes
- **6B/6C** use the 18 reporter genes split into 3 expression quantiles (5–25 / 25–75 / 75–95, six
  genes each). Per-gene AUROC/AUPRC dip at the 100 bp site (min ≈0.76 for a single gene, matching
  the published per-gene dashed curves), but the three quantile aggregates stay > 0.95 — consistent
  with the manuscript's "AUROC and AUPRC > 0.95".
- **6H gene count:** the on-disk `motif_tiling_seqs` stranded tree has 12 scored genes (vs 22 for
  the other dual sets); the published 0.561 is reproduced exactly with those 12 at 180 bp, so this
  is not a discrepancy.
- **Axis orientation:** the published panels plot *model prediction on X, expression on Y* (the
  source scripts wrote them the other way); Pearson/Spearman are symmetric, so the printed numbers
  are identical. The reproduced panels follow the published orientation for visual fidelity.
- **No GPU / no fabrication:** every panel is recomputed from cached NPZ / TSV on disk. The
  prior "6F/6G/6H GPU gap" framing was incorrect — all scored predictions were already cached.
