# Figure 6 — MPRA promoter variant effects

> *"Shorkie predicts promoter variant effects validated by MPRAs."*

Reproduction package for **main-text Figure 6**. Published reference:
[`../../paper/Figures/Figure_6.pdf`](../../paper/Figures/Figure_6.pdf) (`published/Figure_6_full.png`).

- **Reproduce:** [`reproduce_figure_06.ipynb`](reproduce_figure_06.ipynb) — delegates to the
  `recheck/` builders (executed with `nbconvert`, 0 errors).
- **Verify:** `reproduced/verify_fig06.csv` — **26/26 PASS** against the values printed on the
  published figure (both the Shorkie and DREAM-RNN subpanels of D–I).

> **Deep recheck (2026-06-22):** the whole figure is reproduced **exactly from cached data — no
> GPU**. Reading the published PDF panel-by-panel revealed the earlier reproduction was incomplete:
> every panel D–I is a **two-scatter composite** (Shorkie blue **+** DREAM-RNN green), panel E is
> **Random** (not "challenging"), and panels D–H use the **180 bp** insertion context. With those
> fixes every reproduced Pearson/Spearman matches the figure to |Δ| ≤ 0.001. Full root-cause writeup:
> [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md).

---

## Dataset

The Random-Promoter **DREAM Challenge MPRA** (Rafi et al. 2024, *Nat Biotechnol*) — a held-out set
of **71,103** sequences in 8 categories (native yeast, random, high/low-expression, challenging,
SNV, motif-perturbation, motif-tiling); baseline comparator = **DREAM-RNN**.

## Panels (all CPU-reproducible from cached NPZ/TSV)

| Panel | Claim | Shorkie (pub→repro) | DREAM-RNN (pub→repro) | Source script |
|---|---|---|---|---|
| **6A** | MPRA insertion schematic (100–200 bp, 10-bp steps) | schematic | — | programmatic |
| **6B** | AUROC, high- vs low-expr across insertion sites (3 quantiles) | >0.95 → **0.995** | — | `4_mpra_high_low_seq/5_MPRA_classifier_avg.py` |
| **6C** | AUPRC, same | >0.95 → **0.996** | — | same |
| **6D** | Yeast (single) | 0.695 → **0.695** | 0.891 → **0.891** | `5_mpra_viz/MPRA_scatter_regression_single.py` + `MPRA_ref_model_viz_single_index.py` |
| **6E** | **Random** (single) | 0.744 → **0.744** | 0.981 → **0.981** | same |
| **6F** | SNV (dual Alt−Ref) | 0.539 → **0.539** | 0.866 → **0.866** | `5_mpra_viz/MPRA_scatter_regression_dual_trim.py` + `MPRA_ref_model_viz_dual_indices.py` |
| **6G** | Motif Perturbation (dual) | 0.819 → **0.819** | 0.983 → **0.983** | same |
| **6H** | Motif Tiling (dual) | 0.561 → **0.561** | 0.943 → **0.943** | same |
| **6I** | Endogenous RNA-seq coverage density | r=0.895 → **0.895** | r=0.249 → **0.249** | `MPRA_RNASeq/9_combined_density_subplot.py` |

**Recipe (locked against the source scripts + on-disk data):** Shorkie scores come from the complete
stranded NPZ tree `…/MPRA_promoter_seqs/results/single_measurement_stranded/all_seq_types/`, using
the **180 bp** insertion context; per-sequence score = mean logSED over tracks (dual panels use
ALT−REF). DREAM-RNN per-sequence predictions come from
`data/random-promoter-dream-challenge-2022/data/DREAM-RNN_output.txt`. Ground-truth MAUDE expression
from `data/MPRA/filtered_test_data_with_MAUDE_expression.txt`, joined by each subset CSV's
`pos`/`alt_pos`/`ref_pos` columns (random/challenging/SNV/motif subsampled via `fix/*_sample_ids.tsv`).
Panel I is a gaussian-kde 2D density of log2(prediction+1) vs log2(Mean T0 RNA-Seq coverage+1).

## `recheck/` builders

| File | Builds | Output |
|---|---|---|
| `mpra_common.py` | shared loaders (GT, DREAM, NPZ aggregation, index maps; `SITE=180`) | — |
| `build_panels_DE.py` | 6D Yeast, 6E Random (Shorkie + DREAM) | `reproduced/Figure_6{D,E}.png`, `recheck/fig6_DEFGH_R.csv` |
| `build_panels_FGH.py` | 6F SNV, 6G motif-pert, 6H motif-tiling (Shorkie + DREAM) | `reproduced/Figure_6{F,G,H}.png` |
| `build_panel_I.py` | 6I RNA-seq coverage density (Shorkie + DREAM) | `reproduced/Figure_6I.png`, `recheck/fig6_I_R.csv` |
| `build_panels_BC.py` | 6B/6C AUROC/AUPRC trends + 6A schematic | `reproduced/Figure_6BC.png`, `reproduced/Figure_6A_schematic.png` |
| `build_verify_fig06.py` | 26-check verification vs published figure | `reproduced/verify_fig06.csv` |
| `make_sidebyside_fig06.py` | published-vs-reproduced crops + montage | `recheck/panel_*_sidebyside.png` |

## Verification — `reproduced/verify_fig06.csv` (26/26 PASS)

Shorkie + DREAM-RNN Pearson **and** Spearman for panels D/E/F/G/H, both 6I density correlations, and
6B/6C AUROC/AUPRC > 0.95 — every reproduced value matches the published figure to |Δ| ≤ 0.001.
See [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md) for the per-panel root-cause writeup
(missing DREAM subpanels, the E→Random relabel, the 6D NPZ-subtree fix, and the 180 bp recipe).
