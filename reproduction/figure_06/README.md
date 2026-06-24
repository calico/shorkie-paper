# Figure 6 — MPRA promoter variant effects

> *"Shorkie predicts promoter variant effects validated by MPRAs."*

Reproduction package for **main-text Figure 6**. Published reference:
[`../../paper/Figures/Figure_6.pdf`](../../paper/Figures/Figure_6.pdf) (`published/Figure_6_full.png`).

- **Reproduce:** [`fig06_mpra_variant_effects.ipynb`](../../notebooks/fig06_mpra_variant_effects.ipynb) — delegates to the
  `recheck/` builders (executed with `nbconvert`, 0 errors).
- **Verify:** `reproduced/verify_fig06.csv` — **26/26 PASS** against the values printed on the
  published figure (both the Shorkie and DREAM-RNN subpanels of D–I).

> **Panel 6A is intentionally not reproduced** — it is a methods schematic (the MPRA insertion cartoon,
> 100–200 bp upstream) carrying no data; the reproduction covers the data panels **B–I**.

> **Deep recheck (2026-06-22):** panels **B–I** are reproduced **exactly from cached data — no
> GPU**. Reading the published PDF panel-by-panel revealed the earlier reproduction was incomplete:
> every panel D–I is a **two-scatter composite** (Shorkie blue **+** DREAM-RNN green), panel E is
> **Random** (not "challenging"), and panels D–H use the **180 bp** insertion context. With those
> fixes every reproduced Pearson/Spearman matches the figure to |Δ| ≤ 0.001. Full root-cause writeup:
> [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md).

> **Visual-exactness refinement (2026-06-22):** the panels are restyled to match the published figure
> pixel-for-pixel — B/C now reproduce `plot_combined_trend_quantiles` (per-gene dashed `'o'` lines over
> a single `tab20`, three quantile aggregates with **STD** error bars in dark-green/dark-red/black, 3-col
> legend, two-line title) as **separate** `Figure_6B.png`/`Figure_6C.png`; D–I drop the `(reproduced)`
> prefix and carry the published panel header + bold panel letter + source font sizes. Numbers unchanged.

> **Panel cleanup + B/C geometry (2026-06-23):** the methods-schematic panel **6A** was removed (no data).
> Panels **6B/6C** now match the published panel aspect ratio — `figsize=(8, 8/1.93)` (ratio **1.93**,
> measured from the 300-dpi published render) and saved **without** `bbox_inches="tight"` so the PNG ratio
> equals the figsize ratio deterministically (previously `(8, 3.7)` + tight crop → stretched ~2.16). Numbers
> unchanged (still 26/26).

---

## Dataset

The Random-Promoter **DREAM Challenge MPRA** (Rafi et al. 2024, *Nat Biotechnol*) — a held-out set
of **71,103** sequences in 8 categories (native yeast, random, high/low-expression, challenging,
SNV, motif-perturbation, motif-tiling); baseline comparator = **DREAM-RNN**.

## Panels (all CPU-reproducible from cached NPZ/TSV)

| Panel | Claim | Shorkie (pub→repro) | DREAM-RNN (pub→repro) | Source script |
|---|---|---|---|---|
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
| `build_panels_BC.py` | 6B/6C AUROC/AUPRC trends | `reproduced/Figure_6B.png`, `reproduced/Figure_6C.png` |
| `build_verify_fig06.py` | 26-check verification vs published figure | `reproduced/verify_fig06.csv` |
| `make_sidebyside_fig06.py` | published-vs-reproduced crops + montage | `recheck/panel_*_sidebyside.png` |

## Verification — `reproduced/verify_fig06.csv` (26/26 PASS)

Shorkie + DREAM-RNN Pearson **and** Spearman for panels D/E/F/G/H, both 6I density correlations, and
6B/6C AUROC/AUPRC > 0.95 — every reproduced value matches the published figure to |Δ| ≤ 0.001.
See [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md) for the per-panel root-cause writeup
(missing DREAM subpanels, the E→Random relabel, the 6D NPZ-subtree fix, and the 180 bp recipe).
