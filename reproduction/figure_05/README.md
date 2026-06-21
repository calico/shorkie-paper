# Figure 5 — Time-course stress-responsive TF induction (MSN2 & MSN4)

> *"Time-course analysis of stress-responsive transcription factor induction."*

Reproduction package for **main-text Figure 5**. Published reference: [`../../paper/Figures/Figure_5.pdf`](../../paper/Figures/Figure_5.pdf) (`published/Figure_5_full.png`).

- **Reproduce:** [`reproduce_figure_05.ipynb`](reproduce_figure_05.ipynb) (executed in tmux, 0 errors).
- **Verify:** `reproduced/verify_fig05.csv` — **8/8 PASS**.

This figure was the largest gap (no prior topic notebook). **Everything is CPU-reproducible** — the ISM `scores.h5` were precomputed (GPU) in the original run; this notebook only reads them.

---

## Phase 1 — Discovery

Two genes / two halves (β-estradiol induction RNA-seq; timepoints T0,T5,T10,T15,T30,T45,T60,T90):
- **5A–E = MSN2 @ ATG42** (YBR139W), promoter window **chrII:515,214–515,714** (−450..+50 rel. TSS).
- **5F–J = MSN4 @ TSL1** (YML100W), promoter window **chrXIII:70,173–70,673** — panels analogous.

| Panel | Claim (from caption) | Source script | Computation |
|---|---|---|---|
| **5A/5F** | Shorkie ISM sequence logos, rows = successive timepoints | `…/3_timepoint_analysis/1_timepoint_viz_scores_h5_diff.py` | per-base logSED T0-averaged within each timepoint, mean-centered, projected on the reference base |
| **5B/5G** | experimental vs Shorkie-predicted fold-change at the locus, per timepoint | `motif_shorkie_time_series/1_time_track_metrics_viz.py` (`--gene YBR139W` / `YML100W`) | per-gene log2 fold-change vs T0 |
| **5C/5H** | pairwise Euclidean-distance heatmap of ISM logos | `…/3_timepoint_analysis/2_timepoint_viz_scores_h5_pairwise.py` | per-timepoint mean-centered PWM → pairwise Euclidean distance (8×8) |
| **5D/5I** | TF-MoDISco motifs from ΔT ISM matrices | `…/2_timepoint_analysis/modisco_analysis` | TF-MoDISco on T-vs-T0 ISM-difference contributions |
| **5E/5J** | **boxplot of normalized Pearson's R across all genes, per timepoint** | `1_time_track_metrics_viz.py` (`pearsonr_norm`) | per-track normalized, mean-centered Pearson R between measured & predicted profiles |

**Data (under `work_root`):** ISM `gene_exp_motif_test_{MSN2,MSN4}_targets/f0c0/part*/scores.h5` (logSED `(N,500,4,3053)` float16); gene-level TSVs `self_supervised_unet_small_bert_drop/gene_target_preds/f0c0/RNA-Seq/gene_{targets,preds}_norm.tsv`; targets sheet `cleaned_sheet_RNA-Seq.txt` (track→timepoint, `track_offset=1148`); ΔT TF-MoDISco `2_timepoint_analysis/modisco_analysis/results/.../{T0..T90}/modisco_results_10000_500_diff.h5`.

### Reproduction approach
- **5B/5E/5G/5J** — **re-ran the original `1_time_track_metrics_viz.py`** (CPU) for MSN2 (`--tf MSN2 --gene YBR139W`) and MSN4 (`--tf MSN4 --gene YML100W`); outputs under `reproduced/eval_MSN2/`, `reproduced/eval_MSN4/`. This is the faithful path (the script's order-sensitive T0-baseline / representative-track selection is non-trivial to re-derive).
- **5F/5H** — recomputed in-notebook from the **TSL1** window (`MSN4_targets/f0c0/part2 idx7` = chrXIII:70,173–70,673, the exact published locus): per-timepoint ISM logos + the 8×8 Euclidean-distance heatmap (track indices grouped by timepoint via `cleaned_sheet_RNA-Seq.txt`, `offset=1148`).
- **5D/5I** — rendered ΔT(T90−T0) TF-MoDISco motif logos from the precomputed `modisco_results_10000_500_diff.h5` (same IC-weighted logo helper as Figure 4H).
- **5A/5C** — illustrated with a representative MSN2-target promoter (see gap below).

---

## Phase 3 — Verification

**`reproduced/verify_fig05.csv`: 8/8 PASS.**

| Panel | Metric | Reported | Reproduced | Verdict |
|---|---|---|---|---|
| **5E** | normalized Pearson R median (MSN2, all genes) | 0.55–0.65 | **0.591** | PASS |
| **5J** | normalized Pearson R median (MSN4, all genes) | 0.55–0.65 | **0.618** | PASS |
| 5F | TSL1 window | chrXIII:70,173–70,673 | exact | PASS |
| 5H | TSL1 ISM distance diverges monotonically from T0 | yes | [0, .04, .09, .21, .36, .42, .49, .55] | PASS |
| 5B | MSN2 global ΔlogFC Pearson R | 0.4949 (on-disk ref) | 0.4949 | PASS |
| 5G | MSN4 global ΔlogFC Pearson R | 0.3992 (on-disk ref) | 0.3992 | PASS |
| 5D | MSN2 ΔT TF-MoDISco motifs | ≥1 | 20 | PASS |
| 5I | MSN4 ΔT TF-MoDISco motifs | ≥1 | 10 | PASS |

The **headline anchor** — the genome-wide normalized, mean-centered Pearson's R of **0.55–0.65** (caption panel E/J) — is reproduced exactly (MSN2 0.591, MSN4 0.618). The per-timepoint raw R rises with induction (MSN2 T5→T90: 0.40→0.63), and the TSL1 ISM distance heatmap shows monotone temporal divergence — both directional confirmations of induction-driven signal.

### Discrepancy log (honest)
| Item | Note |
|---|---|
| **ATG42 (MSN2 @ chrII:515,214–515,714) ISM not in released artifacts** | The per-locus ISM `scores.h5` for ATG42 is **absent** from the released `gene_exp_motif_test_MSN2_targets` set (30 chrII windows, none near 515 kb; no ATG42-specific `scores.h5` on disk). So panels **5A/5C** are shown with a **representative MSN2-target promoter** (clearly labelled), plus the genome-wide MSN2 ΔT TF-MoDISco (5D). The **MSN4 @ TSL1** ISM **is** present, so **5F/5H** are reproduced at the exact published locus. |
| 5B/5G label | The metric script labels its output by the example gene (`YBR139W_ATG42`), but `global_fc_metrics` / `global_lfc_scatter` / the R boxplots are computed **genome-wide** (all genes × timepoints, N=45,682 pairs) — not gene-specific. The per-gene fold-change panel (`fold_change_by_timepoint_bar.png`) is the locus-specific 5B/5G. |
| 0.4949/0.3992 vs 0.55–0.65 | Two distinct quantities: the **global ΔlogFC** Pearson R (0.49/0.40, raw fold-change) vs the **normalized mean-centered per-track** R (0.55–0.65, the manuscript anchor / panel E·J). Both reproduced. |

**Changes to legacy scripts:** none. `1_time_track_metrics_viz.py` was re-run unmodified (absolute `--out_dir` into `reproduced/`).
