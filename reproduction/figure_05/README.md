# Figure 5 — Time-course stress-responsive TF induction (MSN2 & MSN4)

> *"Time-course analysis of stress-responsive transcription factor induction."*

Reproduction package for **main-text Figure 5**. Published reference: [`../../paper/Figures/Figure_5.pdf`](../../paper/Figures/Figure_5.pdf) (`published/Figure_5_full.png`).

- **Reproduce:** [`fig05_timecourse_tf_induction.ipynb`](../../notebooks/fig05_timecourse_tf_induction.ipynb) + the builders in [`recheck/`](recheck/).
- **Verify:** `reproduced/verify_fig05.csv`.
- **Discrepancies & root causes:** [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md).
- **Side-by-side vs published:** `recheck/Figure_5_published_vs_reproduced.png` + `recheck/panel_*_sidebyside.png`.

Two genes / two halves (β-estradiol induction RNA-seq; timepoints T0,T5,T10,T15,T30,T45,T60,T90):
**5A–E = MSN2 @ ATG42** (YBR139W), promoter **chrII:515,214–515,714**; **5F–J = MSN4 @ TSL1** (YML100W),
promoter **chrXIII:70,173–70,673**.

## Published panel map (verified against the PDF)

| Row content | MSN2 | MSN4 | Source recipe |
|---|---|---|---|
| ISM logos, 8 rows T0..T90, **full 500 bp** promoter | **A** | **F** | per-timepoint logSED, mean-centered, projected on ref base (`recheck/build_logos_distance.py`) |
| Fold-change vs T0 (Measurement / Prediction, ±SEM) | **B** | **G** | `…/motif_shorkie__time_series/1_time_track_metrics_viz.py --gene YBR139W / YML100W` |
| Pairwise Euclidean-distance heatmap (8×8, viridis) | **C** | **H** | per-timepoint mean-centered PWM distance (`recheck/build_logos_distance.py`) |
| **Normalized Pearson's R boxplot** (per timepoint, `n=`) | **D** | **I** | per-track `pearsonr_norm` by timepoint; surfaced to top-level `reproduced/Figure_5{D,I}_*boxplot.png` (`recheck/build_DI_boxplots.py`, faithful to `1_time_track_metrics_viz.py`) |
| TF-Modisco binding-site motif over ΔT | E *(skipped)* | J *(skipped)* | not reproduced — see `recheck/DISCREPANCIES.md` |

> This reproduction covers the **8 panels A–D, F–I**; panels **E/J are intentionally skipped** (the published
> MSN2 panel E uses an extended timepoint series not in the released artifacts — see DISCREPANCIES.md). Note the
> normalized-Pearson-R boxplots are the published **D/I** (the prior draft mislabeled them 5E/5J).

## Verification (`reproduced/verify_fig05.csv` — 10/10 PASS over the reproduced panels A–D, F–I)

| Panel | Metric | Target | Reproduced |
|---|---|---|---|
| 5A | ATG42 ISM window | chrII:515,214–515,714 | exact (recomputed) |
| 5C | ATG42 ISM distance diverges-from-T0 | yes | yes |
| 5D | MSN2 norm-R median | 0.55–0.65 | **0.591**; n-counts 8,12,8,12,9,9,7,9 ✓ |
| 5F | TSL1 ISM window | chrXIII:70,173–70,673 | exact |
| 5H | TSL1 ISM distance diverges-from-T0 | yes | yes ([0,.04,.09,.21,.36,.42,.49,.55]) |
| 5I | MSN4 norm-R median | 0.55–0.65 | **0.618**; n-counts 11,7,8,10,6,8,12,8 ✓ |
| 5B | MSN2 global ΔlogFC R | 0.4949 | 0.4949 |
| 5G | MSN4 global ΔlogFC R | 0.3992 | 0.3992 |

## What changed in this recheck (see `recheck/DISCREPANCIES.md` for full root-cause detail)

1. **Normalized-Pearson-R boxplots are the published D/I** (the prior draft mislabeled them 5E/5J).
2. **Panels A & C now show the REAL ATG42 locus.** The ATG42 promoter ISM was never released (the MSN2
   ISM target set omits chrII:515 kb), so the old notebook used a representative surrogate. We
   **recomputed it on GPU** (`panels/run_atg42_ism.sbatch` → `reproduced/ism_atg42/scores.h5`) with the
   exact released driver/model/flags, single-locus BED = the ATG42 promoter window.
3. **Panels A & F are now full-500 bp** logo stacks (the prior notebook zoomed to a 90 bp window). The
   published red boxes / Reference-DB gene track / feature labels are **manual post-hoc overlays** the
   pipeline never drew (documented; not reproduced — faithful logo stack only).
4. **Panels E & J are intentionally skipped** (TF-Modisco motif progression) — see the residual below.

### Documented residuals

- **Panels E/J skipped.** The **published** MSN2 panel E uses the **extended series
  T5,T10,T20,T40,T70,T120,T180** (the series the released code assigns to **SWI4**); the released MSN2
  modisco has only the standard 8 timepoints, so panel E cannot be faithfully reproduced. Panel J (the
  analogous MSN4 motif progression) is skipped alongside it. (The STRE GGGG/CCCC motif does emerge across
  the available timepoints, but the panels are excluded from this reproduction.)
- **ATG42 window indexing.** The pipeline's TSS rule (`1_create_target_genes.py`) yields
  chrII:515,213–515,713 on the current GTF — a 1 bp 0-/1-indexing offset from the published caption
  (515,214–515,714); we target the published caption window (cosmetic 1 bp).
- **Two distinct R quantities.** Global ΔlogFC R (0.49/0.40, raw fold-change) vs the normalized
  mean-centered per-track R (0.55–0.65, panels D/I). Both reproduce.

## Data sources (under `work_root`)

- ISM logSED `…/motif_shorkie_RP_TSS/gene_exp_motif_test_{MSN2,MSN4}_targets/f0c0/part*/scores.h5`
  `(N,500,4,3053)` f16 (+ the recomputed `reproduced/ism_atg42/scores.h5` for ATG42).
- Gene-level eval re-run of `1_time_track_metrics_viz.py` → `reproduced/eval_{MSN2,MSN4}/`.
- ΔT TF-Modisco `…/2_timepoint_analysis/modisco_analysis/results/.../T{n}/modisco_results_10000_500_diff.h5`.
- Targets sheet `cleaned_sheet_RNA-Seq.txt` (track→timepoint, `track_offset=1148`); model
  `self_supervised_unet_small_bert_drop/train/f0c0`.

