# Figure 5 — discrepancies, root causes, and fixes (deep recheck)

**Figure 5 = "Time-course stress-responsive TF induction (MSN2 & MSN4)."**
Published reference: `paper/Figures/Figure_5.pdf` → `published/Figure_5_full.png`.

Two genes / two halves, β-estradiol induction RNA-seq, timepoints T0,T5,T10,T15,T30,T45,T60,T90:
**5A–E = MSN2 @ ATG42 (YBR139W), chrII:515,214–515,714**;
**5F–J = MSN4 @ TSL1 (YML100W), chrXIII:70,173–70,673**.

Published panel layout (read off the PDF), and where the prior reproduction differed:

| Row content | MSN2 | MSN4 | Prior repro | Status |
|---|---|---|---|---|
| ISM logos (8 rows T0..T90, **full 500 bp** promoter) | **A** | **F** | 90 bp zoom; A was a surrogate | **fixed** |
| Fold-change bars (Measurement blue / Prediction orange, ±SEM) | **B** | **G** | correct | match |
| Pairwise Euclidean-distance heatmap (8×8, viridis) | **C** | **H** | C was a surrogate | **fixed** |
| **Normalized Pearson's R boxplot** (per timepoint, n=) | **D** | **I** | mislabeled **5E/5J** | **relabeled** |
| TF-Modisco binding-site motif progression | E | J | mislabeled 5D/5I | **skipped** (see §1, §4) |

---

## 1. Panels E/J intentionally skipped; boxplots relabeled to the published D/I

The prior notebook labeled the **normalized-Pearson-R boxplots** "5E/5J" and the **TF-Modisco motifs**
"5D/5I". The **published** figure is the opposite: **D/I are the boxplots**, **E/J are the motif
progression** (panel D sits top-right of the MSN2 block as a boxplot; panel E is the full-width motif row
below it; same for I/J on the MSN4 side). The boxplots are therefore relabeled to the published **D/I**
(filenames, titles, and the verify CSV).

**Panels E/J (the motif progression) are intentionally SKIPPED from this reproduction** (per user
direction). The substantive reason is in §4: the published MSN2 panel E uses an extended timepoint series
that is not in the released MSN2 artifacts, so it cannot be faithfully reproduced; panel J is skipped
alongside it. This reproduction covers the 8 panels **A–D, F–I**.

## 2. Panels A & C used a representative surrogate, not the real ATG42 locus  → FIXED (GPU recompute)

The MSN2 ISM target set (`gene_exp_motif_test_MSN2_targets`, 481 genes / 13 parts) **does not include
ATG42 / YBR139W** — searched all parts, no chrII:515 kb window exists on disk. The prior notebook
therefore drew 5A/5C from a *representative* MSN2 target (part0 idx0, chrI:1.46 Mb), clearly mislabeled.

**Fix:** we **recomputed the ATG42 promoter ISM on GPU** with the *same* driver and model the released
set used — `hound_ism_bed.py`, the fine-tuned `self_supervised_unet_small_bert_drop` fold **f0c0** model,
flags `-l 500 --rc --stats logSED`, the 3053-track RNA-seq sheet, the cleaned R64 FASTA — over a single
500 bp BED window (`panels/atg42.bed` = chrII:515,214–515,714, −450..+50 rel. YBR139W TSS @ 515,664, +
strand; built by `make_bed_seqs` centering so the scored region is exactly the published window). Output
`reproduced/ism_atg42/scores.h5` has the byte-compatible schema (logSED (1,500,4,3053) f16, seqs, chr,
start, end, strand) and is read unchanged by `load_locus`. Panels 5A (logos) and 5C (distance heatmap)
now show the **real published locus**. See `panels/run_atg42_ism.sbatch`.

*Note:* B and D for MSN2 were always real ATG42 — they come from the gene-level eval (`--gene YBR139W`),
which needs only the gene-level prediction TSVs, not the per-base ISM. Only the ISM-derived A/C were affected.

*Indexing nuance:* the pipeline's `1_create_target_genes.py` (TSS = gtf_start−1; −450..+50) yields
chrII:515,213–515,713 on the current GTF — a 1 bp 0-/1-indexing offset from the published caption
(515,214–515,714). We target the **published caption window** (515,214–515,714) so the panel label matches
the figure; the 1 bp shift is cosmetic (the STRE/TATA sites sit well inside the 500 bp window either way).

## 3. Panels A & F were zoomed to 90 bp, not the full 500 bp promoter  → FIXED

The prior `plot_time_logos(..., focus=90)` cropped to a 90 bp window around the T90 peak. The published
A/F show the **entire 500 bp promoter**. The recheck builder (`build_logos_distance.py`) renders the full
window, one row per timepoint, shared symmetric y-scale (so the induction-driven growth of the motif is
visible), A/C/G/T = green/blue/orange/red (matching the source `dna_letter_at` palette).

**Manual-overlay residual (documented, not reproduced):** the published A/F additionally carry a
**Reference-DB gene track**, **red dashed boxes** around the MSN2/MSN4 sites + start codons + TATA box,
**450 nt / 50 nt** window brackets, and feature text labels (YBR138C/ATG42; YML100W-A/TSL1; etc.). The
source script `3_timepoint_analysis/1_timepoint_viz_scores_h5_diff.py` draws **none** of these — it plots
only the logo letters + a y=0 baseline. Those annotations are **manual post-hoc (vector-editor) overlays**
added at figure assembly; they are not produced by the pipeline, so the reproduced A/F are the faithful
script-exact logo stacks without them (per the user's "faithful logo stack only" choice).

## 4. Panels E & J — intentionally skipped (not reproduced)

The published E/J are a curated **per-ΔT motif progression**: a YeTFaSCo reference motif on the left, then
one small logo per ΔT(Tn–T0) showing the TF-Modisco-detected MSN2/MSN4 **STRE** binding site emerging with
induction. These two panels are **skipped** from this reproduction, for one substantive reason:

- The **published MSN2 panel E** is labeled with the **extended series T5,T10,T20,T40,T70,T120,T180**. In the
  released code (`…/modisco_analysis/2_modisco_script_diff.sh`) that exact series is assigned to **SWI4**,
  while MSN2/MSN4 use the standard T0,5,10,15,30,45,60,90. The released MSN2 modisco directory contains only
  the standard 8 timepoints — no T20/T40/T70/T120/T180 — so the published MSN2 panel E (made from a
  longer/earlier MSN2 induction run out to 180 min, or carrying SWI4 labels) **cannot be faithfully
  reproduced** from the released artifacts.
- Panel **J** (the analogous MSN4 motif progression; its T5..T90 series *does* match the released data) is
  skipped **alongside** E for consistency.

For the record, the STRE motif *is* recoverable from the released ΔT modisco diffs — the GGGG (MSN2) /
CCCC (MSN4) binding site emerges across the available timepoints — but the panels are excluded from the
reproduction. (The exploratory progression builder lives only in the git history at commit `55b6173`.)

---

## What reproduces cleanly (kept / relabeled only)

- **B/G — fold-change bars.** Re-run of `…/motif_shorkie__time_series/1_time_track_metrics_viz.py`
  (`--gene YBR139W` / `YML100W`); Measurement-FC blue / Prediction-FC orange ±SEM. TSL1 rises to ~3.9
  (measured) / ~2.0 (predicted); matches published G. MSN2/ATG42 matches published B.
- **C/H — distance heatmaps.** viridis 8×8 Euclidean distance of per-timepoint mean-centered PWMs; TSL1
  diverges monotonically from T0. ATG42 now exact (recomputed).
- **D/I — normalized Pearson R boxplots.** Per-timepoint `pearsonr_norm`; medians **MSN2 0.591 / MSN4 0.618**
  (published band 0.55–0.65); per-timepoint n-counts **MSN2 8,12,8,12,9,9,7,9** and **MSN4 11,7,8,10,6,8,12,8**
  match the published boxplots exactly.
- **Global ΔlogFC R (supporting).** MSN2 **0.4949**, MSN4 **0.3992** (raw fold-change, all genes×timepoints,
  N=45,682) — distinct from the normalized per-track R of D/I; both reproduce.

## Naming-collision note (not part of Figure 5)

`notebooks/fig05_promoter_umap.ipynb` is a **different analysis** (Shorkie-LM promoter/feature embedding
UMAP/t-SNE); despite the `fig05` filename prefix it is **not** main-text Figure 5. Main-text Figure 5 is
the MSN2/MSN4 time-course reproduced here under `reproduction/figure_05/`.
