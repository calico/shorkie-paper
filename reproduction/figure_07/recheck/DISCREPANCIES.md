# Figure 7 — deep-recheck discrepancy report

Figure 7 = **cis-eQTL variant-effect benchmark.** This pass re-verified every panel against
the **actual published PDF** (`paper/Figures/Figure_7.pdf`), panel-by-panel, rather than
against the intermediate reference PNGs the first reproduction used. Doing so surfaced one
**major numeric error** (panel G), several **plot-type / layout / styling gaps**, and
**4 missing + 2 mislabeled ISM panels** — all now corrected.

**Verdict: Figure 7 reproduced.** `recheck/recheck_checks_fig07.csv` = **66/66 PASS** against
the published targets. E/F reproduce to 3 decimals; G now reproduces (was backwards); A–D and
H–O match the published plot types, layouts, colors, and labels; J–O Shorkie ISM logos are
recomputed bit-for-bit from the released ISM cache. Genuine residuals are documented below.

Recheck artifacts (this directory): `build_7EFG_roc_pr.py`, `build_7AB_coverage.py`,
`build_7CD_schematic.py`, `build_7HI_distance.py`, `build_7JO_ism.py`, `build_verify_fig07.py`,
`make_sidebyside_fig07.py`; CSVs `fig7EFG_auc.csv`, `fig7AB_logsed.csv`, `fig7HI_auprc.csv`,
`fig7JO_logsed.csv`, `verify_fig7JO.csv`, `recheck_checks_fig07.csv`; side-by-sides
`panel_{AB,C,D,EFG,HI,JO}_sidebyside.png`, `Figure_7_published_vs_reproduced.png`.

## Refinement pass (match-published)
A second pass tightened three panel groups to the published rendering exactly:
- **A/B coverage** now use the published **bar style** (`ax.bar(width=1, alpha=0.7)` — Ref `tab:blue`,
  Alt `tab:orange`; GT green) with a single consolidated right-side legend (Center / gene Start / End /
  SNP / Ref / Alt / Ground Truth), matching `plot_coverage_track_pair_bins_w_ref` in the source
  notebook. The cached track-mean `cov_ref/cov_alt` already equal the published values (soft-clip 384 is
  a no-op at these loci; y-scales match: A Signal 0–10, GT 0–12).
- **E/F/G** now set the **exact published axis limits** — PR `xlim(0,1)`/`ylim(0.45,1.05)`, ROC
  `0–1×0–1`, every subplot square (`set_box_aspect(1)`), 0.2/0.1 tick spacing — matching
  `1_roc_pr_shorkie_fold.py`. Numbers unchanged (max |Δ|=0.0074).
- **J–O Shorkie ISM logos** now use the **exact source recipe** — `plot_seq_scores` semantics
  (per-position argmax-|·| base scaled by the row-sum) on the **raw** cached `pred_ism_wt/mut` over the
  80 bp window (`center±40`); the prior `(pred_ism − rowmean)×onehot` was ~0.75× scaled. A new
  `verify_fig7JO.csv` confirms PASS (region, SNP, ref/alt allele, motif, ISM-recomputed for all 6 loci).
  DREAM-RNN ISM was confirmed on disk for 4/6 loci here (K, O appeared absent) — **corrected in pass 3:
  all 6/6, K/O sourced from the targeted `_additional` run.**

## Refinement pass 2 (height + per-subplot limits + DREAM alignment)
- **A/B "not scaled" (height/aspect).** Reading the published A/B at 300+ dpi: each subplot **auto-scales
  to its own data** (A Signal 0–~10.5 from Ref/Alt, GT 0–~13 from observed — *independent*, not shared),
  and the panels are **wide-and-short** (helper `figsize≈(16,4)` for the two coverage subplots). The prior
  build was too tall (`figsize=(12.5,5.4)`), which vertically stretched the coverage ("scaled"). Fixed:
  `figsize=(16,4.8)` with thin gene track (≈9:1 coverage aspect, matching published); Signal auto-scales
  to `max(cov_ref,cov_alt)` with `vlines(x,0,max_sig)` + SNP star at `0.075*max_sig`; GT auto-scales to its
  own observed max — independent scales, as published.
- **E/F/G per-subplot check.** Cropped all six published subplots (300+ dpi) and confirmed each uses
  identical limits — **PR x 0–1, y 0.45–1.05; ROC x 0–1, y 0–1** (no per-subplot deviation), matching the
  uniform limits `build_7EFG_roc_pr.py` already enforces.
- **J–O DREAM-RNN ISM shift — fixed.** The DREAM MPRA ISM window is **110 bp** with `left_pad=17`,
  `right_pad=13` (`2_plot_DNA_logo.py`), so the 80 bp core is `pos[17:97)` and the **SNP is at `pos=57`**
  (verified: `orig_base@57` == the ref allele for all loci; M shows `GTTACCC`). The prior build centered
  DREAM on `(min+max)//2 = 54` and cropped ±40 → a ~3 bp shift. Fixed by porting the exact recipe
  (delta-matrix → negate → global mean-normalize → ref-base-average) and cropping the `[17:97)` core
  (SNP at index 40), and by setting the Shorkie window to `[ci-40:ci+40)` (80 bp, SNP at index 40) so the
  two are the same width and SNP-aligned; a light-blue SNP highlight is drawn at the column. After the
  fix, Shorkie and DREAM motif peaks coincide (J 39/40, L 38/36, M 44/41, N 34/35 — within model noise,
  no systematic offset). K & O still had no DREAM ISM on disk *here* (resolved in pass 3).

## Refinement pass 3 (E/F/G exact limits + J–O: K/O DREAM, N rev-comp, gene-windowed coverage)
A third pass (this session) matched the remaining E/F/G axis scaling and completed J–O:
- **E/F/G exact limits/scale.** Re-reading `1_roc_pr_shorkie_fold.py::plot_ensemble_roc_pr`: PR sets
  **only** `plt.ylim(0.45,1.05)` and leaves x to autoscale; ROC sets **neither** axis. With matplotlib's
  default 5 % data margin the published limits are therefore **PR x≈(−0.05,1.05) y(0.45,1.05)** and
  **ROC x≈(−0.05,1.05) y≈(−0.05,1.05)** — the curves sit *inset* from the frame (the 0.0/1.0 ticks are
  not on the spines). The prior build forced `xlim(0,1)` / ROC `(0,1)×(0,1)` + `set_box_aspect(1)`, which
  is exactly why the scale read as "off". Fixed: drop the hard limits (autoscale + `margins(0.05)`), drop
  `box_aspect`, and use a `(16.5,10)` grid so each cell is `5.5×5` (the source per-subplot figsize). AUCs
  unchanged (max |Δ|=0.0074). *(This supersedes the pass-2 note that read the limits as a hard
  `0–1`/`0.45–1.05` square — that was the prior build's forcing, not the published autoscale.)*
- **J–O K & O DREAM-RNN ISM — now on disk (6/6).** The main `eQTL_MPRA_models_ISM/results` TSV lacks
  K (YLR036C) and O (YGR046W), but the targeted **`eQTL_MPRA_models_ISM_additional/results`** run holds
  all six loci — and is byte-identical to the main run where they overlap (panel N: corr 1.0, max|Δ|=0).
  Both use the same 110 bp / SNP-at-`pos=57` convention, so `dream_logos` simply falls back to `_additional`
  for K/O. All six panels now render DREAM-RNN ISM (REF+ALT); `verify_fig7JO.csv` = **24/24** (adds a
  "DREAM-RNN ISM present" check per locus).
- **Panel N reference motif — reverse-complemented.** N's Reb1.1 site is on the (−) genomic strand, so the
  published N Ref-DB logo is **CGGGTAA** (rev-comp of the M/O `TTACCCG`), matching N's own data-derived
  Shorkie ISM (which already reads CGGGTAA). The Ref-DB row now embeds `viz_self_motif_db/REB1.1_rc.png`
  for N (forward `REB1.1.png` retained for M/O).
- **J–O coverage centred on the gene.** Replaced the SNP-centred ±1024 bp track with the published zoomed
  renderer's window — `[min(SNP,gene_start)−100, max(SNP,gene_end)+100]` snapped to the 16 bp model-bin grid
  (`plot_coverage_track_pair_bins_w_ref_zoomed`) — with SNP / Variant / gene-start (green) / gene-end (red) /
  ±40 bp-ISM-region (grey) markers and a `chrom:region_start-region_end bp` xlabel. R64 gene spans
  (0-based start = GTF_start−1) reproduce the published ranges exactly (M `chrXI:603094-604456`,
  N `chrXIV:200228-202033`, O `chrVII:584583-586152`, …). Coverage signal here was still the observed
  RNA-seq proxy (predicted Ref/Alt needs the GPU ensemble) — replaced with predicted coverage in pass 4.

## Refinement pass 4 (ISM padding bands + GPU-predicted Ref/Alt coverage + coverage y-axis)
A fourth pass (this session) matched the remaining J-O details to the published panels:
- **ISM padding bands (both sides).** The published ISM logos show the 80 bp ISM core flanked by grey
  padding on both sides (the source standalone renderers: Shorkie `2_viz_ism_dna_logo.py` uses 18 bp
  left / 14 bp right -> 112 bp; DREAM `2_plot_DNA_logo.py` uses 17/13 -> 110 bp). `plot_logo` now shows
  those full windows with the padding columns grey-shaded and the SNP column light-blue, for all four
  ISM rows (Shorkie REF/ALT, DREAM REF/ALT) of every panel.
- **GPU-predicted Ref/Alt coverage + matching y-axis.** The published coverage overlays the model's
  PREDICTED Ref (blue) + Alt (orange) RNA-Seq(T0) coverage. Those arrays were cached only for M/N
  (OMA1/LAP3); the observed RNA-seq is on a different, locus-specific scale (~11-17x smaller; M ~17x,
  N ~11x) so it could not match the published y-axis. Re-ran the 8-fold Shorkie ensemble for all six
  loci (`panels/run_cov_eqtl_jo.py` -> `reproduced/ism/cov_<panel>.npz`) using the renderer's simple
  SNP-centred window (`start = pos-8192`, off=1024, stride=16) so the predicted output bins align 1:1
  with the published bin grid. The coverage track now draws genuine overlapping Ref/Alt bars at the
  predicted scale; the **y-axis limits match the published** (e.g. M 0-7.5, N 0-27, O 0-6) and the
  x-axis (bin window) already matched. `coverage_track` falls back to observed only if a predicted npz
  is absent. `verify_fig7JO.csv` = **24/24** (ref/alt/ISM-recomputed + DREAM-present per locus).

## Refinement pass 5 (panel G curve-shape fix — uniform 6-way subset join)
The user reported the panel-G (Renganaath) PR/ROC **curves were shifted for all six models** vs the
published, even though the legend AUCs were close (max |Δ| = 0.0074). Root cause: the pass-3 build used a
**hybrid** sourcing for G — Shorkie family from the 142-variant `viz_new/results_subset_tss/`
**per-model-own** (no join), DREAM from the **full 395-variant** `…/eQTL_MPRA_models_eval_Renganaath_etal/
results/` joined with the full Shorkie set. Each model's curve was therefore built on a **different variant
set** → the shifts (and the residual AUC error).

The source plotter `1_roc_pr_shorkie_fold.py` does **no such hybrid**: `get_mpra_base(subset="subset_tss")`
loads DREAM from a **subset-specific** dir `…/eQTL_MPRA_models_eval_Renganaath_etal/**results_subset_tss/**`
(which the prior reproduction never used), and `load_combined` does a **uniform 6-way inner-join on
`Position_Gene`** so all six models share the *same* 142-variant subset. The build now mirrors this exactly:
`_dreamdir(Renganaath)` → the subset DREAM dir; `model_valid` → one `joined_all(exp, "results_subset_tss"
if Renganaath else "results", neg)` path for all three datasets. The inner-join is clean (≈149 pos / 149
neg per negset — the earlier "AP inflates to ~0.99" worry came only from joining the subset Shorkie with the
*full* DREAM dir). Result: every published G line reproduces to **±0.000–0.001**, the whole 36-cell E/F/G
grid is **max |Δ| = 0.0005** (was 0.0074), and the curves overlay the published with no per-model shift.

---

## Per-panel findings

| Panel | Type | Match to published | Fix applied | Root cause of prior gap |
|---|---|---|---|---|
| **A** OMA1 | coverage | structure exact | added GT track + neighbor gene + orange Alt + markers | prior = single line plot, no ground truth |
| **B** LAP3 | coverage | structure exact | same | same |
| **C** | schematic | conceptual match | refined logSED cartoon (bins, formula, SUM call-outs) | acceptable schematic |
| **D** | schematic | **plot type fixed** | rendered the pos/neg-eQTL cartoon | prior showed an ECDF (wrong plot); kept ECDF as support |
| **E** Caudal | ROC/PR | **exact (Δ≤0.001)** | transposed to published grid (cols=datasets, rows=PR/ROC) | layout transposed |
| **F** Kita | ROC/PR | **exact (Δ≤0.001)** | same | same |
| **G** Renganaath | ROC/PR | **exact: 0.618/0.629** (was 0.536/0.555) | uniform 6-way inner-join on the 142-variant `results_subset_tss` (incl. subset DREAM dir); Δ≈0.000 | **wrong scoring dir + hybrid sourcing** (see below) |
| **H** Caudal | AUPRC/dist | **exact incl. per-bin counts** | AUPRC-only, 4 models, Pos/Neg x-labels | prior was a 2×2 AUROC+AUPRC grid |
| **I** Kita | AUPRC/dist | **exact incl. per-bin counts** | same | same |
| **J–O** | ISM logos | structure + Shorkie ISM exact | **6 correctly-labeled** logo stacks from cache | prior had 2 bar charts mislabeled "J-K"/"N-O" |

---

## The major finding: panel G (Renganaath) was backwards

The previous reproduction reported **Shorkie ROC 0.536 / PR 0.555** on Renganaath and
documented a "**text↔figure inconsistency**: DREAM beats Shorkie on Renganaath, the body text
overstates." **That conclusion is wrong** — it disagrees with the published Figure 7G itself.

Reading `paper/Figures/Figure_7.pdf`, panel **G plots Shorkie ROC 0.618±0.006 / PR 0.629±0.013**
— Shorkie is the **top** model, beating every DREAM baseline (≈0.59), exactly as the body text
says ("Shorkie achieved superior ROC and PR metrics for … Renganaath et al.").

**Root cause.** The source plotter `1_roc_pr_shorkie_fold.py` reads Shorkie scores from
`viz_new/results/` for *all* experiments. For Renganaath that directory holds the **full
395-variant** set (→ 0.536/0.555). The published panel was instead made from
`viz_new/results_subset_tss/` — the **142-variant** TSS-proximal subset that matches the
body-text "142 causal core-promoter variants." Recomputing Shorkie on that subset gives
**ROC 0.614 / PR 0.624** (within 0.4–0.5 % of the published 0.618/0.629). Only Renganaath has
subset dirs; Caudal/Kita correctly use the full `results/` set (and reproduce to 3 decimals).

The corrected builder (`build_7EFG_roc_pr.py`) sources panel G exactly as the source plotter
`1_roc_pr_shorkie_fold.py` does (refinement pass 5): a **uniform 6-way inner-join on the
142-variant `results_subset_tss` set for ALL six models** — Shorkie family from
`viz_new/results_subset_tss/` and **DREAM-Atten/CNN/RNN from the matching subset DREAM dir**
`eQTL_MPRA_models_eval_Renganaath_etal/results_subset_tss/` (`get_mpra_base(subset="subset_tss")`).
Because every model shares the same subset, the join is clean (≈149 pos / 149 neg per negset) and
every published G line reproduces to **±0.000–0.001** (`fig7EFG_auc.csv`). The whole 36-cell E/F/G
grid reproduces with **max |Δ| = 0.0005, zero cells off by >0.01**.

*(An interim pass-3 build used a hybrid — subset Shorkie per-model-own + DREAM from the FULL
395-variant set — believing the subset's matched negatives weren't shared with the DREAM negsets.
That belief was wrong: the subset-specific DREAM dir exists and its negatives DO match, so the
uniform join is correct and exact. The hybrid scored each model on a different variant set, which
is what shifted the G curves.)*

The stale claim has been corrected in `README.md` and `reproduction/VERIFICATION_REPORT.md`.

---

## Honest residuals (documented, not fabricated)

1. **Renganaath — now exact (≤0.001), no residual.** (Resolved in refinement pass 5.) Panel G
   reproduces to **±0.000–0.001** for all six models (Shorkie 0.618/0.629, the top model; DREAM
   0.585–0.596) via the source's uniform 6-way inner-join on the 142-variant `results_subset_tss`
   set — including DREAM from the subset-specific DREAM dir. The earlier "~0.4–0.7 % / subset-Shorkie
   + full-DREAM assembly inconsistency" note was an artifact of the prior hybrid sourcing and no
   longer applies: the published panel uses a single coherent subset for every model, and we match it.

2. **Dataset sizes.** Scored positives (post inner-join, neg-set 1): Caudal ≈1712, Kita ≈655,
   Renganaath 142 (subset) / 395 (full) vs body text 1901 / 683 / 142. The differences reflect
   post-filter inclusion and the cross-family Position_Gene intersection; **142 = the subset the
   figure's panel G uses.** H/I per-bin Pos/Neg counts reproduce the published figure exactly.

3. **J–O DREAM-RNN ISM: all 6 of 6 loci on disk** (resolved in refinement pass 3). J/L/M/N come from
   `eQTL_MPRA_models_ISM/results`; K (YLR036C) and O (YGR046W) — absent there — come from the targeted
   `eQTL_MPRA_models_ISM_additional/results` run (byte-identical to the main run where they overlap).
   DREAM-RNN is an external baseline; its ISM is rendered from cached deltas (ref-averaged, sign-negated),
   not re-run.

4. **J–O Ref DB motif logos** are embedded from the project motif DB
   (`experiments/motif_DB/…`: `REB1.1`, `PAC_motif`, `TATATA`), the authentic "Ref DB" source,
   rather than re-derived. **J–O Avg-logSED and |logSED|/|Δ| quantiles** are carried from the
   released scoring run (recorded in `fig7JO_logsed.csv` as `*_published`); the single-SNP logSED
   we recompute for A/B (OMA1 −0.220, LAP3 +0.234) differs from the gene-level "Average logSED"
   shown in J–O (−0.271 / 0.177), which is a per-track/per-gene average. The **Shorkie ISM REF/ALT
   saliency logos themselves (rows 2–3) are recomputed bit-for-bit from the released ISM cache.**

5. **J–O Shorkie Coverage** is the 8-fold ensemble's PREDICTED Ref/Alt RNA-Seq(T0) coverage
   (refinement pass 4; `panels/run_cov_eqtl_jo.py` -> `reproduced/ism/cov_<panel>.npz`), gene-windowed
   and drawn in the published overlapping Ref (blue) + Alt (orange) style with **matching x and y axis
   limits**. (Earlier passes used the observed RNA-seq proxy, which sits on a different, locus-specific
   scale ~11-17x smaller and so could not match the published predicted y-axis.)

6. **C/D are schematics** (cartoons), reproduced to match the published illustrations conceptually.
   The panel-D TSS-distance ECDF (genuine evidence the negatives are distance-matched) is retained
   as the supporting panel `reproduced/panel_D_matched_controls.png`.
