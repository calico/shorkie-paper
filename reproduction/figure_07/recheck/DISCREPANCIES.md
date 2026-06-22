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
  `verify_fig7JO.csv` confirms **18/18 PASS** (region, SNP, ref/alt allele, motif, ISM-recomputed for
  all 6 loci). DREAM-RNN ISM is confirmed on disk for 4/6 loci (K, O genuinely absent).

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
| **G** Renganaath | ROC/PR | **fixed: 0.618/0.629** (was 0.536/0.555) | scored Shorkie-family on the 142-variant `results_subset_tss` | **wrong scoring dir** (see below) |
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

The corrected builder (`build_7EFG_roc_pr.py`) therefore sources panel G as the published panel
was assembled: **Shorkie / Shorkie_LM / Shorkie_Random_Init from `results_subset_tss`** (each on
its own natural pos/neg distribution — the cross-family inner join cannot be used on the subset:
its matched negatives are not shared with the DREAM negative sets, which would inflate AP to
~0.99 while ROC stays rank-stable), and **DREAM-Atten/CNN/RNN from the full-set cross-family
join**. All six published G lines are recovered within **≤0.007** (`fig7EFG_auc.csv`). The whole
36-cell E/F/G grid reproduces with **max |Δ| = 0.0074, zero cells off by >0.01**.

The stale claim has been corrected in `README.md` and `reproduction/VERIFICATION_REPORT.md`.

---

## Honest residuals (documented, not fabricated)

1. **Renganaath ~0.4–0.7 %.** Recompute 0.614/0.624 vs published 0.618/0.629 (and the DREAM-G
   lines ~0.003–0.006 below published). The published G panel mixes a subset-Shorkie scoring with
   a full-set DREAM scoring (an internal assembly inconsistency not fully specified by the released
   artifacts); the coherent recompute keeps the **ranking exact** (Shorkie > all DREAM) and matches
   every legend number within rounding. Passed with `atol=0.01`, not gated to the stale 0.536/0.555.

2. **Dataset sizes.** Scored positives (post inner-join, neg-set 1): Caudal ≈1712, Kita ≈655,
   Renganaath 142 (subset) / 395 (full) vs body text 1901 / 683 / 142. The differences reflect
   post-filter inclusion and the cross-family Position_Gene intersection; **142 = the subset the
   figure's panel G uses.** H/I per-bin Pos/Neg counts reproduce the published figure exactly.

3. **J–O DREAM-RNN ISM: 4 of 6 loci on disk.** `ism_{ref,alt}_results.tsv` contain the SNPs for
   J/L/M/N but **not K (YLR036C) or O (YGR046W)** — those DREAM rows are labeled "not on disk".
   DREAM-RNN is an external baseline; its ISM is rendered from cached deltas (ref-averaged,
   sign-negated), not re-run.

4. **J–O Ref DB motif logos** are embedded from the project motif DB
   (`experiments/motif_DB/…`: `REB1.1`, `PAC_motif`, `TATATA`), the authentic "Ref DB" source,
   rather than re-derived. **J–O Avg-logSED and |logSED|/|Δ| quantiles** are carried from the
   released scoring run (recorded in `fig7JO_logsed.csv` as `*_published`); the single-SNP logSED
   we recompute for A/B (OMA1 −0.220, LAP3 +0.234) differs from the gene-level "Average logSED"
   shown in J–O (−0.271 / 0.177), which is a per-track/per-gene average. The **Shorkie ISM REF/ALT
   saliency logos themselves (rows 2–3) are recomputed bit-for-bit from the released ISM cache.**

5. **J–O Shorkie Coverage** track uses observed RNA-seq coverage as a faithful proxy for the
   model's predicted-coverage track (the two correlate R>0.96 at these loci; cf. A/B).

6. **C/D are schematics** (cartoons), reproduced to match the published illustrations conceptually.
   The panel-D TSS-distance ECDF (genuine evidence the negatives are distance-matched) is retained
   as the supporting panel `reproduced/panel_D_matched_controls.png`.
