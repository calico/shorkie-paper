# Figure 7 — Shorkie accurately predicts cis-eQTL variant effects

> *"Shorkie accurately predicts cis-eQTL variant effects."*

Reproduction package for **main-text Figure 7**. Published reference: [`../../paper/Figures/Figure_7.pdf`](../../paper/Figures/Figure_7.pdf) (`published/Figure_7_full.png`).

- **Reproduce:** [`reproduce_figure_07.ipynb`](reproduce_figure_07.ipynb) (executed, 0 errors).
- **Verify:** `reproduced/verify_fig07.csv` — **66/66 PASS** against the **published PDF** (`paper/Figures/Figure_7.pdf`).
- **Deep-recheck:** `recheck/` — every panel rebuilt to match the published figure panel-for-panel; see [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md) and the side-by-sides `recheck/panel_{AB,C,D,EFG,HI,JO}_sidebyside.png`.
- **Match-published refinement:** A/B use the published **bar** coverage style + GT track + unified legend, at the published **wide-short aspect** with each subplot **auto-scaled to its own data** (independent Signal/GT scales, as published — so the coverage is not vertically stretched); **E/F/G match the source's exact scaling** — `1_roc_pr_shorkie_fold.py` fixes **only** PR `ylim(0.25,1.05)` (lowest tick 0.3; uniform for E/F/G) and sets **no** ROC limits, so the other axes autoscale with matplotlib's default 5 % margin (PR x and ROC x,y ≈ −0.05–1.05, curves *inset* from the frame), **no `box_aspect`**, each subplot at the source `5.5×5` aspect; **J–O** Shorkie ISM logos use the **exact `plot_seq_scores` recipe** on the raw cached ISM with SNP-aligned DREAM-RNN ISM (110 bp, pads 17/13, SNP `pos 57`, core `[17:97)`), and three completions: **all 6/6 DREAM-RNN ISM panels now on disk** (K/O sourced from the targeted `_additional` run), **panel N's Ref-DB motif reverse-complemented** (CGGGTAA — its (−)-strand Reb1.1), and the **coverage centred on the gene** (`[min(SNP,gene_start)−100, max(SNP,gene_end)+100]`, published markers/range/xlabel). `recheck/verify_fig7JO.csv` = **24/24** (region/SNP/ref/alt/motif + DREAM-present for all 6 loci).
- **Match-published refinement (J–O, pass 4):** the ISM logos now show the published **grey padding bands on both sides** (Shorkie 112 bp = 80 bp core + 18/14 pad; DREAM 110 bp = 80 + 17/13), and the coverage track is the **8-fold ensemble PREDICTED Ref/Alt** RNA-Seq(T0) coverage (`panels/run_cov_eqtl_jo.py` → `reproduced/ism/cov_<panel>.npz`) drawn as overlapping **Ref (blue) + Alt (orange)** bars — so both the **x and y axis limits match the published predicted scale** (e.g. M 0–7.5, N 0–27, O 0–6). Observed RNA-seq (a different, ~11–17× smaller scale) is only a fallback.
- **GPU panels:** `panels/run_ism_eqtl.{py,sbatch}` (8-fold ensemble logSED-ISM for the OMA1/LAP3 eQTL SNPs) and `panels/run_cov_eqtl_jo.{py,sbatch}` (8-fold ensemble predicted Ref/Alt coverage for the six J–O loci). J–O Shorkie ISM logos are rebuilt from the released ISM cache (no GPU rerun).

The headline result: ensemble **AUROC/AUPRC matches the published Figure 7E/F/G** (Caudal/Kita to 3 decimals; Renganaath now to ±0.000–0.001, see below), the GPU ISM reproduces the released per-SNP logSED **bit-exactly** (OMA1 −0.2198, LAP3 +0.2344), and the J–O Shorkie ISM saliency logos are recomputed bit-for-bit from cache.

> **Deep-recheck correction (panel G).** The earlier reproduction matched intermediate
> reference PNGs and reported **Shorkie 0.536/0.555 on Renganaath (panel G)**, concluding "DREAM
> beats Shorkie." Reading the **published PDF** shows panel G plots **Shorkie 0.618/0.629** — the
> top model — consistent with the body text. Root cause: panel G is scored on the **142-variant**
> `results_subset_tss/` set (= the "142 causal core-promoter variants"), not the full 395-variant
> `results/` set. Final fix (pass 5) follows the source plotter exactly — a **uniform 6-way
> inner-join on `results_subset_tss` for all six models**, with DREAM from the matching subset DREAM
> dir `eQTL_MPRA_models_eval_Renganaath_etal/results_subset_tss/` — so all six G lines reproduce to
> **±0.000–0.001** and the curves overlay the published with no per-model shift. Details in
> `recheck/DISCREPANCIES.md`.

---

## Phase 1 — Discovery

**Caption.** A = positive eQTL, reduced expr. with alt allele at **OMA1 (chrXI:603,195–604,232)**; B = positive eQTL, increased expr. with alt allele at **LAP3 (chrXIV:200,569–201,933)**; C = logSED computation schematic; D = negative-control generation (matched by ref/alt allele, TSS distance, MAF ≥ 5% from ~1,000 yeast isolates); E–G = PR & ROC comparing **Shorkie / Shorkie_Random_Init / Shorkie_LM / DREAM (Atten/CNN/RNN)** for **Caudal et al. (E)**, **Kita et al. (F)**, **Renganaath et al. (G)**; H,I = AUPRC/AUROC by TSS-distance bins, **Caudal (H)** & **Kita (I)**; J–O = ISM maps centered on eQTL SNPs (Shorkie SNP light-blue, DREAM-RNN, adaptor gray).

**Datasets (body text):** Caudal et al. **1,901** local cis-eQTLs (~1,000 isolates); Kita et al. 2017 **683** eQTLs (Promoter/UTR5/UTR3/ORF); Renganaath et al. 2020 **142** causal core-promoter variants. **Metric:** `|logSED|` (Shorkie/Random_Init use `logSED_agg`, the LM uses `LLR`, DREAM uses `logSED`); positives vs **4 matched negative sets**, ensemble curves = per-set ROC/PR → **mean ± SEM**.

| Panel | Claim | Source script | Config key | Tier |
|---|---|---|---|---|
| **A** | OMA1 locus, alt reduces expr. | `2_variant_scoring/score_variants_shorkie.py` (+ ensemble ISM) | `models.shorkie_finetuned` | GPU ✅ |
| **B** | LAP3 locus, alt increases expr. | same | same | GPU ✅ |
| **C** | logSED schematic | — (programmatic) | — | schematic ✅ |
| **D** | matched negative controls | `0_data_generation/1_generate_negs.py` | `results.eqtl_scores` | CPU ✅ |
| **E/F/G** | ROC & PR, 4 models × 3 datasets | `3_visualization/1_roc_pr_shorkie_fold.py` | `results.eqtl_scores`, `results.mpra_eval` | CPU ✅ |
| **H/I** | AUROC/AUPRC by TSS-distance | `3_visualization/2_AUROC_AUPRC_by_dsitance.py` | same | CPU ✅ |
| **J–O** | eQTL-SNP ISM saliency | `2_variant_scoring/` + ensemble ISM | `models.shorkie_finetuned` | GPU ✅ |

**Data (under `work_root`):** per-SNP scored TSVs `revision_experiments/eQTL/viz_new/results/negset_{1..4}/{exp}_Shorkie{,_LM,_Random_Init}_scores.tsv` (`results.eqtl_scores`); DREAM baselines under `results.mpra_eval` (Caudal/Kita) and `revision_experiments/eQTL/eQTL_MPRA_models_eval_Renganaath_etal/results` (Renganaath — a **different path**, handled in `get_mpra_base`). **Panel G is scored on the 142-variant TSS subset for all six models** — Shorkie family from `viz_new/results_subset_tss/` and DREAM from the matching `eQTL_MPRA_models_eval_Renganaath_etal/results_subset_tss/` (the subset-specific DREAM run). Reference ensemble PNGs at `viz_new/results{,_subset_tss}/{exp}/combined_plots/`. The eQTL SNPs come from the released positive-variant CSVs (`{exp}_Shorkie/positive/results/*.csv`). The A/B/J–O ISM `scores.h5` were **not** on disk → regenerated here on GPU.

### Reproduction approach
- **C/D/E/F/G/H/I (CPU)** — ported verbatim from `1_roc_pr_shorkie_fold.py` and `2_AUROC_AUPRC_by_dsitance.py` (the script's `args.root_dir` → `shorkie.config`). 6-model inner-join on `Position_Gene` per negset; ensemble mean±SEM over 4 negsets.
- **A/B + J–O (GPU)** — `panels/run_ism_eqtl.py` loads the released 8-fold Shorkie ensemble (RNA-Seq(T0) target slice — the sheet the eQTL benchmark used), replicates the scorer's **variant-aware 16 kb window placement** (variant inside input, gene body in the cropped output) and `gene.output_slice` gene-body bins, predicts ref/alt coverage (A/B) and runs a ±40 bp per-base logSED ISM scan (J–O). Saved to `reproduced/ism/{oma1,lap3}.npz`; the notebook renders from the npz.

---

## Phase 3 — Verification

**`reproduced/verify_fig07.csv`: 66/66 PASS** (against the published PDF; rebuilt by `recheck/build_verify_fig07.py`).

| Block | Checks | Result |
|---|---|---|
| **E/F/G match published figure** | 36 (3 datasets × 6 models × {AUROC, AUPRC}) | all PASS (Caudal/Kita Δ≤0.001; Renganaath via the uniform 6-way `results_subset_tss` join, Δ≤0.001; whole grid max \|Δ\|=0.0005) |
| **E/F/G direction** | 18 (Shorkie > Random_Init, > LM, > best-DREAM — all 3 datasets) | all PASS (incl. Shorkie > best-DREAM on **Renganaath**, now correct) |
| **H/I bin dominance** | 2 (Shorkie ≥ DREAM-RNN in **all** TSS-distance bins, Caudal & Kita) | all PASS (5/5 and 4/4 bins) |
| **A/B direction + coverage** | 4 (OMA1 logSED<0, LAP3 logSED>0; ref-pred vs obs R≥0.8) | PASS (−0.220 / +0.234; R 0.965 / 0.996) |
| **J–O Shorkie ISM saliency** | 6 (per-locus ISM saliency recomputed from cache) | all PASS |

**Reproduced vs paper Figure 7E/F/G (AUROC | AUPRC):**

Published Figure 7 legend (AUROC | AUPRC); reproduced values in `recheck/fig7EFG_auc.csv`:

| Model | Caudal | Kita | Renganaath |
|---|---|---|---|
| **Shorkie** | **0.564 \| 0.585** | **0.650 \| 0.643** | **0.618 \| 0.629** |
| Shorkie_LM | 0.513 \| 0.532 | 0.523 \| 0.555 | 0.474 \| 0.492 |
| Shorkie_Random_Init | 0.541 \| 0.551 | 0.641 \| 0.614 | 0.424 \| 0.447 |
| DREAM-Atten | 0.526 \| 0.536 | 0.534 \| 0.568 | 0.585 \| 0.588 |
| DREAM-CNN | 0.529 \| 0.537 | 0.539 \| 0.567 | 0.596 \| 0.593 |
| DREAM-RNN | 0.525 \| 0.538 | 0.533 \| 0.564 | 0.590 \| 0.596 |

Caudal & Kita reproduce to 3 decimals; Renganaath (panel G) now reproduces to ±0.000–0.001 (uniform 6-way `results_subset_tss` join), so the whole 36-cell E/F/G grid reproduces with **max |Δ| = 0.0005**. **Shorkie is the top model on all three datasets**, including Renganaath (0.618/0.629 > every DREAM ≈0.59) — consistent with the body text.

The **GPU ISM** reproduces the released per-SNP logSED **bit-exactly**: OMA1 (chrXI:604356 A>G) = **−0.2198** (= positive CSV −0.2198; alt reduces expr.), LAP3 (chrXIV:200328 G>A) = **+0.2344** (= CSV +0.2344; alt increases expr.) — both directions match the caption.

### Discrepancy log (honest)

| Item | Note |
|---|---|
| **Renganaath (panel G) — corrected, now exact** | The published Figure 7G plots **Shorkie 0.618/0.629**, the *top* model (beating DREAM ≈0.59), consistent with the body text. The previous reproduction matched an intermediate full-set reference PNG (Shorkie 0.536/0.555) and wrongly concluded "DREAM beats Shorkie." Root cause: panel G is scored on the **142-variant** `results_subset_tss/` set, not the full 395-variant `results/` set. Final fix (pass 5) mirrors the source plotter — a **uniform 6-way inner-join on `results_subset_tss` for all six models** (DREAM from the matching subset DREAM dir) — reproducing every G line to **±0.000–0.001** with curves overlaying the published (no per-model shift). See `recheck/DISCREPANCIES.md`. |
| **Dataset sizes** | Scored positives (post inner-join) ≈ Caudal 1712, Kita 655, Renganaath 142 (subset, = body text) / 395 (full) vs body text 1901 / 683 / 142. Differences reflect post-filter inclusion + the Position_Gene intersection. H/I per-bin Pos/Neg counts reproduce the published figure exactly. |
| **A/B SNPs vs display windows** | The dominant-effect eQTL SNP for each gene sits just **outside** the gene-body display window (OMA1 SNP 604356; LAP3 SNP 200328) — promoter-proximal, as expected for a cis-regulatory eQTL. |
| **J–O (6 panels)** | All six published ISM panels are reproduced (J YER080W, K YLR036C, L YKL078W, M YKR087C/OMA1, N YNL239W/LAP3, O YGR046W). Shorkie ISM REF/ALT logos are recomputed from the released ISM cache; **DREAM-RNN ISM now renders for all 6/6 loci** (K/O from the targeted `eQTL_MPRA_models_ISM_additional` run, byte-identical to the main run where they overlap); **panel N's Ref-DB motif is reverse-complemented** (CGGGTAA — its Reb1.1 site is on the (−) strand); the ISM logos show the published **grey padding bands on both sides** (Shorkie 112 bp, DREAM 110 bp); the **coverage track** is the **8-fold ensemble predicted Ref/Alt** RNA-Seq(T0) coverage (`run_cov_eqtl_jo.py`), gene-windowed and drawn as overlapping Ref(blue)+Alt(orange) bars with **x/y axis limits matching the published predicted scale**. Ref DB motif logos are embedded from the project motif DB; Avg-logSED/quantile annotations are carried from the released scoring. |

**Changes to legacy scripts:** none. `1_roc_pr_shorkie_fold.py` / `2_AUROC_AUPRC_by_dsitance.py` were ported into notebook cells with `args.root_dir` → `shorkie.config` (identical metric/label/binning logic, verified by the Δ=0.000 exact match). The GPU ISM reuses `shorkie.models.ensemble` (cf. `notebooks/fig10`).
