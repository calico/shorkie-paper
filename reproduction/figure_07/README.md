# Figure 7 — Shorkie accurately predicts cis-eQTL variant effects

> *"Shorkie accurately predicts cis-eQTL variant effects."*

Reproduction package for **main-text Figure 7**. Published reference: [`../../paper/Figures/Figure_7.pdf`](../../paper/Figures/Figure_7.pdf) (`published/Figure_7_full.png`).

- **Reproduce:** [`reproduce_figure_07.ipynb`](reproduce_figure_07.ipynb) (executed, 0 errors).
- **Verify:** `reproduced/verify_fig07.csv` — **66/66 PASS** against the **published PDF** (`paper/Figures/Figure_7.pdf`).
- **Deep-recheck:** `recheck/` — every panel rebuilt to match the published figure panel-for-panel; see [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md) and the side-by-sides `recheck/panel_{AB,C,D,EFG,HI,JO}_sidebyside.png`.
- **GPU panels:** `panels/run_ism_eqtl.{py,sbatch}` (8-fold ensemble logSED-ISM for the OMA1/LAP3 eQTL SNPs). J–O Shorkie ISM logos are rebuilt from the released ISM cache (no GPU rerun).

The headline result: ensemble **AUROC/AUPRC matches the published Figure 7E/F/G** (Caudal/Kita to 3 decimals; Renganaath within 0.4–0.7 %, see below), the GPU ISM reproduces the released per-SNP logSED **bit-exactly** (OMA1 −0.2198, LAP3 +0.2344), and the J–O Shorkie ISM saliency logos are recomputed bit-for-bit from cache.

> **Deep-recheck correction (this pass).** The earlier reproduction matched intermediate
> reference PNGs and reported **Shorkie 0.536/0.555 on Renganaath (panel G)**, concluding "DREAM
> beats Shorkie." Reading the **published PDF** shows panel G plots **Shorkie 0.618/0.629** — the
> top model — consistent with the body text. Root cause: panel G uses the **142-variant**
> `viz_new/results_subset_tss/` set (= the "142 causal core-promoter variants"), not the full
> 395-variant `viz_new/results/` set. Corrected here; details in `recheck/DISCREPANCIES.md`.

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

**Data (under `work_root`):** per-SNP scored TSVs `revision_experiments/eQTL/viz_new/results/negset_{1..4}/{exp}_Shorkie{,_LM,_Random_Init}_scores.tsv` (`results.eqtl_scores`); DREAM baselines under `results.mpra_eval` (Caudal/Kita) and `revision_experiments/eQTL/eQTL_MPRA_models_eval_Renganaath_etal/results` (Renganaath — a **different path**, handled in `get_mpra_base`). Reference ensemble PNGs at `viz_new/results/{exp}/combined_plots/`. The eQTL SNPs come from the released positive-variant CSVs (`{exp}_Shorkie/positive/results/*.csv`). The A/B/J–O ISM `scores.h5` were **not** on disk → regenerated here on GPU.

### Reproduction approach
- **C/D/E/F/G/H/I (CPU)** — ported verbatim from `1_roc_pr_shorkie_fold.py` and `2_AUROC_AUPRC_by_dsitance.py` (the script's `args.root_dir` → `shorkie.config`). 6-model inner-join on `Position_Gene` per negset; ensemble mean±SEM over 4 negsets.
- **A/B + J–O (GPU)** — `panels/run_ism_eqtl.py` loads the released 8-fold Shorkie ensemble (RNA-Seq(T0) target slice — the sheet the eQTL benchmark used), replicates the scorer's **variant-aware 16 kb window placement** (variant inside input, gene body in the cropped output) and `gene.output_slice` gene-body bins, predicts ref/alt coverage (A/B) and runs a ±40 bp per-base logSED ISM scan (J–O). Saved to `reproduced/ism/{oma1,lap3}.npz`; the notebook renders from the npz.

---

## Phase 3 — Verification

**`reproduced/verify_fig07.csv`: 66/66 PASS** (against the published PDF; rebuilt by `recheck/build_verify_fig07.py`).

| Block | Checks | Result |
|---|---|---|
| **E/F/G match published figure** | 36 (3 datasets × 6 models × {AUROC, AUPRC}) | all PASS (Caudal/Kita Δ≤0.001; Renganaath via `results_subset_tss`, max \|Δ\|=0.0074) |
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

Caudal & Kita reproduce to 3 decimals; the whole 36-cell E/F/G grid reproduces with **max |Δ| = 0.0074**. **Shorkie is the top model on all three datasets**, including Renganaath (0.618/0.629 > every DREAM ≈0.59) — consistent with the body text.

The **GPU ISM** reproduces the released per-SNP logSED **bit-exactly**: OMA1 (chrXI:604356 A>G) = **−0.2198** (= positive CSV −0.2198; alt reduces expr.), LAP3 (chrXIV:200328 G>A) = **+0.2344** (= CSV +0.2344; alt increases expr.) — both directions match the caption.

### Discrepancy log (honest)

| Item | Note |
|---|---|
| **Renganaath (panel G) — corrected** | The published Figure 7G plots **Shorkie 0.618/0.629**, the *top* model (beating DREAM ≈0.59), consistent with the body text. The previous reproduction matched an intermediate full-set reference PNG (Shorkie 0.536/0.555) and wrongly concluded "DREAM beats Shorkie." Root cause: panel G uses the **142-variant** `viz_new/results_subset_tss/` set, not the full 395-variant `results/` set. Recompute on the subset → 0.614/0.624 (within 0.4–0.7 % of published). See `recheck/DISCREPANCIES.md`. |
| **Dataset sizes** | Scored positives (post inner-join) ≈ Caudal 1712, Kita 655, Renganaath 142 (subset, = body text) / 395 (full) vs body text 1901 / 683 / 142. Differences reflect post-filter inclusion + the Position_Gene intersection. H/I per-bin Pos/Neg counts reproduce the published figure exactly. |
| **A/B SNPs vs display windows** | The dominant-effect eQTL SNP for each gene sits just **outside** the gene-body display window (OMA1 SNP 604356; LAP3 SNP 200328) — promoter-proximal, as expected for a cis-regulatory eQTL. |
| **J–O (6 panels)** | All six published ISM panels are reproduced (J YER080W, K YLR036C, L YKL078W, M YKR087C/OMA1, N YNL239W/LAP3, O YGR046W). Shorkie ISM REF/ALT logos are recomputed from the released ISM cache; DREAM-RNN ISM is rendered from cached deltas for **4/6 loci on disk** (K, O absent); Ref DB motif logos are embedded from the project motif DB; Avg-logSED/quantile annotations are carried from the released scoring. |

**Changes to legacy scripts:** none. `1_roc_pr_shorkie_fold.py` / `2_AUROC_AUPRC_by_dsitance.py` were ported into notebook cells with `args.root_dir` → `shorkie.config` (identical metric/label/binning logic, verified by the Δ=0.000 exact match). The GPU ISM reuses `shorkie.models.ensemble` (cf. `notebooks/fig10`).
