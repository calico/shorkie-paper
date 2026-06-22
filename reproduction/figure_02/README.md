# Figure 2 — Shorkie LM identifies conserved TF motifs across fungal genomes

> *"Shorkie LM identifies conserved transcription factor binding motifs across fungal genomes."*

Reproduction package for **main-text Figure 2**. Published reference: [`../../paper/Figures/Figure_2.pdf`](../../paper/Figures/Figure_2.pdf) (`published/Figure_2_full.png`).

- **Reproduce:** [`reproduce_figure_02.ipynb`](reproduce_figure_02.ipynb).
- **Verify:** [Verification](#phase-3--verification) + `reproduced/verify_fig02.csv`.

---

## Phase 1 — Discovery

Figure 2 demonstrates that the masked DNA LM reconstructs canonical yeast TF motifs *de novo* and that they generalize across fungi. 5 panels:

| Panel | Claim | Type | Generating script | Input + config key | Notebook |
|---|---|---|---|---|---|
| **A** | SMT3 promoter prediction (chrIV:1,469,090-198): Shorkie-LM vs SpeciesLM vs 15% iterative inference; poly(dA:dT), Cbf1, Tye7, Reb1 | comp (Shorkie row) + gpu/external (other rows) | `04_analysis/shorkie_lm/lm_SMT3_viz/2_viz_dna_pwm_shorkie_lm.py` | TF-MoDISco `.h5` → `results.modisco_lm`; SpeciesLM = external (Karollus et al.); iterative = LM GPU forward pass | `fig07` |
| **B** | iterative 15%-masked region prediction (per-iteration A/C/G/T matrix) | gpu | LM forward pass (`use_bert=true`, `mask_rate=0.15`) | `models.shorkie_lm` | none |
| **C** | TF-MoDISco motif grid, 6 datasets × 11 motifs (TBP, 5′ splice donor, branch pt, Cbf1p, Reb1.1, Snf1.1, Mcm1.1, Rap1.1, Sfp1.2, Abf1.1, Dot6) | comp | `04_analysis/shorkie_lm/motif_analysis/motif_lm/4_viz_motif.py` (in-dist); `…/motif_lm__unseen_species/` (cross-tier) | modisco `.h5` per tier → `results.modisco_lm` (+ `results.modisco_unseen`) | `fig03`, `fig04` |
| **D** | motif enrichment vs TSS + splice/branch within genes (histograms, True vs background) | comp | `04_analysis/shorkie_lm/motif_analysis/motif_lm/4_motif_to_tss_dist/3_plot_tss_dist_freq.py` | `motif_tss_distances.csv`, `background_tss_distances.csv` (`experiments_root/motif_LM/4_motif_to_tss_dist`) | none |
| **E** | t-SNE of genomic elements from the 1st self-attention layer | comp | `04_analysis/shorkie_lm/umap_cluster_promoter/2_viz_clusters_LM.py` | `embeddings_chr*.h5` (16) → `results.umap`, `genome.gtf` | `fig05` |

**Env:** `yeast_ml` (CPU). Data on disk: modisco `.h5` (173 MB, `saccharomycetales_viz_seq/unet_small_bert_drop/modisco_results_w16384_n100000.h5`); TSS-distance CSVs (58,170 motif hits, 101 motifs); 17 `embeddings_chr*.h5`. GPU/external only for 2A's SpeciesLM + iterative rows and 2B.

### Reproduction scope (after the deep recheck)
- **Fully reproduced (CPU):** 2C (the full 6×11 conservation grid), 2D (the published TFs), 2E (t-SNE, all 16 chr).
- **Reproduced (GPU):** 2B + the 2A 15%-iterative row — iterative-masking SMT3 inference (`panels/run_iterative_smt3.{py,sbatch}`).
- **Reproduced from cache (CPU):** the 2A Shorkie-LM row.
- **External (documented):** the 2A SpeciesLM row (Karollus / Tomaz da Silva et al. model; cached prediction shown, alignment not reconstructable).

The notes below are the *original* (pre-recheck) scope; the deep recheck closed the GPU gaps and rebuilt 2C/2D — see the Deep recheck section.

---

## Phase 2 — Reproduction (status)

Executed `reproduce_figure_02.ipynb` (8/8 cells, 0 errors, 5 embedded figures). Reproduced panels in `reproduced/`: `Figure_2A_reproduced.png` (Shorkie-LM SMT3 PWM), `Figure_2C_reproduced.png` (top-6 TF-MoDISco motifs, CWM+PWM), `Figure_2D_reproduced.png` (TSS-distance histograms), `Figure_2E_reproduced.png` (t-SNE). The 2E t-SNE is precomputed by `panels/precompute_tsne.py` (PCA-50 → capped t-SNE; the notebook loads the cache so it executes in ~70 s). **Node note:** t-SNE thrashes without thread limits — run heavy steps with `OMP_NUM_THREADS=4` and via `reproduction/common/run_in_tmux.sh` so they survive disconnection.

## Phase 3 — Verification

**`reproduced/verify_fig02.csv`: 16/16 PASS** (deep recheck — tightened from the original 3 qualitative checks). 2A ×3 · 2B ×2 · 2C ×5 · 2D ×3 · 2E ×3. See [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md).

### Visual-exactness refinement (second pass)
The panels were then re-rendered to match the published *appearance* (same DNA logos, regions, colours, styles; 2B skipped per request):
- **2A — real conservation DNA letter logos** (A green / C blue / G orange / T red), replicating the upstream `plot_dna_logo`: Shorkie LM (SMT3 gene-averaged upstream, plot[204:500], motifs poly(dA:dT)/Cbf1/Tye7/Reb1 annotated, 95.7% recon), Shorkie 15% iterative, and the external SpeciesLM (`all_prbs[97:207]`). The external SpeciesLM row **cannot be position-aligned** to the Shorkie window (argmax agreement ~0.3 across all mappings) — documented.
- **2C — the de-novo TF-MoDISco CWM-logo grid** (53/66 cells), replacing the q-value heatmap (kept as `recheck/Figure_2C_qval_heatmap.png`).
- **2D — the 3 published TFs in the exact published style** (green True / salmon Background); counts match exactly (Abf1.1 n=745, Rap1.1 n=644, Reb1p[CCGGGTAA] n=821). The genic features (ATG/5′SS/branch) use the authors' manual curation (different hit-filtering) — documented, not plotted.
- **2E — published class palette** (Promoter blue, Protein-coding green, Intergenic orange, tRNA red, Transposable purple).

See `recheck/DISCREPANCIES.md` → "Visual-exactness refinement" for full detail.

### Deep recheck (`recheck/`)
A stricter pass verified every panel against the **published** version, corrected two stale panels, and ran the GPU pieces. Highlights:
- **2C — rebuilt as the real 6-dataset × 11-motif conservation grid** (was 6 single-tier logos). Data-driven from the per-tier `modisco report` TOMTOM tables: recovered-TF count declines **R64=9 · strains=9 · Saccharomycetales=9 → Ascomycota=5 → Orbiliales=4 · Schizosacc.=4**; **Mcm1.1 confident through Orbiliales but absent in Schizosaccharomycetales**; Rap1.1/Abf1.1/Dot6 lost beyond Saccharomycetales. 47/54 cells match the published *curated* grid (the 7 differences — Sfp1 promiscuity, weak TATA, borderline Reb1/Snf1 — are documented).
- **2D — reproduced the published TFs Abf1.1/Rap1.1/Reb1p** (was the wrong subset Sfp1p/AZF1.6/GCR2.6/RPN4.7), each enriched near the TSS vs background (Abf1.1 n=745 matches the published panel exactly). The genic features (ATG/5′SS/branch) are not in the released TSS CSV.
- **2B + 2A-iterative — run on GPU** (`panels/run_iterative_smt3.{py,sbatch}`): iterative 15% masking (convention from `hound_eval_mlm_perplexity_region.py`), 7 iterations, full coverage; reconstructs the SMT3 promoter incl. the poly(dA:dT) motif. Shorkie unmasked↔iterative PWM corr = 0.60; unmasked recovers the true promoter at 95.1%.
- **2E — recomputed on all 16 chromosomes** (16,384 intervals, was capped at 2,500); 5 classes, silhouette = 0.079 (tRNA/TE/Promoter well separated).
- **2A SpeciesLM row** is external (`johahi/specieslm-fungi-upstream-k1`); its cached prediction is shown but cannot be position-aligned to the Shorkie window without re-running the external model (documented).
- Per-panel `recheck/panel_{A..E}_sidebyside.png` + `recheck/Figure_2_published_vs_reproduced.png`.

### Discrepancy log (superseded by `recheck/DISCREPANCIES.md`)
| Panel | Status | Note |
|---|---|---|
| 2A | rows 2–3 reproduced; row 1 external | Shorkie unmasked + 15% iterative reproduced (GPU); SpeciesLM external, alignment documented |
| 2B | **reproduced (GPU)** | iterative 15%-masked PPM reconstruction, 7 iterations, full coverage |
| 2C | **grid reproduced** | data-driven 6×11 TOMTOM conservation grid; 47/54 vs published curation; curation differences documented |
| 2D | **published TFs reproduced** | Abf1.1/Rap1.1/Reb1p (corrected from a wrong subset); each near-TSS enriched |
| 2E | reproduced (all 16 chr) | silhouette-quantified separation; t-SNE layout stochastic |
