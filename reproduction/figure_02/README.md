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

### Reproduction scope
- **Fully reproduced (CPU):** 2C (in-distribution TF-MoDISco motifs), 2D (TSS-distance histograms), 2E (t-SNE).
- **Partial (CPU proxy + documented):** 2A — the Shorkie-LM PWM logo is reproduced from modisco; the SpeciesLM comparison row needs the external Karollus et al. model and the iterative-inference row needs a GPU forward pass over the SMT3 window.
- **Documented gap (GPU):** 2B (iterative 15%-masked prediction matrix).
- **Cross-tier (2C grid):** the in-distribution motifs are shown here; the 6-dataset conservation grid (Ascomycota/Orbiliales/Schizosaccharomycetales/strains rows) is the subject of `fig04` / `motif_lm__unseen_species`.

---

## Phase 2 — Reproduction (status)

Executed `reproduce_figure_02.ipynb` (8/8 cells, 0 errors, 5 embedded figures). Reproduced panels in `reproduced/`: `Figure_2A_reproduced.png` (Shorkie-LM SMT3 PWM), `Figure_2C_reproduced.png` (top-6 TF-MoDISco motifs, CWM+PWM), `Figure_2D_reproduced.png` (TSS-distance histograms), `Figure_2E_reproduced.png` (t-SNE). The 2E t-SNE is precomputed by `panels/precompute_tsne.py` (PCA-50 → capped t-SNE; the notebook loads the cache so it executes in ~70 s). **Node note:** t-SNE thrashes without thread limits — run heavy steps with `OMP_NUM_THREADS=4` and via `reproduction/common/run_in_tmux.sh` so they survive disconnection.

## Phase 3 — Verification

**`reproduced/verify_fig02.csv`: 3/3 PASS.**
- **2C** — TF-MoDISco recovers ≥6 motifs (the released `.h5` has the full pos-pattern set). ✅
- **2D** — pooled TF-motif hits sit nearer the TSS than the genome-wide background (median |distance to TSS|: TF ≪ background). ✅
- **2E** — all 5 genomic-element classes present and cluster (Protein-coding 1046, Promoter 960, Intergenic 449, tRNA 38, Transposable element 7). ✅

**Visual:** reproduced panels vs `published/Figure_2_full.png` — 2C motif logos (poly(dA:dT)/TATA-like, Reb1, etc.), 2D upstream-of-TSS enrichment, and 2E element clusters match the published structure.

### Discrepancy log
| Panel | Status | Note |
|---|---|---|
| 2A | partial | Shorkie-LM PWM reproduced (CPU); SpeciesLM comparison row = external Karollus et al. model; 15% iterative-inference row = GPU forward pass over the SMT3 window |
| 2B | documented gap | iterative 15%-masked prediction matrix requires a GPU LM forward pass |
| 2C | reproduced (in-distribution) | the full 6-dataset × 11-motif conservation grid's cross-tier rows come from per-tier MoDISco (`fig04`/`motif_lm__unseen_species`); motif→TF naming via TOMTOM/FIMO |
| 2D | reproduced | a representative promoter-TF subset is shown (not the exact 6 published TFs); the near-TSS enrichment claim holds |
