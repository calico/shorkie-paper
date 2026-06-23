# Figure 2 — deep-recheck discrepancy report

Exact reproduction of every reproduced Figure-2 panel against the **published** version,
with root-cause for each residual. Figure 2 = *"Shorkie LM identifies conserved
transcription factor binding motifs across fungal genomes."*

**Scope (final).** The notebook reproduces **2A, 2D, 2E**; **2C** is provided as the
original author scripts and not re-rendered; **2B** is out of scope.

- **2A** — three SMT3 promoter logos (SpeciesLM / Shorkie-LM / Shorkie 15% iterative),
  all registered to `SMT3_seq[690:800]`.
- **2B** — *removed.* The per-iteration 15%-masked prediction matrix (a GPU forward
  pass) is not part of this reproduction.
- **2C** — *provided as upstream scripts, not re-rendered.* The 6-dataset × 11-motif
  TF-MoDISco conservation grid is generated directly from the per-tier MoDISco results by
  `motif_lm/4_viz_motif.py` (R64) + `motif_lm__unseen_species/4_viz_motif.py`
  (cross-tier), refactored to config-driven paths with `.sh` wrappers. The earlier
  data-driven re-derivation (a TOMTOM q-value grid) was removed in favour of the actual
  generating scripts.
- **2D** — the published **6-panel** TSS-distance grid (was only 3 TFs).
- **2E** — t-SNE on all 16 chromosomes, **faithful to the original** (no PCA pre-reduction).

Recheck artifacts (this directory): `build_2A_logos.py`, `build_2D_tss.py`,
`build_2E_tsne.py`, `render_2E.py`, `build_verify_fig02.py`, `make_sidebyside_fig02.py`;
CSVs (`fig2D_enrichment`, `fig2E_separation`, `fig2A_consistency`, `recheck_checks_fig02`);
`panel_{A,C,D,E}_sidebyside.png` (C published-only) + `Figure_2_published_vs_reproduced.png`.

---

## Panel notes (matched to the published appearance)

- **2A — three conservation DNA logos, all registered to the same window.** Replicates the
  upstream `plot_dna_logo` (conservation = 2−entropy; per-position letter heights =
  p·conservation) with the **published colour scheme (A green, C blue, G orange, T red)**,
  three stacked rows over the **same 110 bp window `SMT3_seq[690:800]`** so **poly(dA:dt),
  Cbf1p, Tye7.1** line up vertically. **SpeciesLM** row = the external
  `johahi/specieslm-fungi-upstream-k1` model's reconstruction PWM over the 1003 bp SMT3
  upstream, **regenerated** by `lm_SMT3_viz/0_compute_specieslm_smt3.py`
  (`reproduced/specieslm_smt3/all_prbs_SMT3.npy`; argmax-vs-genome **0.946**), plot
  `[690:800]`. **Shorkie LM** row = SMT3 (YDR510W) gene-averaged 512 bp-upstream `x_pred`
  PWM, slice `[201:311]` (== `SMT3_seq[690:800]`; genome recon **0.955**). **Shorkie 15%
  iterative** row = the masked-LM 15%-iterative reconstruction from the **precomputed
  3-architecture ensemble** (`unet_small_bert_drop` + `retry_1` + `retry_2`,
  `preds_train.npz` — the published source that `2_viz_dna_pwm_shorkie_lm.py` averages),
  gene-averaged via `6_extract_iterative_3arch.py` → `preds_smt3_iterative_3arch.npz`,
  slice `[201:311]` — recon **0.61** (corr **0.81** to unmasked). The three rows are
  genomically registered (`SMT3_seq[690]==upstream[201]`, string-verified); **SpeciesLM-vs-
  Shorkie registered PWM correlation 0.95**.
- **2D — the published 6-panel grid, exact `3_plot_tss_dist_freq.py` style** (seaborn
  whitegrid, 50 bins, xlim ±2500, flipped distance, green **True** / salmon **Background**,
  dashed TSS line, "Distance Distribution for … (True: n=.., Background: n=..)" titles).
  All six panels come straight from the released `motif_tss_distances.csv` /
  `background_tss_distances.csv`. Three of them are TF motifs the paper **relabelled** to
  the genic feature they mark (confirmed by the authors), the other three are promoter TFs:

  | published title | CSV `motif_name` | n |
  |---|---|---|
  | start codon (ATG) | `MIG3.4` | 2218 |
  | Abf1.1 | `ABF1.1` | 745 |
  | Rap1.1 | `RAP1.1` | 644 |
  | Reb1p | `Reb1p&consensus=CCGGGTAA` | 821 |
  | 5' splice site (donor site) | `CHA4.11` | 603 |
  | branch point | `SWI5.7` | 779 |

  `build_2D_tss.py` asserts each motif's True hit count against the published n, so the
  panel both reproduces the figure and verifies the relabelling against the released data.
- **2E — faithful to the original** (no PCA pre-reduction; sklearn-default t-SNE), all 16
  chromosomes, published class palette (Promoter blue, Intergenic orange, Protein-coding
  green, tRNA red, Transposable purple), no title, no grid, s=3. silhouette ≈ 0.08.

---

## Per-panel findings

| Panel | Type | Match to published | Residual / note | Root cause |
|---|---|---|---|---|
| **2A** | logos (3 rows, 1 external regenerated) | **all 3 rows reproduced + aligned** | none | wrong cached file was chrX (fixed); see below |
| **2B** | GPU (iterative MLM) | **out of scope** | not reproduced | per-iteration matrix needs a GPU forward pass |
| **2C** | computational (MoDISco) | **upstream scripts provided** | grid not re-derived in the notebook | depends on the full per-tier MoDISco `.h5` set |
| **2D** | computational | **published 6-panel grid reproduced** | none (counts asserted) | relabel: start codon=MIG3.4, 5′SS=CHA4.11, branch=SWI5.7 |
| **2E** | computational (t-SNE) | reproduced, faithful (no PCA) | t-SNE layout stochastic | seed/implementation (separation quantified) |

---

## What was wrong before, and fixed

### 2A — all three rows reproduced + aligned (SpeciesLM regenerated; the old "not alignable" was a wrong-file bug)
- Published 2A overlays the SMT3 (YDR510W, chrIV:1,469,400 ATG, + strand) promoter logo
  from 3 sources, all over the same 110 bp window `SMT3_seq[690:800]`.
- **Root cause of the old failure:** the committed reproduction loaded
  `dependencies_DNALM/all_prbs.npy`, which on disk is **byte-identical to
  `all_prbs_chrX_607855_608355.npy`** — a chrX:607,855-608,355 example (shape `(500,4)`),
  a different locus entirely (argmax vs SMT3 ~0.26–0.34). The prior recheck then wrongly
  concluded the external row was "not position-alignable (best corr 0.17)".
- **Fix:** `0_compute_specieslm_smt3.py` re-runs the external model over the 1003 bp
  `SMT3_five_prime_seq` to regenerate `all_prbs_SMT3.npy` (argmax-vs-genome **0.946**);
  `SMT3_seq[690]==Shorkie-upstream[201]` (string-verified), so all three rows are
  registered and the motifs line up. Registered SpeciesLM-vs-Shorkie corr **0.95**.
- The 15%-iterative row was also corrected to the **precomputed 3-architecture ensemble**
  (`6_extract_iterative_3arch.py`), the source the published `2_viz` averages — recon
  0.61 / corr 0.81 to unmasked, vs the earlier single-model GPU job's 0.52.

### 2D — was the wrong subset / "genic features not reproducible"; now the full published 6-panel grid
- The first reproduction plotted the wrong TF subset; a later pass narrowed it to the 3
  promoter TFs **Abf1.1 / Rap1.1 / Reb1p** and documented the bottom-row "genic features"
  (start codon ATG / 5′SS donor / branch point) as **not reproducible** from released data
  — the auto-identified raw modisco hits (5′SS n=3077, branch n=12859) did not match the
  published curated counts (603 / 779), and ATG was "not a clean modisco PWM".
- **Resolution (author correction):** those three "genic features" are **TF motifs that
  the paper relabelled** — **start codon (ATG) = MIG3.4**, **5′ splice site (donor site) =
  CHA4.11**, **branch point = SWI5.7** — and they live in the **same released
  `motif_tss_distances.csv`** as Abf1.1/Rap1.1/Reb1p. Plotting all six by their CSV
  `motif_name` reproduces the published 6-panel grid exactly, with hit counts matching the
  published n (2218 / 745 / 644 / 821 / 603 / 779). The "not reproducible" residual was an
  artifact of looking in the wrong file (the modisco-pattern genic-stats analysis) under
  the wrong (generic) labels.

### 2C — the data-driven re-derivation was removed in favour of the original scripts
- An intermediate recheck **re-derived** the 6×11 grid from the per-tier `modisco report`
  TOMTOM tables (a `-log10(q)` presence/heatmap grid). That re-derivation reproduced the
  conservation structure (recovered-TF count declines R64=9 → Schizosacc.=4; Mcm1.1
  confident through Orbiliales but absent in Schizosaccharomycetales; promoter TFs lost
  beyond Saccharomycetales) but did **not** match the published *appearance* (a CWM-logo
  grid) and was a re-implementation rather than the authors' code.
- **Decision:** drop the re-derivation (`build_2C_grid.py`, `build_2C_logos.py`, the CSVs
  and the q-value heatmap) and keep the **original generating scripts**
  `motif_lm/4_viz_motif.py` (R64) + `motif_lm__unseen_species/4_viz_motif.py`
  (cross-tier), refactored to config-driven output paths with `.sh` wrappers. The notebook
  cites these and skips re-rendering the grid (it needs the full per-tier MoDISco `.h5`
  set) — the same treatment as the Fig 5 motif-progression panels.

### 2B — removed (out of scope)
- An intermediate recheck ran 2B on GPU (`panels/run_iterative_smt3.{py,sbatch}`). That
  reproduction has been **removed** — the per-iteration 15%-masked prediction matrix is
  not part of this reproduction. (The 2A 15%-iterative row does not depend on it; it uses
  the precomputed 3-architecture ensemble.)

### 2E — t-SNE made faithful to the original (was PCA-50 → capped t-SNE)
- The committed reproduction inserted a **PCA-50 pre-reduction** + `init="pca"` (and an
  earlier version capped at 2,500 points), which changed the t-SNE geometry so the layout
  did not match the published scatter.
- **Fix:** match the original `2_viz_clusters_LM.py` exactly —
  `TSNE(n_components=2, random_state=42, verbose=1)` run **directly on the full
  embeddings (no PCA)** over all qualifying intervals across the 16 chromosomes (≈16,384
  points), published palette, no title, no grid, s=3. Cluster separation quantified by
  **silhouette ≈ 0.08** (tRNA ~0.88, Transposable ~0.39, Promoter ~0.22 well separated;
  Intergenic/Protein-coding overlap slightly — biologically sensible). t-SNE layout is
  stochastic across sklearn versions, so the separation metric is the quantitative tie.

## Numbers
`build_verify_fig02.py` → `verify_fig02.csv` / `recheck_checks_fig02.csv`: **all PASS**
— 2A ×6, 2D ×12 (6 enriched-near-TSS + 6 published-count matches), 2E ×3.
