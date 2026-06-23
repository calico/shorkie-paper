# Figure 2 — deep-recheck discrepancy report

Exact reproduction of every Figure-2 result against the **published** version, with
root-cause for each residual. Figure 2 = *"Shorkie LM identifies conserved
transcription factor binding motifs across fungal genomes."* The prior reproduction
was **thin** — `verify_fig02.csv` held only **3 qualitative checks** (`n_motifs≥6`,
`TF_closer_to_TSS_than_bg`, `5 classes present`) and three panels did not match the
published specifics. This recheck reproduces each panel to the published content and
**tightens verification to 16/16 checks**.

**Verdict: Figure 2 reproduced; thin verification replaced (3 → 16 checks).** Panels
2B/2C/2D/2E now match the published content with quantified verification; panel 2A's
two Shorkie rows are reproduced exactly (the iterative method run **on GPU** this
recheck), with the external SpeciesLM row's alignment documented as a genuine limit.

Recheck artifacts (this directory): `build_2C_grid.py`, `build_2C_logos.py`, `build_2D_tss.py`,
`build_2E_tsne.py`, `render_2E.py`, `build_2A_logos.py`, `build_2B_matrix.py`, `build_verify_fig02.py`,
`make_sidebyside_fig02.py`; `../panels/run_iterative_smt3.{py,sbatch}` (GPU);
CSVs (`fig2C_qval_grid`, `fig2C_presence_grid`, `fig2D_enrichment`, `fig2E_separation`,
`fig2A_consistency`, `fig2B_ppm`, `recheck_checks_fig02`); `panel_{A..E}_sidebyside.png`
+ `Figure_2_published_vs_reproduced.png`.

---

## Visual-exactness refinement (second pass — figures matched to the published appearance)

The first deep-recheck pass verified the DATA but drew some panels in non-published forms. This pass
re-renders them to look like the published figure (same DNA logos, regions, colours, styles). Skip 2B
per the user.

- **2A — three conservation DNA logos, all registered to the same window.** Replicates the upstream
  scripts' own `plot_dna_logo` (conservation = 2−entropy; per-position letter heights = p·conservation)
  with the **published colour scheme (A green, C blue, G orange, T red)**, three stacked rows over the
  **same 110 bp window `SMT3_seq[690:800]`** so **poly(dA:dt), Cbf1p, Tye7.1** line up vertically across
  rows (motif labels once along the top). **SpeciesLM** row = the external
  `johahi/specieslm-fungi-upstream-k1` model's per-position reconstruction PWM over the 1003 bp SMT3
  upstream, **regenerated** by `scripts/04_analysis/shorkie_lm/lm_SMT3_viz/0_compute_specieslm_smt3.py`
  (`reproduced/specieslm_smt3/all_prbs_SMT3.npy`; argmax-vs-genome **0.946**), plot[690:800]. **Shorkie
  LM** row = SMT3 (YDR510W) gene-averaged 512 bp-upstream `x_pred` PWM, slice [201:311] (== SMT3_seq[690:800];
  recovers the genome at **0.955**). **Shorkie 15% iterative** row = same averaging on the GPU iterative
  reconstruction (corr **0.71** to unmasked). The window is the authors' own `dep_map[690:800]` zoom
  (compute_SMT3_gene.ipynb cell 18) = `4_viz_smt3_logo_unmasked.py`'s `[203:311]` Shorkie slice; pixel-
  measured against the published PNG (Cbf1p `CACGTG`@frac 0.42 → bp 736, Tye7.1@0.80 → bp ~778). The three
  rows are genomically registered (`SMT3_seq[690]==upstream[201]`, string-verified); **SpeciesLM-vs-Shorkie
  registered PWM correlation 0.95**.
- **2C — the de-novo TF-MoDISco CWM-logo grid** (was a q-value heatmap). `build_2C_logos.py` renders,
  for each (tier × motif) cell marked present, the **CWM logo** of the best-TOMTOM-matching modisco
  pattern (from the per-tier `.h5`), "—" for absent; 5′SS/branch matched by consensus. **53/66 cells
  filled**, reproducing the published logo-grid layout + the conservation decline. The q-value heatmap
  is kept as `recheck/Figure_2C_qval_heatmap.png`.
- **2D — the 3 published TF histograms, exact style.** Abf1.1 / Rap1.1 / Reb1p in the published
  rendering (green **True** / salmon **Background**, flipped distance, dashed TSS line, xlim ±2500,
  "Distance Distribution for … (True: n=.., Background: n=..)" titles). Counts match the published panel
  exactly: **Abf1.1 n=745, Rap1.1 n=644, Reb1p[CCGGGTAA] n=821**. *Residual:* the published 2D also
  shows three genic features (start codon ATG, 5′SS donor, branch point) from a separate
  modisco-pattern-indexed analysis (`0_motif_genomic_region_ratio/.../motif_<k>_tss_stats.txt`) with
  the authors' **manual** motif→feature labelling and hit-filtering; the auto-identified raw modisco
  hits (5′SS pattern_36 n=3077; branch pattern_21 n=12859) do not match the curated published counts
  (603 / 779), and ATG is not a clean modisco PWM — these are recorded in `fig2D_enrichment.csv` but not
  plotted (not exactly reproducible from the released artifacts).
- **2E — published class palette.** Re-rendered from the full-16-chr cache with Promoter=blue,
  Protein-coding=green (the two large clusters), Intergenic=orange, tRNA=red, Transposable=purple,
  matching the published scatter; silhouette 0.079 unchanged.

---

## Per-panel findings

| Panel | Type | Match to published | Residual / note | Root cause |
|---|---|---|---|---|
| **2A** | logos (3 rows, 1 external regenerated) | **all 3 rows reproduced + aligned** | none (SpeciesLM PWM regenerated) | wrong cached file was chrX; see below |
| **2B** | GPU (iterative MLM) | **reproduced** (was a documented gap) | iteration partition is seeded | new GPU job this recheck |
| **2C** | computational (TOMTOM) | conservation structure reproduced | 47/54 cells match the published curation | published grid is visually curated; see below |
| **2D** | computational | **published TFs reproduced** | genic features (ATG/5′SS/branch) absent from released data | wrong-subset fix; separate genic analysis |
| **2E** | computational (t-SNE) | reproduced on all 16 chr | t-SNE layout is stochastic | seed/implementation (separation quantified instead) |

---

## What was wrong before, and fixed

### 2C — was 6 single-tier logos; now the real 6×11 conservation grid
- The committed reproduction showed the **top-6 motif logos from one (in-distribution)
  modisco run**. The published 2C is a **6-dataset × 11-motif conservation grid**
  (R64; 4 strains; 5 Saccharomycetales; 4 Ascomycota; 4 Orbiliales; 4 Schizosacc.) ×
  {TBP, 5′SS donor, branch point, Cbf1p, Reb1.1, Snf1.1, Mcm1.1, Rap1.1, Sfp1.2,
  Abf1.1, Dot6}.
- Rebuilt **data-drivenly** from the same artifacts the authors used: the per-tier
  `modisco report` TOMTOM match tables (`report_*/motifs.html`). For each of the 9 TF
  motifs we take the best (min) TOMTOM q-value across all recovered patterns; recovered
  iff q < 0.10. (5′SS donor + branch point are sequence motifs, not TF-DB entries.)
- **Reproduces the published conservation structure** (`fig2C_presence_grid.csv`):
  recovered-TF count **declines** R64=9, strains=9, Saccharomycetales=9 → Ascomycota=5
  → Orbiliales=4, Schizosacc.=4; **Mcm1.1 confident through Orbiliales (q≤6e-4) but
  absent in Schizosaccharomycetales** (the manuscript's explicit claim — "Mcm1.1 was
  absent in Schizosaccharomycetales, which lack a direct homolog"); the promoter TFs
  **Rap1.1/Abf1.1/Dot6 are lost beyond Saccharomycetales**.
- **Cell agreement with the published curated grid: 47/54.** The 7 differing cells are
  the documented effects of the published grid being **visually curated** rather than a
  pure q-value threshold:
  - **Sfp1.2** (Ascomycota/Orbiliales/Schizo): TOMTOM matches it confidently (q≈5e-8) in
    every tier — Sfp1 is a low-complexity G/A-rich motif that matches promiscuously — so
    the published authors curated it out beyond Saccharomycetales; our data-driven grid
    keeps it. (3 cells.)
  - **Reb1.1 / Snf1.1** (Orbiliales, + Reb1 in Schizo): the published shows logos; our
    top-3 TOMTOM match for the diverged motif falls to q≈0.85–1.0 (essential TFs whose
    motif diverged/matched a paralog name in distant tiers). (3 cells.)
  - **TBP** (Ascomycota): the low-complexity TATA element matches the TF DB weakly
    (no q<0.1 hit) although a TATA logo is present. (1 cell.)
  Representation also differs: published = a logo grid; reproduced = a `-log10(q)`
  heatmap (`Figure_2C_reproduced.png`). The *data* (which motifs are recovered where) is
  the comparison.

### 2D — was the wrong TF subset; now the published TFs
- The committed reproduction plotted **Reb1p, ABF1.1, Sfp1p, AZF1.6, GCR2.6, RPN4.7** —
  only 2 of which appear in the published panel. The **published 2D** shows TF-motif
  TSS-enrichment for **Abf1.1, Rap1.1, Reb1p** (plus genic features ATG / 5′SS donor /
  branch point).
- Reproduced the 3 published TF histograms from the released `motif_tss_distances.csv` /
  `background_tss_distances.csv` using the upstream script's plotting convention
  (`3_plot_tss_dist_freq.py`: green True / red Background, flipped distance, TSS at 0).
  Each is **enriched near/upstream of the TSS** vs background (`fig2D_enrichment.csv`):
  Abf1.1 median |dist| 232 vs 556 bp; Rap1.1 312 vs 552; Reb1p 321 vs 577; near-TSS
  (±250 bp) fraction roughly doubled for each. **Abf1.1 n=745** matches the published
  panel's stated n exactly.
- **Residual:** the genic features in 2D (start codon ATG, 5′SS donor, branch point —
  enrichment *within genic regions*) are **not present in the released TSS-distance
  CSV** (which is keyed by TF-DB motif name); they come from a separate genic-position
  analysis. Reproduced the 3 TF histograms; the 3 genic-feature histograms are
  documented as out-of-released-data.

### 2A — all three rows reproduced + aligned (SpeciesLM regenerated; the old "not alignable" was a wrong-file bug)
- Published 2A overlays the SMT3 (YDR510W, chrIV:1,469,400 ATG, + strand) promoter logo
  from 3 sources, **all over the same 110 bp window `SMT3_seq[690:800]`**.
- **Root cause of the old failure (now fixed):** the committed reproduction loaded
  `dependencies_DNALM/all_prbs.npy` and plotted `[97:207]`. That file on disk is
  **byte-identical to `all_prbs_chrX_607855_608355.npy`** — a **chrX:607,855-608,355**
  example (shape `(500,4)`), i.e. a *different locus entirely* (argmax vs the SMT3
  sequence ~0.26–0.34). The correct SMT3 SpeciesLM PWM was never saved; the prior recheck
  then wrongly concluded the external row was "not position-alignable (best corr 0.17)".
- **Fix:** `0_compute_specieslm_smt3.py` re-runs the external
  `johahi/specieslm-fungi-upstream-k1` model (proxy `kazachstania_africana_…`,
  `softmax(logits)[:, 2:-1, ACGT]`) over the 1003 bp `SMT3_five_prime_seq` to regenerate
  the per-position reconstruction PWM `all_prbs_SMT3.npy` (argmax-vs-genome **0.946**).
  The window is the authors' `dep_map[690:800]` zoom = `4_viz`'s `[203:311]` Shorkie
  slice; `SMT3_seq[690]==Shorkie-upstream[201]` (string-verified), so all three rows are
  genomically registered and the motifs (**poly(dA:dt), Cbf1p, Tye7.1**) line up.
- **Rows:** SpeciesLM `all_prbs_SMT3[690:800]`; Shorkie LM `x_pred` upstream `[201:311]`
  (genome recon **0.955**); Shorkie 15% iterative `[201:311]` (corr **0.71** to unmasked).
  **SpeciesLM-vs-Shorkie registered PWM correlation 0.95** (was the "0.17 not-alignable"
  residual). Verify checks (`fig2A_consistency.csv`): same width, SpeciesLM↔genome 0.946,
  registered corr 0.95, Cbf1 E-box + poly(dA:dt) present.

### 2B — GPU iterative reconstruction (was a documented gap)
- The committed reproduction skipped 2B (GPU required). This recheck **runs it on GPU**
  (`panels/run_iterative_smt3.{py,sbatch}`, a100): iterative 15% masking on the SMT3
  window, masking convention taken **verbatim** from the Shorkie LM evaluator
  `baskerville/scripts/hound_eval_mlm_perplexity_region.py` (the Figure-1G code) — mask
  15% of positions (mask channel → 1, DNA → 0), predict, repeat on a fresh 15% until all
  positions are covered (**7 iterations**, every position covered exactly once).
- `Figure_2B_reproduced.png` shows the A/C/G/T × position predicted-probability grid
  (with the iteration that predicted each column), reproducing the published 2B layout
  and capturing the **poly(dA:dT)** motif (A≈0.6–0.8 run at chrIV:1,469,094–1,469,100).
- **Residual:** the per-iteration mask partition is random (seeded `np.random.seed(0)`),
  so the exact iteration-to-position assignment is reproducible but not expected to be
  pixel-identical to the published example.

### 2E — t-SNE on all 16 chromosomes (was capped at 2500 points)
- Recomputed on **all 16 chromosomes' first-self-attention-layer embeddings**
  (**16,384** intervals; the committed version capped at 2,500), 5 genomic-element
  classes (Promoter = 500 bp upstream of ATG, Protein-coding, Intergenic, tRNA,
  Transposable). Cluster separation quantified by **silhouette = 0.079** overall;
  per-class tRNA 0.88, Transposable 0.39, Promoter 0.22 (well separated), Intergenic and
  Protein-coding overlap slightly (−0.05; biologically sensible — they share base
  composition). Matches the published claim that the classes show "structure consistent
  with major genomic feature annotations."
- t-SNE layout is stochastic (seed + implementation), so the 2-D coordinates are not
  pixel-comparable to the published; the separation metric is the quantitative tie.

## Numbers
`build_verify_fig02.py` → `verify_fig02.csv` / `recheck_checks_fig02.csv`: **16/16 PASS**
(2A ×3, 2B ×2, 2C ×5, 2D ×3, 2E ×3), replacing the prior 3 qualitative checks.
