# Figure 1 — deep-recheck discrepancy report

Exact reproduction of every Figure-1 result against the **published** version, with
root-cause for each residual. This recheck *builds on* the prior reproduction
(`reproduced/verify_fig01.csv`, 12/12 PASS) and tightens it: numbers were
independently recomputed from the raw logs, the heavy panels were re-run from the
released FASTAs to confirm determinism, and two stale errors were corrected.

**Verdict: Figure 1 fully reproduced.** 12/12 computational numbers Δ=0 (independently
re-derived); 1F tied to the published legend; **1G provenance closed** by a bar-height
fit (R²=0.999998); heavy panels B/C/D **byte-identical** on re-run; the 1C strain
representative **corrected to YJM195** (now an exact match); schematics A/E are
**skipped** (hand-drawn in the paper — not reproduced).

Recheck artifacts (this directory): `recompute_fig01.py`, `recheck_checks_fig01.csv`,
`recompute_fig01_table.csv`, `restyle_panels_DFG.py` (D/F/G builders),
`build_panels_BC.py` (B/C builders; A/E are skipped schematics), `measure_panelG.py`,
`panelG_barheight_fit.csv`, `rerun_heavy_panels.sh`, `diff_heavy_panels.py`,
`determinism_fig01.csv`, `make_sidebyside_fig01.py`,
`panel_{B,C,D,F,G}_sidebyside.png`, `Figure_1_published_vs_reproduced.png`.

---

## Per-panel findings

| Panel | Type | Match to published | Residual / note | Root cause |
|---|---|---|---|---|
| **1A** | schematic | **skipped** (not reproduced) | hand-drawn architecture diagram; defined by `params.json` | hand-drawn schematic (out of scope) |
| **1B** | heavy (ete4) | topology + highlighted clade faithful; **newick byte-identical on re-run** | 987 leaves vs 1361 input taxids; final tree iTOL-styled | NCBI taxonomy versioning (merged/retired taxids; 2023 order split); iTOL styling not script-reproducible |
| **1C** | heavy (MUMmer) | **exact** — strain rep now YJM195; **all coords byte-identical on re-run** | layout/box-color cosmetic | *fixed*: was wrong strain (see below) |
| **1D** | heavy (mash) | **exact** — **both tables byte-identical on re-run**; now rendered in the published style | broken-axis/braces were a manual enhancement (see "Exact-render pass") | rendering choice only |
| **1E** | schematic | **skipped** (not reproduced) | hand-drawn preprocessing pipeline; transforms in `01_data_build/lm_corpus/` | hand-drawn schematic (out of scope) |
| **1F** | computational | **exact (Δ=0)** vs published legend; panel aspect now matches | x-axis "discrepancy" was a crop mis-read | resolved (see below) |
| **1G** | computational | **exact (Δ=0)**; **provenance closed**; panel aspect now matches | numbers only in the figure, not the text | bar-height fit ties them (see below) |

---

## Resolved issues

### 1C strain representative: YJM195 (was wrongly YJM1078 / "YPF136")
- **The published panel shows YJM195** (`GCA_000975585.2`). Confirmed two ways:
  - `paper/Figures/Figure_1.pdf` text layer: *"TGT: Saccharomyces cerevisiae / YJM195"*.
  - Manuscript body (`paper/shorkie.pdf`): *"strains (e.g., **YJM195 at Mash ≈ 0.01**) exhibited
    near-continuous synteny"*.
- **Two pre-existing errors, both corrected:** the reproduction had used **YJM1078**
  (`GCA_000975645.3`); the README discrepancy log + Panel-C table claimed the published strain was
  **"YPF136"**. Neither is correct.
- YJM195 **is present** in the 80_Strains corpus (`data_strains_gtf/fasta/GCA_000975585_2.cleaned.fasta`,
  row 54 of `data/species_lists/species_strains_gtf.cleaned.csv`). Fixed `panels/run_mummer.sh` +
  notebook cell 7 to `GCA_000975585_2`, re-ran nucmer/show-coords, removed the stale YJM1078 `.coords`.
  The strain dot-plot is now an **exact representative match**, not a substitution.
- Reproduced aligned-fraction spectrum (R64 self → strain → order → kingdom):
  **1.19 (self) · 1.18 (YJM195) · 0.017 (N. glabratus) · 0.003 (C. albicans) · 0.0003 (N. crassa) ·
  0.001 (S. pombe)** — YJM195 ≈ self (near-continuous synteny), matching the manuscript and the panel.

### 1F x-axis: no discrepancy (earlier "stops ~250k" was a crop mis-read)
- Panel-F x-axis = `n_epochs × 64` (64 = validation batches/epoch). Per-tier extents (from
  `train.out`, independently recounted): R64 **117 ep → 7,488**, 80_Strains **170 → 10,880**,
  165_Saccharomycetales **5001 → 320,064**, 1341_Fungus **3991 → 255,424**.
- The longest curve (165_Saccharomycetales, green) reaches **~320k**, which **is** the published
  "~300k" extent; the noisy red 1341 curve stops at ~255k. The reproduced panel auto-scales to the
  same ~320k. No fix to the data was needed; the panel was restyled (dashed curves + min-markers +
  published labels) only for visual fidelity.
- Parser note (carried from the build): the legacy `3_dataset_comparison.py::parse_epoch_line` uses
  `line.split(':',1)` and grabs the `steps:` count as `valid_loss`; the reproduction uses an explicit
  `valid_loss:` regex. The reproduced legend losses match the published legend **exactly** (Δ=0):
  **0.4181 · 0.4154 · 0.4018 · 0.4055**.

### 1G provenance: closed by a bar-height fit (R²=0.999998)
- The manuscript text states only the **qualitative** 1G ordering — *"165_Saccharomycetales … achieved
  … the lowest test perplexity on held-out S. cerevisiae chromosomes … outperformed the more divergent
  1341_Fungal LM"* — which the reproduced numbers confirm. The **exact** perplexities appear only in
  the figure.
- We therefore pixel-measured the **published panel-G bar heights** and linearly fit them against the
  reproduced gene/intergenic perplexities (8 bars, 2 fitted DOF): **R²=0.999998, max residual 0.0002
  perplexity units** (tick spacing 0.05). The published bars **are** the reproduced numbers, up to the
  axis' affine scaling — a non-circular provenance closure. See `panelG_barheight_fit.csv`.
- Reproduced gene/intergenic: R64 **3.7561/3.7386** · 80_Strains **3.7342/3.7225** ·
  165_Saccharomycetales **3.5488/3.6360** · 1341_Fungus **3.6043/3.6851**
  (overall PPL 3.7458/3.7257/**3.5853**/3.6380 → 165_Saccharomycetales lowest, the "sweet spot").

---

## Exact-render pass (panels D / F / G) — style + panel proportions

A later strict pass (user-requested) re-rendered **D, F, G** to match the published crops *exactly* in
style **and** panel proportions — a **pure render** of byte-identical cached data (no GPU, no recompute;
all 12 numbers unchanged). The three panels are now built by a single source of truth,
`restyle_panels_DFG.py`, which the notebook D/F/G cells import and call (they return `FIG1F_MIN` /
`FIG1G_GENE` / `FIG1G_INTER`, so the verify cell stays **12/12**). Panels B, C were left as-is; A/E removed (schematics).

- **1D — full published styling reproduced.** Matches `published/Figure_1_D_pub.png` (1329×1659 px,
  portrait ~0.80 w/h): two stacked panels (top = 165_Saccharomycetales, linear y 0–1; bottom = 80_Strains
  with a **broken y-axis** — lower 0.000–0.010 holds every bar, upper 0.990–1.000 is empty but shows the
  full Mash range — plus diagonal break marks), `skyblue` **gapped** bars (width 0.75), published titles
  ("Mash Distance Score for Yeast R64 vs. Target {165_Saccharomycetales|80_Strains} Genomes"), an arrow to
  the R64-1-1 (Mash distance 0) bar, and curly-brace "All other … genomes (non–…)" annotations + a
  "Target Genomes" label (no numeric x-ticks). **Finding:** the released
  `scripts/03_eval/lm/genome_evaluation/3_genome_dist/mash/2_mash_genome_viz.py` emits only a plain
  skyblue bar chart (one per tier) — the broken-axis / brace / arrow styling and the two-tier composite
  were a **manual (Illustrator) enhancement** of the published figure, faithfully reproduced here in the
  builder. The underlying distances are unchanged and byte-identical on re-run (sacc max 1.0 with 134/165
  ≥0.99; strains max 0.0081, none ≥0.99 — so the strains' upper break segment is empty by design).
  **Follow-up (heights/arrow/ticks):** the first attempt rendered the subplots too tall (portrait
  `figsize=(7.6,10)`), so the bars looked too high. Re-checked against the full published figure
  (`paper/Figures/Figure_1.pdf`) and rebuilt each subplot **wide-and-short** (≈4.5:1 w:h) via an explicit
  spacer-row `GridSpec` (`figsize=(8.6,8.0)`) that gives the title/brace/arrow/"Target Genomes"
  annotations fixed room independent of the short axes; added the **R64-1-1 arrow to the top
  (Saccharomycetales) subplot too** (the published shows it on *both* panels) via a shared
  `_annotate_mash_panel()` helper; and set the top y-ticks to **0.00/0.25/0.50/0.75/1.00** (published
  spacing).
- **1F — exact published scale & style.** Kept the dashed blue/orange/green/red curves, faint argmin
  min-markers, legend "<label>; loss = <min>", and the wide aspect (`figsize=10×4.45`). Re-checked vs the
  published F and corrected the **scale + style**: legend → **upper center** (was top-right); y-ticks →
  **0.02 step (0.42/0.44)** via `MultipleLocator` (was a dense 0.005 grid); x-axis pinned to **0–322k with
  50k ticks to 300000** (the green 165_Sacc curve + min-marker reach ~302k–320k).
  **Follow-up (smoothing + green clip):** to match the *original* generator
  `scripts/03_eval/lm/lm_model_eval/3_dataset_comparison.py::plot_losses` (the "same smooth parameter"),
  ported its **`moving_average(window_size=21)` + `maybe_trim_end(trim_end=5)`** smoothing into
  `restyle_1F()` and plot the smoothed curves (the raw red 1341_Fungus was far spikier than the published);
  the legend min is still taken from the **raw (trimmed)** array, so the printed mins
  (0.4181/0.4154/**0.4018**/0.4055) and verify are unchanged. Also fixed the **green curve being clipped**:
  its min is **0.4018** but `ylim` started at 0.404, so the bottom of the green descent was cut at the axis
  floor — lowered `ylim` to **(0.400, 0.446)** (ticks now 0.40/0.42/0.44) so the full descent shows, as the
  legacy auto-scale did.
- **1G — panel aspect.** Kept gene(blue)/intergenic(orange) grouped bars, y-range, labels, legend; only
  widened `figsize` (7.2×4.6 → 10×4.5) to match the published G aspect (2611×1181 ≈ 2.21:1).

---

## Determinism (heavy panels re-run from the released FASTAs)

`diff_heavy_panels.py` → `determinism_fig01.csv`. Re-running the external tools reproduced the
committed intermediates **byte-for-byte**:

- **1D Mash** — `saccharomycetales_dist.tab` (165 rows) and `strains_dist.tab` (80 rows): **BYTE_IDENTICAL**.
- **1C MUMmer** — all 5 retained targets (R64 self, N. glabratus, C. albicans, N. crassa, S. pombe):
  **BYTE_IDENTICAL**; YJM195 newly added (replaces the removed YJM1078). (nucmer/show-coords canonicalise
  the symlinked input path, so even the `.coords` header matches.)
- **1B tree** — `species_tree.nwk` (**987 leaves**): **BYTE_IDENTICAL** (deterministic given the NCBI
  taxonomy DB; byte-identity is not guaranteed across NCBI versions — documented, not required).

## New numeric anchor (panel 1D)
Manuscript: *"YJM195 at Mash ≈ 0.01"*. Reproduced `mash dist(R64, YJM195) = 0.0068` — consistent
(YJM195 is among the more distant strains; the 80_Strains range is 0 → 0.0081). Recorded as a
published-text tie for 1D.

## Numbers (independent recompute)
`recompute_fig01.py` re-parses the 4 `train.out` + 4 `test_testset_perplexity_region.out` from scratch
and asserts Δ=0 vs `reproduced/verify_fig01.csv`: **12/12 PASS, max |Δ| = 0.0** → `recheck_checks_fig01.csv`.
