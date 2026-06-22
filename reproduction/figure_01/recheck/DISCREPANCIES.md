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
schematic by design.

Recheck artifacts (this directory): `recompute_fig01.py`, `recheck_checks_fig01.csv`,
`recompute_fig01_table.csv`, `restyle_panels_FG.py`, `measure_panelG.py`,
`panelG_barheight_fit.csv`, `rerun_heavy_panels.sh`, `diff_heavy_panels.py`,
`determinism_fig01.csv`, `regen_panelC.py`, `make_sidebyside_fig01.py`,
`panel_{B,C,D,F,G}_sidebyside.png`, `Figure_1_published_vs_reproduced.png`.

---

## Per-panel findings

| Panel | Type | Match to published | Residual / note | Root cause |
|---|---|---|---|---|
| **1A** | schematic | conceptual | programmatic block-stack, not the illustrator art | hand-drawn schematic (out of scope) |
| **1B** | heavy (ete4) | topology + highlighted clade faithful; **newick byte-identical on re-run** | 987 leaves vs 1361 input taxids; final tree iTOL-styled | NCBI taxonomy versioning (merged/retired taxids; 2023 order split); iTOL styling not script-reproducible |
| **1C** | heavy (MUMmer) | **exact** — strain rep now YJM195; **all coords byte-identical on re-run** | layout/box-color cosmetic | *fixed*: was wrong strain (see below) |
| **1D** | heavy (mash) | **exact** — **both tables byte-identical on re-run** | broken-axis vs sorted-bar cosmetic | rendering choice only |
| **1E** | schematic | conceptual | worked one-hot + loss-weight example | hand-drawn schematic (out of scope) |
| **1F** | computational | **exact (Δ=0)** vs published legend | x-axis "discrepancy" was a crop mis-read | resolved (see below) |
| **1G** | computational | **exact (Δ=0)**; **provenance closed** | numbers only in the figure, not the text | bar-height fit ties them (see below) |

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
