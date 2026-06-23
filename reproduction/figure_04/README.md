# Figure 4 — Promoter & splicing motifs learned during pretraining

> *"Shorkie uses promoter and splicing motifs learned during pretraining."*

Reproduction package for **main-text Figure 4**. Published reference: [`../../paper/Figures/Figure_4.pdf`](../../paper/Figures/Figure_4.pdf) (`published/Figure_4_full.png`).

- **Reproduce:** [`reproduce_figure_04.ipynb`](reproduce_figure_04.ipynb) delegates to the `recheck/build_4*.py` builders.
- **Verify:** `reproduced/verify_fig04.csv` (correctness + panel-completeness checks).

All panels are **CPU-reproducible** from cached numeric data (ISM `scores.h5`, LM `preds.npz`,
modisco `.h5`, the yeast motif database); no GPU is needed to regenerate the figure.

> **Deep recheck (focused upgrade).** The first reproduction passed correctness checks but
> rendered far fewer elements than the published panels (no "Shorkie LM" row, no Reference-DB
> overlay, no gene models/annotations, panel D was text-only, panel H was an unnamed stack).
> The recheck under [`recheck/`](recheck/) closes these gaps: 4-row promoter logos
> (`build_4ABC.py`), splicing schematic + DB/reconstruction (`build_4D.py`), splicing ISM +
> gene models + splice boxes (`build_4EFG.py`), and the 12-TF database-vs-reconstruction grid
> (`build_4H.py`, paired with TomTom — MEME 5.5.7 built from source). See
> [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md) for the per-panel status, root causes,
> and the documented residuals (no SS-MoDISco for panel D; no Random_Init tree for B/C/E–G;
> LM↔ISM window offset; the irreducibly-manual Illustrator composition).

---

## Phase 1 — Discovery

Figure 4 shows that **Shorkie's in-silico-mutagenesis (ISM) saliency** recovers (i) ribosomal-protein/RRB **promoter** motifs and (ii) canonical **splicing** signals — knowledge acquired during masked-LM pretraining. 8 panels (A–H); D is a literature schematic.

| Panel | Claim | Type | Gene → ISM window (coordinate-audited) | Source script | Config key |
|---|---|---|---|---|---|
| **A** | promoter ISM, RPL26A (Shorkie vs Random_Init) | CPU | RP `part4 idx10` = chrXII:818862-819362 | `04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/1_plot_dna_logo_general.py` | `results.ism_scores` |
| **B** | promoter ISM, FUN12 | CPU | RRB `part15 idx3` = chrI:76101-76601 | same | `results.ism_scores` |
| **C** | promoter ISM, KRE33 | CPU | RRB `part11 idx3` = chrXIV:374870-375370 | same | `results.ism_scores` |
| **D** | canonical splicing motifs (5′ donor GTATGT / branch TACTAAC / 3′ YAG) | schem | — (literature consensus) | — | — |
| **E** | splicing ISM, DTD1 | CPU | SS `part22 idx0` = chrIV:65191-65816 | `…/motif_shorkie__RP_TSS/4_plot_dna_SS.py` | `results.ism_scores` |
| **F** | splicing ISM, MMS2 | CPU | SS `part57 idx0` = chrVII:346354-346954 | same | `results.ism_scores` |
| **G** | splicing ISM, HOP2 | CPU | SS `part80 idx0` = chrVII:435573-436451 | same | `results.ism_scores` |
| **H** | TF-MoDISco motifs on Shorkie ISM | CPU | RP modisco h5 | `…/2_modisco_analysis/4_viz_motif.py` | `results.modisco_ism` |

**ISM saliency** = per-base `logSED` averaged over the **384 T0 RNA-seq tracks** (`track_offset=1148` into the 3053-track RNA-seq subset), mean-centered across the 4 bases, then projected onto the reference one-hot — the per-position attribution rendered as a logomaker logo. The **Shorkie** ISM lives in the `motif_shorkie_RP_TSS` tree; the **Shorkie_Random_Init** ISM in the parallel `motif_random_init_RP_TSS` tree. Reuses `shorkie.helpers.yeast_helpers.make_seq_1hot`; helper logic from notebook `fig13`.

### Config fix applied this phase
`results.ism_scores` and `results.modisco_ism` were corrected (in `config/paths.example.yaml`) to resolve to the real on-disk trees under `${experiments_root}/SUM_data_process/motifs` (the previous `motif_LM_fine_tuned_RP_TSS` value pointed at an empty path — the root cause of the earlier un-executed `fig13`).

### Coordinate audit (the careful-verification upgrade)
The gene → `(tree, sub, part, idx)` mapping was **not trusted as hardcoded**. Each of the 6 panel genes (RPL26A, FUN12, KRE33, DTD1, MMS2, HOP2) was resolved symbol → ORF → R64 coordinates from the released GTF (`genome.gtf`), then **every** `scores.h5` window in the RP / RRB / SS subs was enumerated (lazy read of `chr/start/end`) and matched by overlap. The result **confirmed the existing map is correct** (table above), resolving an earlier disagreement between two discovery passes over whether DTD1/MMS2/HOP2 were present in the SS set — they are, at parts 22/57/80. The audit is re-run inside the notebook (`win_overlaps_gene`) and asserted in Phase 3.

---

## Phase 3 — Verification

**`reproduced/verify_fig04.csv`: 25/25 PASS.** Figure 4 is qualitative (saliency maps + motif logos), so each panel is verified two ways (the deep recheck added the panel-completeness checks — "Shorkie LM row rendered" for A/B/C, panel-H TomTom-matched-TF count, panel-D reconstruction rendered; the last 6 checks cover the [clean uniform ISM grid](#clean-uniform-ism-grid-build_4_ism_gridpy) — 10 logos, uniform box size, region coverage, localization):

1. **Coordinate-overlap (correctness):** every panel's ISM window provably overlaps its R64 gene — 6/6 PASS. This is the strong check: it proves the reproduced panel shows the *claimed* gene.
2. **Localized ISM signal (scale-invariant):** the Shorkie saliency peak exceeds **5× the window median** per-site saliency (a concentrated motif, not diffuse noise). Measured localization ratios:

| Panel | gene | max\|saliency\| | localization (peak/median) | verdict |
|---|---|---|---|---|
| 4A | RPL26A | 0.028 | **16.3×** | PASS |
| 4B | FUN12 | 0.004 | **18.2×** | PASS |
| 4C | KRE33 | 0.008 | **28.0×** | PASS |
| 4E | DTD1 | 0.038 | **15.5×** | PASS |
| 4F | MMS2 | 0.033 | **10.0×** | PASS |
| 4G | HOP2 | 0.004 | **17.5×** | PASS |

3. **Shorkie > Random_Init:** at panel A (the only locus present in both trees), Shorkie localization **16.3× > 7.19×** Random_Init — Shorkie concentrates attribution; the from-scratch model is diffuse. PASS.
4. **4H:** **105** TF-MoDISco patterns recovered (≥6). PASS.

### Discrepancy log (honest)
| Item | Note |
|---|---|
| Earlier 3 FAILs (4B/4C/4G, signal = 0.0) | **Two causes, both resolved.** (i) 4B/4C were a *stale-execution* artifact — the RRB `scores.h5` parts were written after the first notebook run, so `ism_viz` returned `None` (file-not-found → 0.0). Re-execution populates them. (ii) 4G HOP2's failure was **not** stale: its absolute `max|saliency|` (0.004) is genuinely low, so the old `>0.01` absolute threshold rejected it. Replacing that fragile threshold with the **localization ratio** (17.5×, the *highest* of all panels — a razor-sharp splice-site peak above near-zero background) is the correct, scale-invariant signal check. |
| Random_Init rows for 4B/4C/4E–G | The `motif_random_init_RP_TSS` tree only contains the **RP** sub on disk — it has no RRB or SS windows. So Shorkie-vs-Random_Init can only be compared at the RP locus (panel A); the FUN12/KRE33/DTD1/MMS2/HOP2 Random_Init rows are labelled "sub not in this tree" rather than fabricated. |
| HOP2 absolute signal | Low (0.004) but **sharply localized** — the figure's claim (Shorkie maps the splice donor) holds; the magnitude is simply smaller than DTD1/MMS2. |

**Changes to legacy scripts:** none edited. The verification helper `reproduction/common/compare.py` gained a backward-compatible `mode` field (`ge`/`gt`/`le`) so threshold checks (localization ≥ 5×, motif count ≥ 6) read cleanly; existing figures are unaffected.

---

## Scripts & regeneration

All builders live in [`recheck/`](recheck/) and import shared recipes from `fig4_common.py`
(ISM/LM saliency, DB-motif IC logos, gene models) and the TF pairings from `match_tfs.py`. They are
CPU-only and resolve every path through `shorkie.config` — no hardcoded machine paths. Build order:

1. `run_tomtom.py` *(optional — needs the MEME `tomtom` binary on `PATH` or via the `tools.tomtom_bin`
   config key)* — regenerates `tomtom_RP_matches.tsv` (MoDISco pattern → yeast-TF matches). **Already
   committed**, so the rest runs without MEME installed.
2. `build_4ABC.py`, `build_4D.py`, `build_4EFG.py`, `build_4H.py` — render the panel PNGs into
   `reproduced/` and write the per-panel metric CSVs.
3. `build_4_ism_grid.py` — the clean uniform ISM grid (see below) → `Figure_4_ISM_grid_reproduced.png`
   + `fig4_ism_grid_metrics.csv`.
4. `build_verify_fig04.py` — recomputes the checks → `reproduced/verify_fig04.csv`.
5. `make_sidebyside_fig04.py` — published-vs-reproduced comparison PNGs (verification aid).

The notebook [`reproduce_figure_04.ipynb`](reproduce_figure_04.ipynb) simply invokes builders 2–3 in order.

**Committed vs regenerable.** Committed: the builders + `fig4_common.py` / `match_tfs.py`, the reproduced
PNGs, `verify_fig04.csv`, and `tomtom_RP_matches.tsv` (so panel H reproduces without MEME). Gitignored
caches, regenerated on demand: `lm_cache_*.npz` (LM per-row slices), `_modisco_RP_query.meme` +
`_tomtom_RP/` (TomTom intermediates), and `__pycache__/`.

---

## Clean uniform ISM grid (`build_4_ism_grid.py`)

A single annotation-free view — `reproduced/Figure_4_ISM_grid_reproduced.png` — that re-plots **every
per-base saliency logo Figure 4 is built from**, on a deliberately uniform footing. It uses only the
**precomputed** scores (ISM `scores.h5`, LM `preds.npz` via the per-row caches) — **no ISM re-run, no
GPU**. Three rules:

1. **Uniform scale** — every logo is drawn in the *identical physical box* (11.0 × 0.85 in). Because the
   panel windows differ in length (196 bp – 776 bp), bp-per-inch varies (letters look wider in short
   windows). Verified by `uniform_box_size` in the verify CSV.
2. **Exact region match** — each logo's x-axis is the *published* panel window. Where the released ISM
   window is offset from the paper's, only the covered intersection is drawn and the covered fraction is
   recorded (the left flank is left blank; rows still align on genomic coordinate).
3. **All saliency rows that exist** — Shorkie LM + Shorkie ISM + Shorkie Random_Init, included only where
   the precomputed data is on disk. No Reference-DB, gene models, TF boxes, splice boxes, dividers or
   curated text — just the logos + a minimal `gene · source` label and the genomic coordinates.

| Panel | Gene | Published window | Rows rendered | ISM coverage |
|---|---|---|---|---|
| A | RPL26A | chrXII:818,862-819,362 | LM · ISM · Random_Init | 100% |
| B | FUN12 | chrI:75,977-76,477 | LM · ISM | **75%** (ISM data is chrI:76,101-76,601, ~124 bp offset) |
| C | KRE33 | chrXIV:374,871-375,371 | LM · ISM | ~100% |
| E | DTD1 | chrIV:65,235-65,431 | ISM | 100% |
| F | MMS2 | chrVII:346,669-347,169 | ISM | **57%** (ISM data is chrVII:346,354-346,954) |
| G | HOP2 | chrVII:435,625-436,401 | ISM | 100% |

> **Residual (not fabricated).** The published B/C panels show a Random_Init row and the published F panel
> shows LM + Random_Init rows, but those came from data not in the released registry (`motif_random_init_RP_TSS`
> has only the **RP** sub; LM `preds.npz` has no `eval_SS`). The grid plots only the saliency rows whose
> data is actually on disk, so FUN12/KRE33 show LM+ISM and the splice genes show ISM only.
