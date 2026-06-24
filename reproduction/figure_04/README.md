# Figure 4 — Promoter & splicing motifs learned during pretraining

> *"Shorkie uses promoter and splicing motifs learned during pretraining."*

Reproduction package for **main-text Figure 4**. Published reference: [`../../paper/Figures/Figure_4.pdf`](../../paper/Figures/Figure_4.pdf) (`published/Figure_4_full.png`).

- **Reproduce:** [`fig04_promoter_splicing_motifs.ipynb`](../../notebooks/fig04_promoter_splicing_motifs.ipynb) delegates to the `recheck/build_4*.py` builders.
- **Verify:** `reproduced/verify_fig04.csv` (correctness + panel-completeness checks).

The ISM/LM panels are reproduced from cached numeric data (ISM `scores.h5`, LM `preds.npz`,
modisco `.h5`, the yeast motif database); the only GPU step is a single re-run of the
**MMS2 Random-Init** ISM (its window was never in the released set — see below).

> **Exact-match ISM pass (this session).** The per-gene ISM saliency logos are made to match
> the published figure **exactly**: the published windows, the published ~33:1 per-row aspect,
> and **all three model rows** (Shorkie LM / Shorkie ISM / Shorkie Random-Init ISM) where the
> published shows them (panels A/B/C/F; E/G are ISM-only). The logos are rendered **clean** —
> the red/purple TF & splice boxes, TF/splice text, TSS dividers, "450/50 nt" labels and
> Reference-DB insets (manual Illustrator overlays) are **removed** at the user's request,
> mirroring the original `2_modisco_DNA_logo.py --no_motif_annotation` mode; the gene-model
> track is kept. **Root cause of the prior mismatch:** the reproduction read the wrong
> `scores.h5` subdirectories — FUN12 used `_RRB` (+124 bp) instead of `_TSS` (exact), MMS2 used
> the `_SS` gene-body window (57% coverage) instead of the `_TSS` 500 bp window, and the
> FUN12/KRE33/MMS2 Random-Init rows were wrongly believed absent (they exist in the `_TSS` /
> `_TSS_select` subs; MMS2's was recomputed on GPU). All builders are driven by the unified
> `fig4_common.PANELS` registry. See [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md).
> Panels **D** and **H** are unchanged (12-TF grid paired with TomTom — MEME 5.5.7 from source).

---

## Phase 1 — Discovery

Figure 4 shows that **Shorkie's in-silico-mutagenesis (ISM) saliency** recovers (i) ribosomal-protein/RRB **promoter** motifs and (ii) canonical **splicing** signals — knowledge acquired during masked-LM pretraining. 8 panels (A–H); D is a literature schematic.

Rows per panel are data-driven (`fig4_common.PANELS`): A/B/C/F = LM + ISM + Random-Init; E/G = ISM only.

| Panel | Claim | Rows | Published window = exact Shorkie-ISM entry | Random-Init entry |
|---|---|---|---|---|
| **A** | promoter ISM, RPL26A | LM/ISM/Random | chrXII:818,862-819,362 = `_RP` part4 idx10 | `_RP` p4 i10 |
| **B** | promoter ISM, FUN12 | LM/ISM/Random | chrI:75,977-76,477 = `_TSS` part0 idx39 | `_TSS` p0 i39 |
| **C** | promoter ISM, KRE33 | LM/ISM/Random | chrXIV:374,871-375,371 = `_RRB` part11 idx3 (−1 bp) | `_TSS_select` p0 i5 |
| **D** | canonical splicing motifs (donor GTATGT / branch TACTAAC / 3′ YAG) | schem | — (literature consensus) | — |
| **E** | splicing ISM, DTD1 | ISM only | chrIV:65,235-65,431 ⊂ `_SS` part22 idx0 (crop) | — |
| **F** | promoter+splicing ISM, MMS2/MAD1 locus | LM/ISM/Random | chrVII:346,669-347,169 = `_TSS` part33 idx31 | **GPU-recomputed** (`run_mms2_random_ism.sh`) |
| **G** | splicing ISM, HOP2 | ISM only | chrVII:435,625-436,401 ⊂ `_SS` part80 idx0 (crop) | — |
| **H** | TF-MoDISco motifs on Shorkie ISM | grid | RP modisco h5 | — |

Source: the ISM/LM saliency recipe of `04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/`
+ `…/shorkie_lm/motif_analysis/motif_lm__RP_TSS/2_modisco_DNA_logo.py`, unified in
`recheck/fig4_common.py`. ISM tree `motif_shorkie_RP_TSS` (`results.ism_scores`); Random-Init
tree `motif_random_init_RP_TSS`; LM preds `motif_LM_RP_TSS/.../eval_{RP,TSS}`.

**ISM saliency** = per-base `logSED` averaged over the **384 T0 RNA-seq tracks** (`track_offset=1148` into the 3053-track RNA-seq subset), mean-centered across the 4 bases, then projected onto the reference one-hot — the per-position attribution rendered as a logomaker logo. The **Shorkie** ISM lives in the `motif_shorkie_RP_TSS` tree; the **Shorkie_Random_Init** ISM in the parallel `motif_random_init_RP_TSS` tree. Reuses `shorkie.helpers.yeast_helpers.make_seq_1hot`; helper logic from the Shorkie ISM analysis (`motif_shorkie__RP_TSS`).

### Config fix applied this phase
`results.ism_scores` and `results.modisco_ism` were corrected (in `config/paths.example.yaml`) to resolve to the real on-disk trees under `${experiments_root}/SUM_data_process/motifs` (the previous `motif_LM_fine_tuned_RP_TSS` value pointed at an empty path — the root cause of the earlier un-executed ISM reproduction).

### Window audit (exact-match)
Every model row is driven by `fig4_common.PANELS` and its window verified against the published PDF
title by reading the on-disk `scores.h5` `chr/start/end`: RPL26A / FUN12 / MMS2 Shorkie-ISM windows
match **to the base** (offset 0); KRE33 is **−1 bp** (the only exact-window finetuned entry; sub-base,
aligned by genomic coordinate); DTD1 / HOP2 published windows sit fully inside the larger `_SS` windows
(cropped, 100% coverage). The LM and Random-Init windows match exactly for every 3-model panel.

---

## Phase 3 — Verification

**`reproduced/verify_fig04.csv`: 41/41 PASS.** Per-panel checks confirm the exact windows, the three
model rows (where the published shows them), and the sharply-localized Shorkie signal:

1. **Window match:** Shorkie-ISM offset 0 for RPL26A/FUN12/MMS2, |Δ|≤1 for KRE33, full coverage (≥0.99)
   for all six. PASS.
2. **Three model rows:** A/B/C/F each render Shorkie LM + Shorkie ISM + Shorkie Random-Init (E/G are
   ISM-only by design); MMS2 (F) is confirmed a 3-model panel. PASS.
3. **Localized ISM signal & Shorkie > Random-Init** (peak/median per-site |saliency|):

| Panel | gene | Shorkie ISM | Shorkie LM | Random-Init | Shorkie>Random |
|---|---|---|---|---|---|
| 4A | RPL26A | **16.3×** | 7.1× | 7.2× | ✓ |
| 4B | FUN12 | **15.6×** | 4.3× | 6.2× | ✓ |
| 4C | KRE33 | **28.0×** | 7.4× | 6.4× | ✓ |
| 4E | DTD1 | **15.5×** | — | — | (ISM-only) |
| 4F | MMS2 | **23.0×** | 7.9× | 14.1× | ✓ |
| 4G | HOP2 | **17.5×** | — | — | (ISM-only) |

4. **4H:** 105 TF-MoDISco patterns (≥6); 12/12 reconstruction TFs TomTom-matched. PASS.
5. **Clean ISM grid:** 14 logos, uniform box, all rows ≥0.99 coverage, all ISM localization ≥5×. PASS.

### Resolved residuals (vs the prior reproduction)
| Item | Resolution |
|---|---|
| FUN12 window +124 bp | wrong sub — switched `_RRB` p15 i3 → **`_TSS` p0 i39** (exact). |
| MMS2 "57% coverage" / ISM-only | MMS2 panel F is a 3-model promoter-style panel at the **MMS2/MAD1 locus**; switched `_SS` gene-body → **`_TSS` p33 i31** (exact); LM row sliced to the exact window; Random-Init **recomputed on GPU**. |
| "no Random_Init for B/C" | wrong — Random-Init exists in the `_TSS`/`_TSS_select` subs (FUN12 i39, KRE33 i5). Recovered. |
| red boxes / TF & splice text / TSS divider / 450-50 labels / Reference-DB insets | **removed** at user request — clean logos (the original `--no_motif_annotation` product); gene-model track kept. |

**Changes to legacy scripts:** none edited. The reproduction builders were refactored to the
`fig4_common.PANELS` registry + the shared clean `render_ism_panel`; `compare.py`'s `mode` field is reused.

---

## Scripts & regeneration

All builders live in [`recheck/`](recheck/) and import shared recipes from `fig4_common.py`
(`PANELS` registry, ISM/LM saliency, the clean `render_ism_panel`, DB-motif IC logos, gene models)
and the TF pairings from `match_tfs.py`. They resolve every path through `shorkie.config` — no
hardcoded machine paths. Build order:

0. `run_mms2_random_ism.sh` *(GPU, one-time)* — recomputes the MMS2 Random-Init ISM on the exact
   published window (`hound_ism_bed.py` + the scratch checkpoint + a 1-row BED). Writes
   `…/motif_random_init_RP_TSS/gene_exp_motif_test_MMS2_panel/f0c0/part0/scores.h5`. The result is
   loaded like any other on-disk entry; the rest of the build is CPU-only.
1. `run_tomtom.py` *(optional — needs MEME `tomtom`; `tomtom_RP_matches.tsv` is committed)*.
2. `build_4ABC.py`, `build_4EFG.py` — clean per-gene ISM panels (LM/ISM/Random rows + gene-model
   track, exact windows, ~33:1 aspect, no boxes/text) via `F.render_ism_panel`. `build_4D.py`,
   `build_4H.py` — panels D/H (unchanged).
3. `build_4_ism_grid.py` — the clean uniform ISM grid (see below) → `Figure_4_ISM_grid_reproduced.png`.
4. `build_verify_fig04.py` — recomputes the checks → `reproduced/verify_fig04.csv`.
5. `make_sidebyside_fig04.py` — published-vs-reproduced comparison PNGs (verification aid).

The notebook [`fig04_promoter_splicing_motifs.ipynb`](../../notebooks/fig04_promoter_splicing_motifs.ipynb) invokes builders 2–4 in order (after the one-time GPU step 0).

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
| A | RPL26A | chrXII:818,862-819,362 | LM · ISM · Random-Init | 100% |
| B | FUN12 | chrI:75,977-76,477 | LM · ISM · Random-Init | 100% (`_TSS` idx39) |
| C | KRE33 | chrXIV:374,871-375,371 | LM · ISM · Random-Init | ~100% (`_RRB` idx3, −1 bp) |
| E | DTD1 | chrIV:65,235-65,431 | ISM | 100% (crop of `_SS`) |
| F | MMS2 | chrVII:346,669-347,169 | LM · ISM · Random-Init | 100% (`_TSS` idx31; Random GPU-recomputed) |
| G | HOP2 | chrVII:435,625-436,401 | ISM | 100% (crop of `_SS`) |

> **14 logos total** (A/B/C/F = 3 model rows each; E/G = ISM only). Every row is full-coverage
> after the sub-fix. The only re-run is MMS2's Random-Init row (GPU); all other rows are
> precomputed (`scores.h5` / `preds.npz` caches).
