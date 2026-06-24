# Figure 4 — Promoter & splicing motifs learned during pretraining

> *"Shorkie uses promoter and splicing motifs learned during pretraining."*

Reproduction package for **main-text Figure 4**. Published reference: [`../../paper/Figures/Figure_4.pdf`](../../paper/Figures/Figure_4.pdf) (`published/Figure_4_full.png`).

- **Reproduce:** [`fig04_promoter_splicing_motifs.ipynb`](../../notebooks/fig04_promoter_splicing_motifs.ipynb) delegates to the `recheck/build_4*.py` builders.
- **Verify:** `reproduced/verify_fig04.csv` (correctness + panel-completeness checks).

The ISM/LM panels are reproduced from cached numeric data (ISM `scores.h5`, LM `preds.npz`); the
only GPU step is a one-time re-run of the four **Shorkie_Random_Init** ISM rows with the published
random-init checkpoint (`run_fig4_random_ism.sh`).

> **Exact-match ISM pass.** The per-gene ISM saliency logos match the published figure **exactly**:
> the published windows, the published ~33:1 per-row aspect, and **all three model rows**
> (Shorkie LM / Shorkie ISM / Shorkie Random-Init ISM) where the published shows them (panels
> A/B/C/F; E/G are ISM-only). Logos are **clean** — the red/purple TF & splice boxes, TF/splice
> text, TSS dividers, "450/50 nt" labels and Reference-DB insets (manual Illustrator overlays) are
> **removed**, mirroring the original `2_modisco_DNA_logo.py --no_motif_annotation`; the gene-model
> track is kept. **Corrections that made the rows exact:**
> - **Shorkie-ISM window** — FUN12 `_RRB` (+124 bp) → `_TSS` p0 i39 (exact); MMS2 `_SS` gene-body
>   (57%) → `_TSS` p33 i31 (exact, the MMS2/MAD1 locus); RPL26A/KRE33/DTD1/HOP2 unchanged.
> - **Shorkie_Random_Init model** — the published random-init row is the **lr=5e-4** scratch
>   checkpoint (`supervised_unet_small_bert_drop_variants/learning_rate_0.0005`, pinned as
>   `Shorkie_Random_Init` in `scripts/.../2_lr_search/{2,3}_compare_lr_variants_*.py`). The released
>   `motif_random_init_RP_TSS` ISM was the **old lr=1e-4** model, so all four random rows
>   (A/B/C/F) were recomputed on GPU with the lr=5e-4 checkpoint.
> - **Panel F Shorkie-LM** — uses the **MAD1** eval_TSS row (2176; +strand), whose 450/50 window is
>   exactly chrVII:346,669-347,169; the prior MMS2 row (2175, −strand) gave the wrong 16,384 bp
>   masked-LM context.
> All builders are driven by the unified `fig4_common.PANELS` registry. **Panels D & H are removed**
> from the reproduction (their `build_4D.py`/`build_4H.py` scripts are kept on disk, not rendered).
> See [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md).
> Panels **D** and **H** are unchanged (12-TF grid paired with TomTom — MEME 5.5.7 from source).

---

## Phase 1 — Discovery

Figure 4 shows that **Shorkie's in-silico-mutagenesis (ISM) saliency** recovers (i) ribosomal-protein/RRB **promoter** motifs and (ii) canonical **splicing** signals — knowledge acquired during masked-LM pretraining. The reproduction renders the saliency panels **A/B/C/E/F/G**; panels **D** (schematic + reconstruction) and **H** (12-TF grid) are **removed** (scripts kept on disk, not rendered).

Rows per panel are data-driven (`fig4_common.PANELS`): A/B/C/F = LM + ISM + Random-Init; E/G = ISM only.
All four Random-Init rows come from one recomputed `scores.h5` (`gene_exp_motif_test_fig4_lr5e4`, idx 0-3).

| Panel | Claim | Rows | Published window = exact Shorkie-ISM entry | Shorkie-LM row | Random-Init |
|---|---|---|---|---|---|
| **A** | promoter ISM, RPL26A | LM/ISM/Random | chrXII:818,862-819,362 = `_RP` part4 idx10 | eval_RP r90 | lr5e4 idx0 |
| **B** | promoter ISM, FUN12 | LM/ISM/Random | chrI:75,977-76,477 = `_TSS` part0 idx39 | eval_TSS r39 | lr5e4 idx1 |
| **C** | promoter ISM, KRE33 | LM/ISM/Random | chrXIV:374,871-375,371 = `_RRB` part11 idx3 (−1 bp) | eval_TSS r5134 | lr5e4 idx2 |
| **E** | splicing ISM, DTD1 | ISM only | chrIV:65,235-65,431 ⊂ `_SS` part22 idx0 (crop) | — | — |
| **F** | promoter+splicing ISM, MMS2/MAD1 locus | LM/ISM/Random | chrVII:346,669-347,169 = `_TSS` part33 idx31 | **eval_TSS r2176 (MAD1)** | lr5e4 idx3 |
| **G** | splicing ISM, HOP2 | ISM only | chrVII:435,625-436,401 ⊂ `_SS` part80 idx0 (crop) | — | — |

`Random-Init = lr5e4 idxN` → `motif_random_init_RP_TSS/gene_exp_motif_test_fig4_lr5e4/f0c0/part0`
scores.h5, recomputed by `recheck/run_fig4_random_ism.sh` with the **lr=5e-4** scratch checkpoint
(`supervised_unet_small_bert_drop_variants/learning_rate_0.0005`).

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
(cropped, 100% coverage). The LM and Random-Init windows match exactly for every 3-model panel — incl.
panel F, where the LM row is **MAD1** (eval_TSS r2176, +strand) so its 450/50 window is the published
chrVII:346,669-347,169 (the MMS2 −strand row gave the wrong masked-LM context).

---

## Phase 3 — Verification

**`reproduced/verify_fig04.csv`: 38/38 PASS** (panels D & H removed → their 3 checks dropped). Per-panel
checks confirm the exact windows, the three model rows (where the published shows them), and the
sharply-localized Shorkie signal:

1. **Window match:** Shorkie-ISM offset 0 for RPL26A/FUN12/MMS2, |Δ|≤1 for KRE33, full coverage (≥0.99)
   for all six. PASS.
2. **Three model rows:** A/B/C/F each render Shorkie LM + Shorkie ISM + Shorkie Random-Init (E/G are
   ISM-only by design); MMS2 (F) is confirmed a 3-model panel. PASS.
3. **Localized ISM signal & Shorkie > Random-Init** (peak/median per-site |saliency|; Random-Init = the
   lr=5e-4 scratch model):

| Panel | gene | Shorkie ISM | Shorkie LM | Random-Init | Shorkie>Random |
|---|---|---|---|---|---|
| 4A | RPL26A | **16.3×** | 7.1× | 9.2× | ✓ |
| 4B | FUN12 | **15.6×** | 4.3× | 6.4× | ✓ |
| 4C | KRE33 | **28.0×** | 7.4× | 8.7× | ✓ |
| 4E | DTD1 | **15.5×** | — | — | (ISM-only) |
| 4F | MMS2 | **23.0×** | 8.3× | 7.9× | ✓ |
| 4G | HOP2 | **17.5×** | — | — | (ISM-only) |

4. **Clean ISM grid:** 14 logos, uniform box, all rows ≥0.99 coverage, all ISM localization ≥5×. PASS.

### Resolved residuals (vs the prior reproduction)
| Item | Resolution |
|---|---|
| FUN12 window +124 bp | wrong sub — switched `_RRB` p15 i3 → **`_TSS` p0 i39** (exact). |
| MMS2 "57% coverage" / ISM-only | MMS2 panel F is a 3-model promoter-style panel at the **MMS2/MAD1 locus**; switched `_SS` gene-body → **`_TSS` p33 i31** (exact). |
| MMS2 Shorkie-LM wrong context | used MMS2 row (2175, −strand) → switched to **MAD1 row 2176** (+strand); 450/50 window = the published window exactly. |
| Shorkie_Random_Init model | the released ISM used lr=1e-4; the published figure uses the **lr=5e-4** scratch checkpoint (`learning_rate_0.0005`) — **all four** random rows recomputed on GPU (`run_fig4_random_ism.sh`). |
| red boxes / TF & splice text / TSS divider / 450-50 labels / Reference-DB insets | **removed** at user request — clean logos (the original `--no_motif_annotation` product); gene-model track kept. |
| panels D & H | **removed** from the reproduction (no figures/verify); `build_4D.py`/`build_4H.py`/`run_tomtom.py`/`match_tfs.py` kept on disk. |

**Changes to legacy scripts:** none edited. The reproduction builders are driven by the
`fig4_common.PANELS` registry + the shared clean `render_ism_panel`; `compare.py`'s `mode` field is reused.

---

## Scripts & regeneration

All builders live in [`recheck/`](recheck/) and import shared recipes from `fig4_common.py`
(`PANELS` registry, ISM/LM saliency, the clean `render_ism_panel`, DB-motif IC logos, gene models)
and the TF pairings from `match_tfs.py`. They resolve every path through `shorkie.config` — no
hardcoded machine paths. Build order:

0. `run_fig4_random_ism.sh` *(GPU, one-time)* — recomputes the **four** Shorkie_Random_Init ISM rows
   (RPL26A/FUN12/KRE33/MMS2) on their exact published windows with the **lr=5e-4** scratch checkpoint
   (`hound_ism_bed.py` + a 4-row BED). Writes `…/motif_random_init_RP_TSS/gene_exp_motif_test_fig4_lr5e4/f0c0/part0/scores.h5`
   (idx 0-3). The result is loaded like any other on-disk entry; the rest of the build is CPU-only.
1. `build_4ABC.py`, `build_4EFG.py` — clean per-gene ISM panels (LM/ISM/Random rows + gene-model
   track, exact windows, ~33:1 aspect, no boxes/text) via `F.render_ism_panel`.
2. `build_4_ism_grid.py` — the clean uniform ISM grid (see below) → `Figure_4_ISM_grid_reproduced.png`.
3. `build_verify_fig04.py` — recomputes the checks → `reproduced/verify_fig04.csv`.
4. `make_sidebyside_fig04.py` — published-vs-reproduced comparison PNGs (A/B/C + E/F/G).

**Panels D & H are not rendered** but their scripts are kept and runnable: `build_4D.py`,
`build_4H.py`, `run_tomtom.py`, `match_tfs.py`, `tomtom_RP_matches.tsv`.

The notebook [`fig04_promoter_splicing_motifs.ipynb`](../../notebooks/fig04_promoter_splicing_motifs.ipynb) invokes builders 1–3 in order (after the one-time GPU step 0); D/H appear as skip-notes.

**Committed vs regenerable.** Committed: the builders + `fig4_common.py` / `match_tfs.py`, the reproduced
PNGs, `verify_fig04.csv`, and `tomtom_RP_matches.tsv` (so panel H reproduces without MEME). Gitignored
caches, regenerated on demand: `lm_cache_*.npz` (LM per-row slices), `_modisco_RP_query.meme` +
`_tomtom_RP/` (TomTom intermediates), and `__pycache__/`.

---

## Clean uniform ISM grid (`build_4_ism_grid.py`)

A single annotation-free view — `reproduced/Figure_4_ISM_grid_reproduced.png` — that re-plots **every
per-base saliency logo Figure 4 is built from**, on a deliberately uniform footing. It reads only the
on-disk scores (ISM/Random `scores.h5`, LM `preds.npz` via the per-row caches) — no model load
(after the one-time GPU step 0 that produced the lr=5e-4 random `scores.h5`). Three rules:

1. **Uniform scale** — every logo is drawn in the *identical physical box* (11.0 × 0.85 in). Because the
   panel windows differ in length (196 bp – 776 bp), bp-per-inch varies (letters look wider in short
   windows). Verified by `uniform_box_size` in the verify CSV.
2. **Exact region match** — each logo's x-axis is the *published* panel window (cropping the larger
   `_SS` windows for DTD1/HOP2; all rows full coverage).
3. **All three model rows** — Shorkie LM + Shorkie ISM + Shorkie Random-Init for A/B/C/F; ISM only for
   E/G. No Reference-DB, gene models, TF boxes, splice boxes, dividers or curated text — just the logos
   + a minimal `gene · source` label and the genomic coordinates.

| Panel | Gene | Published window | Rows rendered | ISM coverage |
|---|---|---|---|---|
| A | RPL26A | chrXII:818,862-819,362 | LM · ISM · Random-Init | 100% |
| B | FUN12 | chrI:75,977-76,477 | LM · ISM · Random-Init | 100% (`_TSS` idx39) |
| C | KRE33 | chrXIV:374,871-375,371 | LM · ISM · Random-Init | ~100% (`_RRB` idx3, −1 bp) |
| E | DTD1 | chrIV:65,235-65,431 | ISM | 100% (crop of `_SS`) |
| F | MMS2 | chrVII:346,669-347,169 | LM · ISM · Random-Init | 100% (`_TSS` idx31; LM = MAD1 r2176) |
| G | HOP2 | chrVII:435,625-436,401 | ISM | 100% (crop of `_SS`) |

> **14 logos total** (A/B/C/F = 3 model rows each; E/G = ISM only). Every row is full-coverage.
> The four Random-Init rows are the **lr=5e-4** scratch model (GPU step 0); all other rows are
> precomputed (`scores.h5` / `preds.npz` caches).
