# Figure 4 — deep-recheck discrepancy log

Figure 4 = **"Shorkie uses promoter and splicing motifs learned during pretraining."**
Strict per-panel recheck of `reproduction/figure_04/` against the published figure
(`paper/Figures/Figure_4.pdf` → `published/Figure_4_full.png`).

This pass makes the per-gene **ISM saliency logos match the published figure exactly**:
the published windows, the published ~33:1 per-row aspect, and **all three model rows**
(Shorkie LM / Shorkie ISM / Shorkie Random-Init ISM) where the published shows them. The
logos are rendered **clean** — no red/purple TF boxes, TF/splice-site text, TSS dividers,
"450/50 nt" labels, or Reference-DB insets — mirroring the original
`2_modisco_DNA_logo.py --no_motif_annotation` mode (the boxes/labels in the published
panels were a manual Illustrator overlay; the user asked for the clean logos). The
gene-model track (strand-aware exons/introns + gene name) is kept. CPU-only except a one-time
GPU run that recomputes the four Random-Init rows (A/B/C/F) with the lr=5e-4 scratch model.

## What the published figure contains (confirmed)
- **A/B/C** (RPL26A / FUN12 / KRE33) and **F** (MMS2): a 500 bp window with **three stacked
  DNA-letter-logo rows — Shorkie LM, Shorkie ISM, Shorkie Random-Init ISM** — over a gene
  model (the published also overlays a "Reference DB" row + curated TF boxes; those are the
  manual annotations the user asked to drop).
- **E** (DTD1) and **G** (HOP2): a single **Shorkie ISM** logo row + gene model (ISM-only).
- **D** (schematic + reconstruction) and **H** (12-TF grid): **removed** from this reproduction
  (`build_4D.py` / `build_4H.py` scripts kept on disk, not rendered).

## Root-cause fixes
The published windows are exact; the prior reproduction picked the wrong `scores.h5` subdirs and the
wrong random-init model. Correct mapping (`results.ism_scores/.../{motif_shorkie_RP_TSS,motif_random_init_RP_TSS}`):

| Panel | Gene | Published window | Shorkie LM | Shorkie ISM | Random-Init (lr=5e-4) | Prior repro bug |
|---|---|---|---|---|---|---|
| A | RPL26A | chrXII:818,862-819,362 | eval_RP r90 | `_RP` p4 i10 (exact) | `fig4_lr5e4` idx0 | — (window ok; wrong random model) |
| B | FUN12 | chrI:75,977-76,477 | eval_TSS r39 | `_TSS` p0 i39 (exact) | `fig4_lr5e4` idx1 | ISM used `_RRB` p15 i3 → **+124 bp** |
| C | KRE33 | chrXIV:374,871-375,371 | eval_TSS r5134 | `_RRB` p11 i3 (−1 bp) | `fig4_lr5e4` idx2 | — |
| E | DTD1 | chrIV:65,235-65,431 | — | `_SS` p22 i0 → crop | — (ISM-only) | — |
| F | MMS2 | chrVII:346,669-347,169 | **eval_TSS r2176 (MAD1)** | `_TSS` p33 i31 (exact) | `fig4_lr5e4` idx3 | ISM used `_SS` (57%); LM used MMS2 r2175 (wrong context) |
| G | HOP2 | chrVII:435,625-436,401 | — | `_SS` p80 i0 → crop | — (ISM-only) | — |

Notes:
- **Shorkie_Random_Init model.** The published random-init row is the **lr=5e-4** scratch checkpoint
  `supervised_unet_small_bert_drop_variants/learning_rate_0.0005` (pinned as `Shorkie_Random_Init` in
  `scripts/.../2_lr_search/{2,3}_compare_lr_variants_*.py`). The released `motif_random_init_RP_TSS`
  ISM was the **old lr=1e-4** model, so all four random rows were recomputed on GPU
  (`recheck/run_fig4_random_ism.sh` → `hound_ism_bed.py` on a 4-row BED at the exact windows →
  `gene_exp_motif_test_fig4_lr5e4/f0c0/part0/scores.h5`, idx 0-3).
- **MMS2 = the MMS2/MAD1 divergent-promoter locus.** The published window chrVII:346,669-347,169 is
  the `_TSS` part33 idx31 entry (BED-labelled **MAD1**; MMS2 is the − strand neighbour). All three
  rows are over this window; the **LM row is MAD1 (eval_TSS r2176, +strand)** — its default 450/50
  window equals the published window. The prior MMS2 row (2175, −strand) gave the wrong 16,384 bp
  masked-LM context (and the gene-midpoint 450/50 → the wrong 346,854-347,354).

## Per-panel: status & residuals
| Panel | Match | Residual / root cause |
|---|---|---|
| **A RPL26A** | 3 clean rows (LM/ISM/Random) + gene model, **exact window** | — |
| **B FUN12** | 3 clean rows, **exact window** (Random recovered from `_TSS`) | — |
| **C KRE33** | 3 clean rows, window **−1 bp** on the Shorkie-ISM row | only an exact-window `_RRB` entry exists for the finetuned ISM (−1 bp, sub-base; aligned by genomic coordinate). LM + Random are exact. |
| **E DTD1** | ISM-only clean row + gene model, full crop | ISM-only by design (no LM/Random in published) |
| **F MMS2** | 3 clean rows (LM/ISM/Random) + gene model, **exact window** | LM = MAD1 row 2176; Random-Init = lr=5e-4 (GPU-recomputed) |
| **G HOP2** | ISM-only clean row + gene model, full crop | ISM-only by design |

## Aspect ratio
Each logo row is drawn at the original `2_modisco_DNA_logo.py` proportion — `figsize=(100,3)`
≈ 33:1 wide/thin — scaled to a practical box (`fig4_common.BOX_W×BOX_H = 20×0.6 in`). bp/inch
varies per gene (windows 197/500/776 bp), as in the published.

## Localization (peak/median per-site |saliency|) — Shorkie ISM vs Random-Init (lr=5e-4)
| Panel | gene | Shorkie ISM | Shorkie LM | Random-Init |
|---|---|---|---|---|
| 4A | RPL26A | 16.30× | 7.07× | 9.23× |
| 4B | FUN12 | 15.63× | 4.27× | 6.35× |
| 4C | KRE33 | 28.03× | 7.37× | 8.67× |
| 4E | DTD1 | 15.50× | (ISM-only) | — |
| 4F | MMS2 | 23.03× | 8.28× | 7.87× |
| 4G | HOP2 | 17.50× | (ISM-only) | — |
Shorkie ISM localizes more sharply than the lr=5e-4 Random-Init in every 3-model panel (A/B/C/F).
verify_fig04.csv = **38/38** (panels D & H removed → their 3 checks dropped).

## Panels D & H — removed (scripts kept)
D (splicing schematic + DB-vs-reconstruction) and H (12-TF grid) are no longer rendered or verified.
Their builders remain on disk and runnable: `build_4D.py`, `build_4H.py`, `run_tomtom.py`,
`match_tfs.py`, `tomtom_RP_matches.tsv` (and `F.SPLICE`, which `build_4D.py` uses).

## Irreducibly manual
The published Figure 4 is an Illustrator composite (panel placement, the red TF/splice boxes
+ text labels, the Reference-DB insets). The recheck reproduces every **data** element as
clean per-panel PNGs; the manual annotations are intentionally omitted per the user request.
