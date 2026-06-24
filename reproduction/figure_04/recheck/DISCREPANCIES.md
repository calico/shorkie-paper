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
gene-model track (strand-aware exons/introns + gene name) is kept. CPU-only except a
single GPU ISM re-run for MMS2's Random-Init row.

## What the published figure contains (confirmed)
- **A/B/C** (RPL26A / FUN12 / KRE33) and **F** (MMS2): a 500 bp window with **three stacked
  DNA-letter-logo rows — Shorkie LM, Shorkie ISM, Shorkie Random-Init ISM** — over a gene
  model (the published also overlays a "Reference DB" row + curated TF boxes; those are the
  manual annotations the user asked to drop).
- **E** (DTD1) and **G** (HOP2): a single **Shorkie ISM** logo row + gene model (ISM-only).
- **D**: splicing schematic + DB-vs-reconstruction. **H**: 12-TF DB-vs-reconstruction grid.

## Root-cause fix — the reproduction read the wrong scores.h5 subdirectories
The published windows are exact; the prior reproduction picked offset/ISM-only entries. The
exact-window data exists on disk (`results.ism_scores/.../{motif_shorkie_RP_TSS,motif_random_init_RP_TSS}`):

| Panel | Gene | Published window | Shorkie LM | Shorkie ISM | Random-Init ISM | Prior repro bug |
|---|---|---|---|---|---|---|
| A | RPL26A | chrXII:818,862-819,362 | eval_RP r90 | `_RP` p4 i10 (exact) | `_RP` p4 i10 (exact) | — (was already ok) |
| B | FUN12 | chrI:75,977-76,477 | eval_TSS r39 | `_TSS` p0 i39 (exact) | `_TSS` p0 i39 (exact) | used `_RRB` p15 i3 → **+124 bp**; claimed "no Random_Init" |
| C | KRE33 | chrXIV:374,871-375,371 | eval_TSS r5134 | `_RRB` p11 i3 (−1 bp) | `_TSS_select` p0 i5 (exact) | claimed "no Random_Init" |
| E | DTD1 | chrIV:65,235-65,431 | — | `_SS` p22 i0 → crop | — (ISM-only) | — |
| F | MMS2 | chrVII:346,669-347,169 | eval_TSS r2175 → slice | `_TSS` p33 i31 (exact) | **GPU-recomputed** | used `_SS` gene-body → **57% coverage**, ISM-only |
| G | HOP2 | chrVII:435,625-436,401 | — | `_SS` p80 i0 → crop | — (ISM-only) | — |

Notes:
- **MMS2 = the MMS2/MAD1 divergent-promoter locus.** The published window chrVII:346,669-347,169
  is the `_TSS` part33 idx31 entry (BED-labelled MAD1; MMS2 is the − strand neighbour). All
  three MMS2 rows are drawn over this exact window; the LM row is the eval_TSS MMS2 row sliced
  to it by coordinate (the gene-midpoint 450/50 rule gives the wrong 346,854-347,354).
- **MMS2 Random-Init** was never in the released ISM set, so it was recomputed with the same
  scratch checkpoint + flags the other random-init entries used
  (`recheck/run_mms2_random_ism.sh` → `hound_ism_bed.py` on a 1-row BED at the exact window).

## Per-panel: status & residuals
| Panel | Match | Residual / root cause |
|---|---|---|
| **A RPL26A** | 3 clean rows (LM/ISM/Random) + gene model, **exact window** | — |
| **B FUN12** | 3 clean rows, **exact window** (Random recovered from `_TSS`) | — |
| **C KRE33** | 3 clean rows, window **−1 bp** on the Shorkie-ISM row | only an exact-window `_RRB` entry exists for the finetuned ISM (−1 bp, sub-base; aligned by genomic coordinate). LM + Random are exact. |
| **E DTD1** | ISM-only clean row + gene model, full crop | ISM-only by design (no LM/Random in published) |
| **F MMS2** | 3 clean rows (LM/ISM/Random) + gene model, **exact window** | Random-Init row is GPU-recomputed (not in the released set) |
| **G HOP2** | ISM-only clean row + gene model, full crop | ISM-only by design |

## Aspect ratio
Each logo row is drawn at the original `2_modisco_DNA_logo.py` proportion — `figsize=(100,3)`
≈ 33:1 wide/thin — scaled to a practical box (`fig4_common.BOX_W×BOX_H = 20×0.6 in`). bp/inch
varies per gene (windows 197/500/776 bp), as in the published.

## Localization (peak/median per-site |saliency|) — Shorkie ISM vs Random-Init
| Panel | gene | Shorkie ISM | Shorkie LM | Random-Init |
|---|---|---|---|---|
| 4A | RPL26A | 16.30× | 7.07× | 7.19× |
| 4B | FUN12 | 15.63× | 4.27× | 6.19× |
| 4C | KRE33 | 28.03× | 7.37× | 6.43× |
| 4E | DTD1 | 15.50× | (ISM-only) | — |
| 4F | MMS2 | 23.03× | 7.89× | 14.11× |
| 4G | HOP2 | 17.50× | (ISM-only) | — |
Shorkie ISM localizes more sharply than Random-Init in every 3-model panel (A/B/C/F). verify_fig04.csv = 41/41.

## Panels D & H (out of scope here — unchanged)
- **D**: splicing schematic + DB consensus (GTATGT/TACTAAC) + reconstruction from the SS-ISM PWM.
- **H**: 12-TF DB-vs-reconstruction grid (TomTom via MEME 5.5.7 built from source; cached
  `report/motifs.html` fallback). See `fig4H_tomtom_pairs.csv`.

## Irreducibly manual
The published Figure 4 is an Illustrator composite (panel placement, the red TF/splice boxes
+ text labels, the Reference-DB insets). The recheck reproduces every **data** element as
clean per-panel PNGs; the manual annotations are intentionally omitted per the user request.
