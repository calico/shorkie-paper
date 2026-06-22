# Figure 4 — deep-recheck discrepancy log

Figure 4 = **"Shorkie uses promoter and splicing motifs learned during pretraining."**
Strict per-panel recheck of `reproduction/figure_04/` against the published figure
(`paper/Figures/Figure_4.pdf` → `published/Figure_4_full.png`).

The earlier reproduction passed 14/14 *correctness* checks (coordinate-overlap +
localization ratio) but **rendered far fewer elements than the published panels**. This
recheck closes the substantive gaps (user-approved **focused upgrade**) and documents
what remains irreducibly manual. All panels are CPU-reproducible from cached numeric
data — **no GPU**. Builders live in `recheck/build_4*.py` (shared `fig4_common.py`).

## What the published figure actually contains (confirmed this session)

- **A/B/C** (RPL26A / FUN12 / KRE33): a single **500 bp window = 450 bp upstream + 50 bp
  downstream of the TSS** (the published "450 nt / 50 nt" split — *not* a separate zoom
  inset), with **four stacked DNA-letter-logo rows**: **Shorkie LM**, **Shorkie ISM**,
  **Shorkie_Random_Init ISM**, **Reference DB**, plus curated TF boxes (Fhl1/Rap1 …), a
  strand-aware gene model, and the TSS divider.
- **D**: splicing schematic (GUAUGU / UACUAAC / YAG) + **Database Motifs** row + **Shorkie
  reconstruction from ISM PWM** row (donor GTATGT, branch TACTAAC).
- **E/F/G** (DTD1 / MMS2 / HOP2): one **Shorkie ISM** logo row + gene model + dashed boxes
  (Start Codon / 5′ donor / Branch point / 3′ acceptor / Stop Codon).
- **H**: 12-TF two-row grid (Database Motifs over Shorkie reconstruction) for Rap1, Fhl1,
  Sfp1.1, TATA Box, Reb1, Abf1, Tbf1.1, Cbf1, Ume6.2, Dot6p, PAC motif (Dot6), RRPE (Stb3).

## Per-panel: status, fix, and residuals

| Panel | Published | Old reproduction | Recheck fix | Residual / root cause |
|---|---|---|---|---|
| **A/B/C** | 4 rows (LM/ISM/Random/RefDB) + TF boxes + gene model | 2 rows (ISM + Random) at 20″ width, no LM, no RefDB, no gene model | `build_4ABC.py`: adds **Shorkie LM** row (LM masked-prediction IC logo from cached `preds.npz`), Reference-DB row (PWM scan of the panel's published TFs) + curated TF boxes (Fhl1/Rap1; Abf1/RRPE/PAC; Rap1/RRPE/PAC), gene model, 450/50 divider; rows aligned by **genomic coordinate** | (i) B/C have **no Random_Init tree** on disk (only the RP sub exists) → row labelled, not fabricated. (ii) The LM window (TSS±450/50) and the ISM `scores.h5` window differ by ~124 bp for FUN12/KRE33 → rows aligned by genomic coordinate (RPL26A aligns exactly). (iii) The TF-box set is the published author curation, placed at their best PWM-scan position. |
| **D** | schematic + DB + reconstruction | text print only | `build_4D.py`: schematic bar + DB consensus logos (GTATGT/TACTAAC) + **reconstruction from the SS-ISM PWM** at the actual donor/branch sites (averaged over DTD1/MMS2/HOP2, rev-comp for − strand) | No **SS-MoDISco** exists on disk → reconstruction is taken directly from the SS-ISM saliency PWM around the splice sites (documented), not a MoDISco motif. |
| **E/F/G** | ISM + gene model + 5 dashed splice boxes | 1 ISM logo row, no gene model, no annotations | `build_4EFG.py`: ISM logo + strand-aware gene model + dashed boxes (Start/donor/branch/acceptor/Stop) from GTF intron boundaries | Branch point is placed ~30 bp upstream of the acceptor (yeast consensus) — approximate, since the exact branch A is not in the GTF. |
| **H** | 12-TF DB-vs-reconstruction grid | vertical stack of 6 unnamed modisco logos | `build_4H.py`: 12-TF grid in published order; DB logo (IC-weighted, trimmed) over the Shorkie reconstruction = RP-MoDISco pattern matched to each TF | Matching uses TomTom (MEME 5.5.7 built from source) on the RP-MoDISco patterns; TFs unmatched by TomTom fall back to Pearson correlation with the DB motif (flagged in `fig4H_tomtom_pairs.csv`). |

## Numeric reproduction (unchanged correctness checks — all reproduce exactly)

| Panel | gene | localization (peak/median per-site |saliency|) |
|---|---|---|
| 4A | RPL26A | Shorkie ISM **16.3×**  (LM 7.1×, Random 7.2× → Shorkie > Random) |
| 4B | FUN12 | 18.2× |
| 4C | KRE33 | 28.0× |
| 4E | DTD1 | 15.5× |
| 4F | MMS2 | 10.0× |
| 4G | HOP2 | 17.5× |

All six ISM windows provably overlap their R64 gene (coordinate-audited). `n` TF-MoDISco
patterns recovered on the Shorkie RP ISM ≥ 6 (105 patterns).

## Tooling note (Step 0)
`tomtom`/MEME is not in the `yeast_ml` env and the conda solver failed to install it (OOM /
solver crash, even on a 32 GB compute node). MEME **5.5.7 was therefore built from source**
to `~/tools/meme` (matching the DB header `MEME version 5.5.7`) and used for the panel-H /
Reference-DB matching (`recheck/run_tomtom.py` → `recheck/tomtom_RP_matches.tsv`). The
cached modisco `report/motifs.html` (the original pipeline's own TomTom) agrees with it and
is used as a fallback.

## Irreducibly manual (documented, not reproduced pixel-for-pixel)
The published Figure 4 is an Illustrator composite. The exact panel placement, scale bars,
font choices, and box positions are manual layout; the recheck reproduces every **data**
element programmatically (per-panel PNGs + a stacked `Figure_4ABC_reproduced.png`) but does
not reconstruct the final single-canvas Illustrator composition.
