# 05 · Figure and caption corrections

Three presentation defects in the published figures. None of them changes a result — each is
verified to leave the underlying numbers untouched.

| Defect | Fix |
|---|---|
| Figure 6B–C colours indistinguishable | `1_replot_fig6BC_palette.py` |
| Figure 4D purple region unexplained and misaligned | `2_redraw_fig4D_schematic.py` |
| Figure 5 caption has D and E reversed | `3_fig5_caption_fix.py` |

---

## 1 · Figure 6B–C palette

The three quantile aggregates in Figures 6B and 6C are hard to tell apart.

**Why this design.** The complaint is specific and measurable, so the fix should be measurable too
rather than "we picked nicer colours". The published aggregates are dark green `#006400`, dark red
`#8B0000` and black `#000000` — three colours that differ little in hue and almost not at all in
lightness. On top of that, the 18 per-gene lines are drawn from `tab20` resampled to 18, and `tab20`
only has 10 hue pairs, so stretching it repeats hues.

The redraw changes three things: the aggregates move to Paul Tol's high-contrast triple
(`#004488` / `#BB5566` / `#DDAA33`), chosen for separation in both hue and lightness; marker shape
and line style become **redundant** encodings so the series survive losing colour entirely; and the
per-gene lines are tinted by the quantile they belong to with markers separating genes within a
quantile — which also makes the panel's three-groups-of-six structure visible, something the original
palette actively hid.

**What it found.** Separation, measured as CIE76 colour distance under normal vision and under
simulated protanopia, deuteranopia and tritanopia (Machado et al. 2009 matrices):

| Palette | worst-case ΔE76 | worst-case lightness gap |
|---|---|---|
| published (dark green / dark red / black) | **4.3** | 2.8 |
| revised (Tol high-contrast) | **31.1** | 12.5 |

A just-noticeable difference is roughly ΔE ≈ 2.3, so the published palette's worst pair sits barely
above the threshold at which two colours are the same colour. The revised palette clears it by more
than an order of magnitude.

The AUROC/AUPRC computation is imported unchanged from
`reproduction/figure_06/recheck/build_panels_BC.py`, and the script asserts the recomputed means
against the committed `fig6_BC.csv` — 0.9953 and 0.9958, both PASS. Nothing but the drawing changed.

## 2 · Figure 4D schematic

The purple region in Figure 4D is unexplained, and the motif labels do not line up with the donor
and acceptor sites.

**Why this design.** Both halves of the observation are correct: the purple block is the intron and
the caption never says so, and the labels sit at eyeballed positions. In the reproduction's port of
the panel the intron spans x=10..90 while the donor label sits at x=18 — drawn well *inside* the
intron rather than at its 5' boundary — and the branch point sits mid-intron rather than near the 3'
end.

Moving the labels by eye would fix the appearance without fixing the cause, so instead the geometry
is **derived from the R64 annotation and sequence**: introns are recovered as the gaps between
consecutive exons of each transcript, the branch point is located by scanning each intron for
`TACTAAC` (falling back to the degenerate `[TC]ACTAA[CT]`), and the schematic is drawn to the median
intron length with the branch point at its median measured offset.

**What it found**, from 321 annotated multi-exon transcripts:

| Measurement | Value |
|---|---|
| median intron length | **116 nt** |
| median branch point → 3' splice site | **40 nt** |
| introns starting `GTATGT` | 205 (63.9%) |
| introns ending `AG` | 279 (86.9%) |
| branch point located | 244 (76.0%), of which 229 exact `TACTAAC` |

So the branch point belongs near the 3' end, not mid-intron: the published placement is wrong. The revised panel draws the donor at the first six nucleotides, the acceptor at the last
three, the branch point 40 nt from the 3' splice site, with leader lines to each interval and the
intron block labelled in-figure so the colour needs no explanation. A corrected caption fragment is
written to `results/fig4D_caption.md`.

## 3 · Figure 5 caption

The Figure 5 caption has panels D and E reversed.

**Why this design.** Confirmed, and worth stating precisely, because the error is narrower than it
looks: **only the caption is wrong**. The Results text at manuscript L274 cites
"(Figure 5D)" for the 0.55–0.65 correlation range — the boxplot — and L276 cites "(Figure 5E)" for
the genome-wide TF-MoDISco analysis. The figure agrees with the text. The caption alone has them
swapped, and because it declares panels F–J "analogous to (A–E)", the swap propagates to I and J.

The script checks this against the reproduction tree rather than asserting it: `build_DI_boxplots.py`
writes `Figure_5D_MSN2_boxplot.png` and `Figure_5I_MSN4_boxplot.png`, which is the independent record
of which letter the boxplot carries. Both PASS. A corrected full caption is written to
`results/fig5_caption_corrected.md`.

## Inputs and cost

CPU only. Step 1 takes about a minute (loading the MPRA logSED NPZ tree under `results.mpra_viz`);
steps 2 and 3 are seconds.

Step 2 needs the R64 FASTA and GTF. If `data/download.sh --genome` has not been run, pass them
explicitly — the naming differs between the two files by design (FASTA `chrI`…`chrXVI`, GTF
`I`…`XVI`) and the script handles that:

```bash
bash 2_redraw_fig4D_schematic.sh --gtf /path/GCA_000146045_2.59.gtf \
                                 --fasta /path/GCA_000146045_2.cleaned.fasta
```

Revised panels are written to `results/`, **not** into `reproduction/figure_0N/reproduced/` —
that tree reproduces what was *published*, and must keep doing so.
