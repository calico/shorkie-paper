# 07 · Gene-restricted ISM

Equation 17 sums the ISM over **all 896 output bins** — at 16 bp resolution that is ~14.3 kb of
genomic coverage, so a promoter mutation is scored against everything in the window, neighbouring
genes included. Should it instead sum only the bins covering the target gene?

## Why this design

What makes this worth measuring rather than simply conceding is that the paper is *internally
inconsistent* about it: the variant-effect scoring (`hound_snp`, and the gene-level evaluation at
manuscript L1175) already restricts to gene-body bins. So the machinery and the intent both exist;
the ISM pipeline (`hound_ism_bed.py`, which has no gene-slice option) just does not use them.

The design computes **all three scopes over the same mutations in the same forward passes**, so the
comparison is exact rather than approximate:

| Scope | Bins summed | What it answers |
|---|---|---|
| `all_bins` | all 896 | the published Equation 17 |
| `gene_body` | bins overlapping the target gene | the gene-restricted alternative |
| `tss_window` | bins within ±1 kb of the TSS | local, but independent of annotation extent |

The third scope is there because `gene_body` inherits whatever the annotation says the gene is —
for a short gene it can be a handful of bins, which makes the score noisy. A fixed TSS window
separates "restricting helps" from "restricting to *this particular* interval helps".

## The time-course half

Neighbouring-gene coverage could also affect the comparison **across induction time points** — that is
**Figure 5C**, the 8×8 pairwise Euclidean-distance heatmap between per-timepoint ISM logos, and
correlating static saliency would not test it at all.

So step 1 does **not** average over a single T0 track subset. It computes logSED per bin scope *and*
per induction time point, reusing the published track partition
(`reproduction/figure_05/recheck/fig05_lib.tp_tracks`), and step 2 rebuilds the distance matrix under
each scope with the published recipe — mean-centre each timepoint's PWM across the four bases, then
take the Frobenius distance between timepoints, verbatim `fig05_lib.distance_heatmap`.

The reported number is the correlation between the scopes' distance matrices. If the distance
structure is preserved, the time-course conclusions stand regardless of scope; if it is not, Figure 5C
needs recomputing. ATG42 carries the MSN2 series and TSL1 the MSN4 series; FUN12 and KRE33 have no
time course and are scored at a single timepoint.

## What it measures

The right question is not whether the numbers change — they will, since the denominators differ —
but whether the **conclusions** change. So step 2 reports:

- **Rank agreement** between scopes (Pearson and Spearman on per-position reference-projected
  saliency). The figures show which positions matter, not absolute magnitudes.
- **Top-25 position overlap** (Jaccard) — the concrete question for a sequence-logo panel: does the
  logo show the same bases?
- **Time-course distance-matrix agreement** — the Figure-5C structure under each scope (above).
- **Contamination**: what fraction of reference predicted coverage in the window falls outside the
  target gene at all. If that is small, neighbours had little room to contribute and the published
  convention is safe; if it is large, the concern is real.

Either outcome is reportable. If agreement is high the response can say so with a number and keep
the published panels; if it is not, the per-position grids needed to redraw them are already in
`results/ism/`.

## Loci

The two Figure-5 time-course promoters (ATG42, TSL1) plus two Figure-4 RRB
promoters (FUN12, KRE33) as controls from the other figure that uses Equation 17. Pass `--genes` to
scan a different set.

## Steps

| Step | Script | Cost | Output |
|---|---|---|---|
| 1 | `1_gene_restricted_ism.py` | **GPU**, ~4 × 500 forward passes per locus through the 8-fold ensemble | `results/ism/<name>.npz` |
| 2 | `2_compare_bin_scopes.py` | CPU, seconds | `results/bin_scope_comparison.{csv,png}` |

```bash
scripts/common/submit.sh --profile gpu 1_gene_restricted_ism.sh
bash 2_compare_bin_scopes.sh
```

## Inputs

Released Shorkie weights (`data/download.sh --models finetuned`), the R64 genome, and the T0 RNA-seq
targets sheet. Window placement, the 170-channel input construction and the logSED definition are
ported from `reproduction/figure_07/panels/run_ism_eqtl.py`, which already builds a gene-body
`output_slice` — so the two conventions are computed by the same code path.
