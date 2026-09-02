# 03 · Accounting for the 3,053 induction RNA-seq tracks

The Methods describe the induction RNA-seq design — YETI strains, ministat array, ~8 timepoints per
culture — but never multiply it out, so "8 TFs plus 460 other genes" and "3,053 tracks" sit in the
paper with no bridge between them. This supplies the arithmetic.

## Why this design

What is missing is arithmetic, not data:
"8 TFs plus 460 genes" and "3,053 tracks" sit in the paper without a bridge between them.

That bridge is already recoverable from committed metadata: `minimal_example/sheet.txt` is the
full 5,215-row targets sheet, and every induction RNA-seq identifier encodes its own design as
`<GENE>_T<minutes>_S<sample>`. So this is a parsing job, not an experiment — which is the right
answer, because it means the number can be stated exactly rather than approximately.

The partition is **derived, not assumed**: genes are split by their sampling schedule, and the
script then cross-checks that the schedule-derived group is the expected set of 8 TFs.

## What it found

```
3,053 induction RNA-seq tracks  =    580  +  2,473
                                      |        |
  8 TF perturbations in replicate  ---+        |
    ARG80 CUP2 GAT4 MET4 MSN2 MSN4 NDT80 RPN4  |
    schedule 0/5/10/15/30/45/60/90 min          |
    4-12 replicates per timepoint               |
                                                |
  broader gene panel, 329 genes  ---------------+
    schedule 0/5/10/20/40/70/120/180 min
    mostly 1 replicate per timepoint (220 genes have all 8 timepoints)
```

The two partitions are cleanly separated by their **sampling schedules**, which are different and
non-overlapping after 10 min — exactly 8 genes use the microarray-matched `0/5/10/15/30/45/60/90`
series, and they are precisely the 8 TFs named in the Methods. That is a strong internal
consistency check on the whole accounting.

**The 460 → 329 gap is real and should be stated.** The Methods say 460 genes were prioritised for
induction; 329 yielded tracks in the released training set, so 131 did not survive library prep,
sequencing or QC. The paper currently implies all 460 contributed.

All seven anchors reproduce exactly (`results/verify_revision_03.csv`): 3,053 / 1,014 / 1,128 / 20
tracks, 5,215 total, 8 TFs, and the two partitions summing to the RNA-seq total.

Worth stating plainly: the
biological analysis of this resource is not in this paper. Here it is training data, and its
description should be sized accordingly — an accounting table plus the design figure, not a results
section.

## Steps

| Step | Script | Output |
|---|---|---|
| 1 | `1_atlas_accounting.py` | `results/atlas_summary.csv` (the table that sums to 3,053), `atlas_by_gene.csv`, `atlas_timepoints.csv`, `verify_revision_03.csv` |
| 2 | `2_plot_atlas.py` | `results/atlas_design.png` — four-panel supplementary figure |

## Inputs and cost

CPU, seconds. The only input is the committed `minimal_example/sheet.txt`; nothing is downloaded and
no model is loaded, so this runs in a fresh clone.
