# 01 · Cross-fold confidence intervals on the headline metrics

The paper reports the headline expression-prediction numbers as single point estimates aggregated
over all eight cross-validation folds. This attaches confidence intervals to them, defined using
those folds.

## Why this design

From a point estimate a reader cannot tell whether the gap is stable across folds or driven by one
or two of them. The
evaluation runs already wrote per-fold, per-track accuracy tables, so no model needs to be re-run —
the interval can be recovered exactly, from the same artifacts the published figure was built from.

The design deliberately reports **three different intervals**, because they answer different
questions, and the fold-based one is what a cross-validated result actually calls for:

| Interval | Resamples | Answers |
|---|---|---|
| 95% t interval on the per-fold statistic (n=8) | — | *"Would another draw of test folds move this?"* — the question at issue |
| Percentile bootstrap over units | tracks / genes | *"Would another draw of tracks move this?"* |
| Hierarchical bootstrap | folds, then units | both sources jointly |

A **paired** per-fold contrast is reported alongside, because Shorkie and the baselines are evaluated
on the *same* eight folds — the paired difference is far more sensitive than comparing two
independent intervals, and it is what actually establishes significance here.

## What it found

Both baselines were run, because **the manuscript text and Figure 3C do not use the same one**:

| Model | Tree | Fig 3C RNA-Seq median |
|---|---|---|
| Shorkie | `self_supervised_unet_small_bert_drop` | **0.776** |
| Shorkie_Random_Init (LR-optimised, what Fig 3C plots) | `supervised_unet_small_bert_drop_variants/learning_rate_0.0005` | **0.703** |
| Shorkie_Random_Init (un-tuned lr 1e-4, where the text's `0.67` comes from) | `supervised_unet_small_bert_drop` | **0.666** |

Bin-level Pearson's R on the transcriptional-regulator induction RNA-seq tracks (n=3,053 tracks,
8 folds), 95% intervals over folds:

```
Shorkie                       0.776  [0.715, 0.841]
Shorkie_Random_Init (5e-4)    0.703  [0.664, 0.767]
Shorkie_Random_Init (1e-4)    0.666  [0.632, 0.718]

paired Δ vs lr 5e-4   +0.063  [+0.028, +0.098]   Wilcoxon p = 0.016
paired Δ vs lr 1e-4   +0.103  [+0.066, +0.141]   Wilcoxon p = 0.008
```

The marginal intervals overlap; the **paired** intervals exclude zero in both comparisons.

Per fold, Shorkie beats the LR-optimised baseline in **7 of 8** folds — fold 2 is the exception, by
−0.004 — and the un-tuned baseline in **all 8**. That is exactly what the two p-values encode: at
n=8 a two-sided signed-rank test bottoms out at `p = 0.0078` when every fold agrees, which is the
un-tuned comparison; one dissenting fold at the smallest rank gives `p = 0.0156`, which is the
LR-optimised comparison. So the harder comparison is significant but not saturated, and it should be
reported that way rather than as a clean sweep.

The recomputation reproduces all four published anchors (0.78 → 0.7763, 0.67 → 0.6664,
0.88 → 0.8797, 0.74 → 0.7425); see `results/verify_revision_01.csv`.

One honest nuance the forest plot exposes: on **3F (quantile-normalised gene-level R) for the
1000-strains RNA-seq tracks**, Shorkie (0.28) and the LR-optimised baseline (0.28) are
indistinguishable. Every other panel × track combination separates.

## Steps

| Step | Script | Output |
|---|---|---|
| 1 | `1_collect_fold_metrics.py` | `results/fold_metrics.csv` — tidy long form, one row per (model, level, track type, fold, unit, metric) |
| 2 | `2_bootstrap_ci.py` | `results/headline_ci.csv`, `results/paired_fold_deltas.csv`, `results/verify_revision_01.csv` |
| 3 | `3_plot_forest.py` | `results/headline_forest.png`, `results/headline_paired_folds.png` |

Each `*.py` has a paired `*.sh`; add `--dry-run` to print the resolved command.

## Inputs and cost

CPU only, about a minute end to end. No model weights. Reads evaluation artifacts already on disk
under `datasets.supervised_root`:

- bin level — `<tree>/train/f{0..7}c0/eval/acc.txt`
- gene level — `<tree>/gene_level_eval_rc/f{0..7}c0/{RNA-Seq,1000-RNA-seq}/{acc.txt,gene_acc.txt}`

The panel recipes (which aggregation, the positive-values-only group mean, the bottom-10%-coverage
drop for 3G) are ported from `reproduction/figure_03/recheck/build_3C_violin.py` and
`build_3DEFG_scatter.py`, so the point estimates match the published figure by construction.

## Note for the manuscript

The text quotes `0.67` (the un-tuned baseline) while Figure 3C plots `0.703` (the LR-optimised one).
Both numbers are correct for their own baseline, but quoting the weaker one in the headline
understates the strongest available baseline. Recommended: lead with **0.78 vs 0.70**
against the LR-optimised baseline and keep 0.67 as the un-tuned reference, with the paired interval
and p-value attached. This is a text/figure consistency fix, not a change to any result.
