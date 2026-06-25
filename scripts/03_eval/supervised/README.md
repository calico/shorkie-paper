# 03_eval/supervised — Shorkie (supervised) track-prediction evaluation

Evaluation of the fine-tuned/scratch 8-fold supervised models against held-out
ChIP-exo / MNase / RNA-seq tracks. The reproducible pipeline lives under
`track_prediction_eval/`:

- `1_train_valid_curves/` — parse per-fold training logs and plot
  train/valid loss & Pearson-r learning curves.
- `2_bin_gene_level_metrics/` — bin-level and gene-level prediction-quality
  metrics (frequency/score distributions, per-track score differences).
- `3_viz_rnaseq_tracks/` — per-gene predicted-vs-observed track visualizations
  (incl. fold-membership checks and gene-annotation overlays).

## Provenance / scope

`track_prediction_eval/` is the **cleaned, de-hardcoded distillation** of the
work-dir aggregation tree `experiments/SUM_data_process/` (subdirs
`shorkie_eval_track_pred_model_stats_over_epoch`, `…_bin_gene_level_viz`,
`…_viz_rnaseq_track`). That `SUM_data_process/` tree (≈790 `.py`) is an
**exploratory output-aggregation area, not a reproducible pipeline** — it bundles
scratch variants (`*_avg`, `*_all`, `_`-prefixed drafts, a `*_benchmark` copy)
and generated outputs alongside duplicated helpers. Its genuinely-reproducible
content is what was lifted here (and, for the other analyses, into
`04_analysis/shorkie/{eqtl,mpra}` and `…/attention_map`); the two superseded
work-dir variants `archive/SUM_data_process_adam_default/` and
`archive/SUM_SQRT_process/` are **not** part of this release.
