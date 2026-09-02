# Changelog

Notable changes to this repository. The models and datasets themselves are versioned by the release
catalogue in [`data/manifest.json`](data/manifest.json); this file tracks the code, docs, and what is
published where.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `scripts/05_revision/` — eleven follow-up analyses and controls for claims in the Shorkie paper,
  one directory per question, with [`EXPERIMENTS.md`](scripts/05_revision/EXPERIMENTS.md) as the
  design document. Six run on CPU from artifacts already on disk and are executed with their results
  committed; the rest are dry-run-verified SLURM scripts. Highlights:
  cross-fold confidence intervals on the headline expression-prediction gap (paired Δ +0.063
  [+0.028, +0.098], Wilcoxon p = 0.016, 7/8 folds); a quantitative replacement for the ISM
  motif-recovery claim (2 of 121 motifs significant after Holm correction, both core
  promoter/splicing rather than TF motifs); the arithmetic reconciling the 3,053 induction RNA-seq
  tracks to 580 + 2,473 over 337 induced genes; a measured, colour-vision-safe redraw of Figure 6B/C
  (worst-case ΔE76 4.3 → 31.1, numbers unchanged); and a Figure 4D schematic redrawn from measured
  intron geometry (321 introns, median 116 nt, branch point 40 nt from the 3' splice site).
  `reproduction/` is unmodified — it reproduces what was published, and revised panels are new
  outputs under `scripts/05_revision/**/results/`.
- Two findings from the corpus-scope analysis that are recorded because they bear on published
  claims: the broad fungal corpus has **1.62× more** training windows than the Saccharomycetales
  corpus (so data volume alone cannot explain the "sweet spot"), and the four corpus tiers were
  **not trained on a matched schedule** — R64/80_strains used `train_epochs_max=500` / `patience=50`
  against `10000` / `1000` for the other two. Each run converged within its own budget, but the
  cross-tier comparison carries that difference.
- `scripts/05_revision/06_mpra_random_init/0_build_context_tsv.py` regenerates the MPRA
  insertion-context TSVs from the R64 GTF, replacing a lost intermediate that had left the MPRA
  pipeline unrunnable for any model. Verified byte-identical to the surviving originals for all 22
  reporter genes.

## [v1.2.0] — 2026-08-14

The documented user path did not actually run end to end. This release fixed that and verified it from
a clean download on CPU.

### Fixed
- `minimal_example/run_shorkie_variant.py` built its default paths from empty placeholder strings, so
  `--params_file` resolved to `/params.json` and the README Quick Start failed for everyone.
- The `models.shorkie_finetuned` config key pointed at a work-directory run whose weights are **not**
  the released ones, so committed example outputs came from a model nobody could download. The demo
  variant's logSED is **+0.0643** against the released weights.
- `scripts/02_train/shorkie_scratch/params.json` was `learning_rate: 1e-4` while the released
  Shorkie_Random_Init is **5e-4** — the shipped config did not reproduce the released ablation.
- Examples 1–2 read `params.json` from `train/`, but the released layout puts it at the model-dir root.
- Four scripts invoked `.py` files that exist nowhere; `slurmify()` raised `NameError`; seven figure
  READMEs linked to gitignored PDFs that 404 on GitHub.

### Added
- **The reference genome is now obtainable**: `data/download.sh --genome` (FASTA + GTF + `.fai`, with
  md5 verification). Chromosome naming is load-bearing and differs per file — FASTA `chrI…chrXVI`, GTF
  `I…XVI` — so an Ensembl download silently fails.
- `examples/6_finetune_minidemo.sh` — the real `--restore` fine-tune on a tiny slice, verified end to
  end on CPU in ~40 s, giving fine-tuning its first proof-of-run.
- `tests/test_release.py` — release-integrity suite; every guard was confirmed to fire against the
  original defect.
- Documentation site: <https://khchao.com/shorkie/>.

## [v1.1.0] — 2026-07-23

### Added
- All three model variants live on `gs://seqnn-share/shorkie_models/`, and the eQTL/MPRA benchmark data
  on `gs://shorkie-paper` — so Figures 6–7 reproduce on CPU without re-scoring.
- Citation metadata (`CITATION.cff`) and the bioRxiv preprint reference.

### Changed
- Model bucket paths repointed under the `shorkie_models/` prefix after a bucket reorganisation.

## [v1.0.0] — 2026-06-25

First public release: the installable `shorkie` package, config-driven pipelines staged
`00_setup → 04_analysis`, figure-by-figure reproduction (`notebooks/fig01`–`fig07` with 206/206 numeric
checks), runnable examples, and the release catalogue.

[Unreleased]: https://github.com/calico/shorkie-paper/compare/v1.2.0...HEAD
[v1.2.0]: https://github.com/calico/shorkie-paper/releases/tag/v1.2.0
[v1.1.0]: https://github.com/calico/shorkie-paper/releases/tag/v1.1.0
[v1.0.0]: https://github.com/calico/shorkie-paper/releases/tag/v1.0.0
