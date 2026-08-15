# Changelog

Notable changes to this repository. The models and datasets themselves are versioned by the release
catalogue in [`data/manifest.json`](data/manifest.json); this file tracks the code, docs, and what is
published where.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **`git clone --recurse-submodules` over HTTPS now works.** `.gitmodules` pinned both submodules to
  SSH URLs, so the clone command the README recommends "for users without SSH access" failed for
  exactly those users — both submodules aborted with *"Could not read from remote repository"*, leaving
  `external/` empty. Both repos are public, so the URLs are now HTTPS.
  ([#1](https://github.com/calico/shorkie-paper/issues/1))
- `CITATION.cff` rendered Emily H. Stoops as "E." in the `preferred-citation` block; the two author
  lists now agree.

### Added
- **LM checkpoints for the other pretraining corpus tiers**, catalogued as `models.lm_variants`: the
  four `unet_small` runs behind the Figure 1F/G corpus-scaling comparison (R64 / 80_strains /
  165_Saccharomycetales / 1341_Fungus), plus 1341_Fungus at `unet_small_bert_drop` — the same
  architecture as the released Shorkie_LM, on the largest corpus. Each records the `num_features` value
  it must be loaded with, since `params.json` does not.
  ([#3](https://github.com/calico/shorkie-paper/issues/3))
- `data/download.sh --models lm-variants`.
- Orientation READMEs for `config/`, `containers/`, `src/shorkie/`, `tests/`, `reproduction/common/`,
  and the `scripts/` stages that lacked one.
- This changelog.

### Changed
- `scripts/01_data_build/supervised_tracks/README.md` now states plainly that the stage still contains
  author-environment paths, and lists them as substitution points.
- Removed `scripts/01_data_build/lm_corpus/phylogentic_tree/` — a misspelled, unreferenced duplicate,
  byte-identical to the `scripts/04_analysis/others/phylogenetic_tree/` copy that Figure 1 actually uses.

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
