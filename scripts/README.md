# scripts/

All pipeline scripts, organized by stage so the project runs end-to-end
`00 → 04`. Numbered-step prefixes (`0_`, `1_`, `2_`, …) inside each leaf
directory encode execution order, and each working `*.py` keeps its paired
`*.sh` runner. Scripts read paths from `config/paths.yaml` via
`shorkie.config` (no hardcoded filesystem paths) and submit through
`scripts/common/` so the SLURM layer is a thin, parameterized wrapper with a
local/container fallback.

| Stage | Directory | Contents |
|-------|-----------|----------|
| 00 | `00_setup/` | submodule init, environment checks, data download wrappers |
| 01 | `01_data_build/lm_corpus/` | LM genome corpus: download → repeat-mask → filter → TFRecords (+ `run_pipeline.sh`) |
| 01 | `01_data_build/supervised_tracks/` | regulatory tracks: FASTQ → BAM → BigWig → peaks → `hound_data.py` (+ `make_data.sh`) |
| 02 | `02_train/shorkie_lm/` | MLM pretraining (`hound_train.py`, `loss=mlm`) |
| 02 | `02_train/shorkie_finetuned/` | supervised, `westminster_train_folds.py --restore <LM>` |
| 02 | `02_train/shorkie_scratch/` | supervised, random init (no `--restore`) |
| 03 | `03_eval/lm/` | MLM perplexity, genome evaluation, architecture comparison |
| 03 | `03_eval/supervised/` | track-prediction metrics |
| 04 | `04_analysis/` | eQTL, MPRA, motif/MoDISco, ISM, attention, UMAP, SMT3, dependency maps, phylogeny |
| —  | `common/` | portable SLURM submit wrapper + shared runners |

> The current `analysis/` and `model/` trees are migrated here in Phase 3
> (`git mv`, preserving history). This README is the target map.
