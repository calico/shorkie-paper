#!/bin/bash
#SBATCH --job-name=finetune_lm_on_rnaseq
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=20g
#SBATCH -o finetune_lm_on_rnaseq.%j.out
# ---------------------------------------------------------------------------
# Example 5 — Fine-tune Shorkie_LM on RNA-seq (+ChIP-exo/MNase) tracks
# ---------------------------------------------------------------------------
# This turns the masked DNA language model (Shorkie_LM) into the supervised
# coverage predictor (Shorkie) by continuing training with a `--restore` of the
# LM trunk and a supervised regression head over the track set. It is the exact
# recipe behind models.shorkie_finetuned (see scripts/02_train/shorkie_finetuned/
# make_model.sh, the canonical copy; this is the documented walkthrough).
#
# GPU + SLURM, ~8 GPU-h/fold (≈64 GPU-h for the full 8-fold ensemble). NOT run by
# the example notebooks. Submit with `sbatch examples/5_finetune_lm_on_rnaseq.sh`,
# or portably: `scripts/common/submit.sh --profile gpu examples/5_finetune_lm_on_rnaseq.sh`.
# Add `--dry-run` to print the resolved command without launching.
#
# INPUTS (all config-driven via shorkie.config):
#   1. Shorkie_LM checkpoint  — the --restore target. Get it with:
#        data/download.sh --models lm            # -> models/shorkie_lm/train/model_best.h5
#      (config key models.shorkie_lm_checkpoint).
#   2. Supervised RNA-seq/ChIP-exo/MNase TFRecords (8-fold) — config key
#      datasets.supervised_data. Build your own track set with the pipeline in
#      scripts/01_data_build/supervised_tracks/ (FASTQ -> BAM -> BigWig -> peaks
#      -> hound_data.py), or use the released set (data/download.sh --supervised).
#   3. params.json (next to this script's canonical copy) — same model block as the
#      LM; train.task="fine-tune", train.learning_rate=2e-5 (vs scratch's 1e-4).
#
# To fine-tune on YOUR OWN RNA-seq tracks: prepare a targets sheet + TFRecords
# with scripts/01_data_build/supervised_tracks/, point datasets.supervised_data
# at them, and run this script. The LM trunk transfers; only the head is new.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }

DRY_RUN=0; [[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

RESTORE="$(cfg models.shorkie_lm_checkpoint)"    # Shorkie_LM model_best.h5
EVAL_DIR="$(cfg datasets.lm_corpus_split_root)"
DATA_DIR="$(cfg datasets.supervised_data)"       # 8-fold supervised RNA-seq/ChIP-exo/MNase TFRecords
PARAMS="${REPO_ROOT}/scripts/02_train/shorkie_finetuned/params.json"

echo "Restore (LM):  $RESTORE"
echo "Supervised:    $DATA_DIR"
echo "Params:        $PARAMS"

CMD=(python "${WESTMINSTER_SCRIPTS}/westminster_train_folds.py"
  --restart
  -f 8                       # 8 cross-validation folds
  -e yeast_ml
  --restore "$RESTORE"       # <-- transfer the Shorkie_LM trunk (omit this for from-scratch / Random_Init)
  --eval_dir "${EVAL_DIR}/"
  -o train
  -q a100
  --rc --shifts "0,1"
  "$PARAMS"
  "$DATA_DIR")

if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}" 1>train.out 2>train.err
echo "done — fine-tuned 8-fold ensemble under ./train/f{0..7}c0/"
