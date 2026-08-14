#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Example 6 — Fine-tuning mini-demo: Shorkie_LM -> Shorkie, end to end, in minutes
# ---------------------------------------------------------------------------
# Example 5 is the *production* recipe: 8 folds, the full 10 GB supervised set,
# ~64 GPU-hours. That is the right thing to run for real, but it is not something
# you can try out to see whether the mechanics work.
#
# This demo runs the SAME mechanism -- `--restore` the pretrained Shorkie_LM trunk
# and train a fresh supervised head over the 5215 tracks -- on a deliberately tiny
# slice of the released data, so it finishes in minutes on one GPU.
#
#   *** The resulting model is NOT useful. ***
#   16 sequences and a couple of epochs cannot train anything; the point is to
#   prove the pipeline runs end to end and to show exactly which knobs matter.
#   For a real model use examples/5_finetune_lm_on_rnaseq.sh.
#
# Requirements: one GPU, ~2 GB disk, and a billing project for the requester-pays
# data bucket. Roughly 100 MB is downloaded.
#
#   bash examples/6_finetune_minidemo.sh -u <your-gcp-project>
#   bash examples/6_finetune_minidemo.sh -u <project> --dry-run
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PROJECT=""; DRY_RUN=0
OUT_DIR="${REPO_ROOT}/minidemo"
# ~96 MB of the fold-0 shard. Each record is ~6 MB compressed, so this yields
# ~16 complete records -- exactly what the 8/4/4 split below needs.
BYTES="0-100663295"

while [[ $# -gt 0 ]]; do case "$1" in
  -u|--project) PROJECT="$2"; shift 2;;
  --out_dir)    OUT_DIR="$2"; shift 2;;
  --dry-run)    DRY_RUN=1; shift;;
  -h|--help)    sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

[[ -n "$PROJECT" ]] || { echo "error: -u PROJECT is required (the data bucket is requester-pays)" >&2; exit 2; }

GSUTIL="$(command -v gsutil || echo "${SHORKIE_GSUTIL:-gsutil}")"
cfg() { python -c "import sys; from shorkie import config; print(config.path(sys.argv[1]))" "$1"; }

LM_CKPT="$(cfg models.shorkie_lm_checkpoint)"
DATA_DIR="${OUT_DIR}/data"
SRC="gs://shorkie-paper/data/supervised/processed"

echo "=== Shorkie fine-tuning mini-demo ==="
echo "  LM checkpoint : $LM_CKPT"
echo "  output        : $OUT_DIR"

run() { echo "+ $*"; [[ "$DRY_RUN" == 1 ]] || "$@"; }

# 1. A tiny slice of the released supervised data ---------------------------
mkdir -p "$DATA_DIR"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "+ gsutil -u $PROJECT cat -r $BYTES $SRC/tfrecords/fold0-0.tfr > $OUT_DIR/fold0_prefix.tfr"
elif [[ ! -s "$OUT_DIR/fold0_prefix.tfr" ]]; then
  echo "+ fetching ~96 MB of fold0-0.tfr"
  "$GSUTIL" -u "$PROJECT" cat -r "$BYTES" "$SRC/tfrecords/fold0-0.tfr" > "$OUT_DIR/fold0_prefix.tfr"
fi
run "$GSUTIL" -u "$PROJECT" cp "$SRC/statistics.json" "$OUT_DIR/statistics.json"
run "$GSUTIL" -u "$PROJECT" cp "$SRC/targets.txt"     "$OUT_DIR/targets.txt"

run python "${REPO_ROOT}/examples/make_minidemo_data.py" \
    --prefix_tfr "$OUT_DIR/fold0_prefix.tfr" \
    --statistics "$OUT_DIR/statistics.json" \
    --targets    "$OUT_DIR/targets.txt" \
    --out_dir    "$DATA_DIR" \
    --n_train 8 --n_valid 4 --n_test 4

# 2. Shrink the training schedule -------------------------------------------
# Identical to the released fine-tuning config except for the knobs that control
# how LONG training runs. The task/loss/optimizer -- the parts that define what
# fine-tuning *is* -- are untouched. warmup_steps must come down from 5000 or the
# demo would finish before the learning rate ever warms up.
PARAMS="${OUT_DIR}/params_minidemo.json"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "+ write $PARAMS (batch_size=1, warmup_steps=1, epochs<=2, steps_per_epoch_max=4)"
else
  python - "$REPO_ROOT/scripts/02_train/shorkie_finetuned/params.json" "$PARAMS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
p["train"].update(batch_size=1, shuffle_buffer=8, warmup_steps=1, patience=1,
                  train_epochs_min=1, train_epochs_max=2, steps_per_epoch_max=4)
json.dump(p, open(sys.argv[2], "w"), indent=4)
print(f"  wrote {sys.argv[2]}  (task={p['train']['task']}, lr={p['train']['learning_rate']})")
PY
fi

# 3. Fine-tune ---------------------------------------------------------------
# --restore loads the pretrained LM trunk; the supervised head is new. Dropping
# --restore is exactly what makes this Shorkie_Random_Init instead.
run python -m baskerville.scripts.hound_train \
    -o "${OUT_DIR}/train" \
    --restore "$LM_CKPT" \
    "$PARAMS" \
    "$DATA_DIR"

echo
echo "=== done ==="
echo "Checkpoint: ${OUT_DIR}/train/model_best.h5"
echo
echo "Two things in the log above are EXPECTED, not errors:"
echo "  * 'Skipping loading weights for layer ... dense_22 ... Weight expects shape"
echo "    (384, 5215). Received saved weight with shape (384, 384)'"
echo "      -> correct: the LM trunk transfers, the supervised head is NEW. That"
echo "         mismatch is precisely what fine-tuning means here."
echo "  * near-zero train_r / negative train_r2"
echo "      -> correct: 8 sequences and 2 epochs cannot learn anything."
echo
echo "Reference run (CPU, no GPU): ~40 s of training after the data step."
echo "Remember: this model is a mechanism demo, not a usable predictor."
echo "For a real run see examples/5_finetune_lm_on_rnaseq.sh."
