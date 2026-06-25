#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Upload the release artifacts that data/manifest.json catalogs but that are
# not yet on the buckets:
#   * Shorkie_Random_Init model (lr 5e-4, 8-fold)  -> gs://seqnn-share/shorkie_random_init/  (PUBLIC)
#   * eQTL scores + DREAM baselines (Figure 7)       -> gs://shorkie-paper/eqtl/   (REQUESTER-PAYS)
#   * MPRA ground-truth + subset ids + scores (Fig 6)-> gs://shorkie-paper/mpra/   (REQUESTER-PAYS)
#
# RUN THIS YOURSELF with your gcloud credentials (writes to the shared/public
# buckets). Source paths resolve from config/paths.yaml via `shorkie.config`.
# Idempotent: uses `gsutil cp -n` (skip already-present objects). Run with
# --dry-run first to print the exact commands; then re-run `verify_release.py`.
#
# Usage:
#   scripts/00_setup/upload_release.sh [--models|--eqtl|--mpra|--all] [-u PROJECT] [--dry-run]
#     -u PROJECT  GCP billing project for the requester-pays data bucket
#                 (default: `gcloud config get-value project`)
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
cd "$REPO"

WHAT="all"; PROJECT=""; DRY=0
while [[ $# -gt 0 ]]; do case "$1" in
  --models) WHAT=models; shift;; --eqtl) WHAT=eqtl; shift;; --mpra) WHAT=mpra; shift;;
  --all) WHAT=all; shift;; -u|--project) PROJECT="$2"; shift 2;; --dry-run) DRY=1; shift;;
  -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0;; *) echo "unknown arg: $1" >&2; exit 2;; esac; done

GSUTIL="$(command -v gsutil || echo /home/kchao10/data_ssalzbe1/khchao/google-cloud-sdk/bin/gsutil)"
[[ -n "$PROJECT" ]] || PROJECT="$("${GSUTIL%gsutil}gcloud" config get-value project 2>/dev/null || true)"
cfg() { python -c "import sys; from shorkie import config; print(config.path(sys.argv[1]))" "$1"; }
W="$(cfg work_root)"

# cp helpers: model bucket is public; data bucket is requester-pays (-u PROJECT).
mcp() { local src="$1" dst="$2"; [[ -e "$src" ]] || { echo "  SKIP (missing): $src" >&2; return 0; }
  echo "+ gsutil cp -n $src $dst"; [[ "$DRY" == 1 ]] || "$GSUTIL" cp -n "$src" "$dst"; }
dcp() { local src="$1" dst="$2"; [[ -e "$src" ]] || { echo "  SKIP (missing): $src" >&2; return 0; }
  [[ -n "$PROJECT" ]] || { echo "requester-pays needs -u PROJECT" >&2; exit 2; }
  echo "+ gsutil -u $PROJECT cp -n -r $src $dst"; [[ "$DRY" == 1 ]] || "$GSUTIL" -u "$PROJECT" cp -n -r "$src" "$dst"; }

if [[ "$WHAT" == "models" || "$WHAT" == "all" ]]; then
  echo "=== Shorkie_Random_Init (lr 5e-4, 8-fold) -> gs://seqnn-share/shorkie_random_init/ ==="
  RI="$(cfg models.shorkie_random_init)/train"
  for f in 0 1 2 3 4 5 6 7; do
    mcp "$RI/f${f}c0/train/model_best.h5" "gs://seqnn-share/shorkie_random_init/f${f}/model_best.h5"
  done
  mcp "$RI/f0c0/train/params.json" "gs://seqnn-share/shorkie_random_init/params.json"
fi

if [[ "$WHAT" == "eqtl" || "$WHAT" == "all" ]]; then
  echo "=== eQTL scores + DREAM baselines -> gs://shorkie-paper/eqtl/ ==="
  dcp "$(cfg results.eqtl_scores)" "gs://shorkie-paper/eqtl/scores/"      # viz_new/results (Shorkie family)
  dcp "$(cfg results.mpra_eval)"   "gs://shorkie-paper/eqtl/dream_eval/"  # DREAM eQTL baselines
fi

if [[ "$WHAT" == "mpra" || "$WHAT" == "all" ]]; then
  echo "=== MPRA ground-truth + subset ids + scores -> gs://shorkie-paper/mpra/ ==="
  MPRA="$(cfg datasets.mpra)"
  dcp "$MPRA/filtered_test_data_with_MAUDE_expression.txt" "gs://shorkie-paper/mpra/ground_truth/"
  dcp "$MPRA/test_subset_ids"                              "gs://shorkie-paper/mpra/test_subset_ids/"
  dcp "$W/experiments/SUM_data_process/MPRA/MPRA_promoter_seqs/results/single_measurement_stranded/all_seq_types" \
      "gs://shorkie-paper/mpra/scores/single_measurement_stranded/all_seq_types/"
  dcp "$W/experiments/SUM_data_process/MPRA/MPRA_RNASeq/predictions/upstream_180bp_predictions.tsv" \
      "gs://shorkie-paper/mpra/scores/"
  dcp "$W/data/random-promoter-dream-challenge-2022/data/DREAM-RNN_output.txt" "gs://shorkie-paper/mpra/dream/"
fi

echo "=== done. Verify with: python scripts/00_setup/verify_release.py -u $PROJECT ==="
