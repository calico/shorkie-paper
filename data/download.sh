#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch + verify released Shorkie artifacts listed in data/manifest.json.
# ---------------------------------------------------------------------------
# Model weights live in the PUBLIC bucket gs://seqnn-share (downloadable with
# gsutil or plain https). Datasets live in the REQUESTER-PAYS bucket
# gs://shorkie-paper — those need gsutil and a billing project (pass -u PROJECT).
#
# Usage:
#   data/download.sh --minimal                  # 8 fine-tuned folds -> ./my_shorkie
#                                                #   (exactly what minimal_example expects)
#   data/download.sh --models [lm|finetuned|random_init|all]  # weights -> <dest>/models/...
#   data/download.sh --lm-corpus <tier|all> -u PROJECT     # corpus genomes + TFRecords
#   data/download.sh --supervised [bigwigs|tfrecords|all] -u PROJECT
#   data/download.sh --eqtl -u PROJECT          # Figure-7 eQTL scores + DREAM baselines
#   data/download.sh --mpra [meta|scores|all] -u PROJECT     # Figure-6 MPRA data
#
#   --dest DIR    base output dir (default: config release_root)
#   -u PROJECT    GCP billing project for the requester-pays data bucket
#   --dry-run     print actions without downloading
#
# Tiers: r64-style names follow the manifest (R64, 80_strains,
# 165_Saccharomycetales, 1341_Fungus).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.json"
REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$HERE/..")"

MODE=""; SEL="all"; DEST=""; DEST_SET=0; PROJECT=""; DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --minimal)     MODE=minimal; shift;;
    --models)      MODE=models;     [[ "${2:-}" =~ ^(lm|finetuned|random_init|all)$ ]] && { SEL="$2"; shift; }; shift;;
    --lm-corpus)   MODE=lm_corpus;  SEL="${2:?--lm-corpus needs a tier or 'all'}"; shift 2;;
    --supervised)  MODE=supervised; [[ "${2:-}" =~ ^(bigwigs|tfrecords|all)$ ]] && { SEL="$2"; shift; }; shift;;
    --eqtl)        MODE=eqtl; shift;;
    --mpra)        MODE=mpra;       [[ "${2:-}" =~ ^(meta|scores|all)$ ]] && { SEL="$2"; shift; }; shift;;
    --dest)        DEST="$2"; DEST_SET=1; shift 2;;
    -u|--project)  PROJECT="$2"; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    -h|--help)     sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -n "$MODE" ]] || { echo "pick a mode: --minimal | --models | --lm-corpus | --supervised | --eqtl | --mpra (see --help)" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }

# Default dest = config release_root (falls back to ./data_local if config absent).
if [[ -z "$DEST" ]]; then
  DEST="$(python -c "from shorkie import config; print(config.get('release_root') or './data_local')" 2>/dev/null || echo ./data_local)"
fi

# Locate gsutil (PATH, then the known SDK install).
GSUTIL="$(command -v gsutil || true)"
[[ -z "$GSUTIL" && -x /home/kchao10/data_ssalzbe1/khchao/google-cloud-sdk/bin/gsutil ]] \
  && GSUTIL=/home/kchao10/data_ssalzbe1/khchao/google-cloud-sdk/bin/gsutil
FETCH="$(command -v wget || true)"

say() { echo "+ $*"; }

# Download one file (public bucket): gsutil if available, else https via wget/curl.
get_file() {  # gs_uri https_uri out_path md5
  local gs="$1" https="$2" out="$3" md5="$4"
  mkdir -p "$(dirname "$out")"
  if [[ "$DRY_RUN" == 1 ]]; then
    say "fetch $gs -> $out"; return 0
  fi
  if [[ -n "$GSUTIL" ]]; then
    say "gsutil cp $gs $out"; "$GSUTIL" cp "$gs" "$out"
  elif [[ -n "$FETCH" ]]; then
    say "wget $https -> $out"; wget -q -O "$out" "$https"
  else
    say "curl $https -> $out"; curl -fsSL -o "$out" "$https"
  fi
  if [[ -n "$md5" && "$md5" != "null" ]]; then
    local got; got="$(md5sum "$out" | awk '{print $1}')"
    if [[ "$got" == "$md5" ]]; then echo "  md5 OK  $out"
    else echo "  md5 FAIL $out (got $got want $md5)" >&2; return 1; fi
  fi
}

# Recursively copy a requester-pays prefix (datasets) with gsutil.
get_prefix() {  # gs_prefix out_dir
  local gs="$1" out="$2"
  if [[ "$DRY_RUN" == 1 ]]; then say "gsutil -u ${PROJECT:-PROJECT} cp -r $gs $out"; return 0; fi
  [[ -n "$GSUTIL" ]] || { echo "gsutil required for $gs (requester-pays)" >&2; return 1; }
  [[ -n "$PROJECT" ]] || { echo "requester-pays bucket needs -u PROJECT for $gs" >&2; return 1; }
  mkdir -p "$out"
  say "gsutil -u $PROJECT cp -r $gs $out"
  "$GSUTIL" -u "$PROJECT" cp -r "$gs" "$out"
  echo "  (integrity verified by gsutil on transfer)"
}

# Emit "gs_uri<TAB>https_uri<TAB>local_path<TAB>md5" rows for a model selection.
emit_model_files() {  # selection strip_prefix
  python - "$MANIFEST" "$1" "${2:-}" <<'PY'
import json, sys
manifest, sel, strip = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(manifest))["models"]
names = {"lm": ["shorkie_lm"], "finetuned": ["shorkie_finetuned"],
         "random_init": ["shorkie_random_init"],
         "all": ["shorkie_lm", "shorkie_finetuned", "shorkie_random_init"]}[sel]
for n in names:
    for f in m[n]["files"]:
        lp = f["local_path"]
        if strip and lp.startswith(strip):
            lp = lp[len(strip):]
        print("\t".join([f["gs_uri"], f.get("https_uri", ""), lp, str(f.get("md5") or "null")]))
PY
}

case "$MODE" in
  minimal)
    # Only the 8 fine-tuned folds + params, into ./my_shorkie/train/f{i}c0/train/...
    OUT="./my_shorkie"; [[ "$DEST_SET" == 1 ]] && OUT="$DEST"
    echo "=== minimal: fine-tuned ensemble -> $OUT (for minimal_example/) ==="
    emit_model_files finetuned "models/shorkie_finetuned/" | while IFS=$'\t' read -r gs https lp md5; do
      get_file "$gs" "$https" "$OUT/$lp" "$md5"
    done
    echo "Run: python minimal_example/run_shorkie_variant.py --model_dir $OUT --params_file minimal_example/params.json --targets_file minimal_example/sheet.txt ..."
    ;;
  models)
    echo "=== models ($SEL) -> $DEST ==="
    emit_model_files "$SEL" | while IFS=$'\t' read -r gs https lp md5; do
      get_file "$gs" "$https" "$DEST/$lp" "$md5"
    done
    ;;
  lm_corpus)
    echo "=== lm_corpus ($SEL) -> $DEST/data/unsupervised/ (requester-pays) ==="
    python - "$MANIFEST" "$SEL" <<'PY' | while IFS=$'\t' read -r gs sub; do get_prefix "$gs" "$DEST/data/unsupervised/$sub"; done
import json, sys
m = json.load(open(sys.argv[1]))["datasets"]["lm_corpus"]["tiers"]
sel = sys.argv[2]
tiers = m.keys() if sel == "all" else [sel]
for t in tiers:
    if t not in m: sys.exit(f"unknown tier '{t}' (choose: {', '.join(m)}, or all)")
    print("\t".join([m[t]["genome"], f"genome/{t}/"]))
    print("\t".join([m[t]["tfrecords"], f"processed/{t}/"]))
PY
    ;;
  supervised)
    echo "=== supervised ($SEL) -> $DEST/data/supervised/ (requester-pays) ==="
    python - "$MANIFEST" "$SEL" <<'PY' | while IFS=$'\t' read -r gs sub; do get_prefix "$gs" "$DEST/data/supervised/$sub"; done
import json, sys
s = json.load(open(sys.argv[1]))["datasets"]["supervised"]; sel = sys.argv[2]
if sel in ("bigwigs", "all"):   print("\t".join([s["bigwigs"]["gs_uri"], "bigwigs/"]))
if sel in ("tfrecords", "all"): print("\t".join([s["tfrecords"]["gs_uri"], "processed/"]))
PY
    ;;
  eqtl)
    echo "=== eqtl (Figure 7 scores + DREAM baselines) -> $DEST/eqtl/ (requester-pays) ==="
    python - "$MANIFEST" <<'PY' | while IFS=$'\t' read -r gs sub; do get_prefix "$gs" "$DEST/eqtl/$sub"; done
import json, sys
e = json.load(open(sys.argv[1]))["datasets"]["eqtl"]
print("\t".join([e["scores"]["gs_uri"], "scores/"]))
print("\t".join([e["dream_eval"]["gs_uri"], "dream_eval/"]))
PY
    ;;
  mpra)
    echo "=== mpra ($SEL) -> $DEST/mpra/ (requester-pays) ==="
    python - "$MANIFEST" "$SEL" <<'PY' | while IFS=$'\t' read -r gs sub; do get_prefix "$gs" "$DEST/mpra/$sub"; done
import json, sys
m = json.load(open(sys.argv[1]))["datasets"]["mpra"]; sel = sys.argv[2]
if sel in ("meta", "all"):
    print("\t".join([m["ground_truth"]["gs_uri"], "ground_truth/"]))
    print("\t".join([m["test_subset_ids"]["gs_uri"], "test_subset_ids/"]))
    print("\t".join([m["dream_rnn"]["gs_uri"], "dream/"]))
if sel in ("scores", "all"):
    print("\t".join([m["scores"]["gs_uri"], "scores/"]))
PY
    ;;
esac
echo "=== done ==="
