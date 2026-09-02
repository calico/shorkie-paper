#!/bin/bash
# Revision 04 step 1 - recompute the coding and repetitive fractions of S. cerevisiae R64.
#
# Run locally (no scheduler needed):
#   bash scripts/05_revision/04_loss_weight_provenance/1_recompute_annotation_fractions.sh
# or submit:
#   scripts/common/submit.sh --profile cpu scripts/05_revision/04_loss_weight_provenance/1_recompute_annotation_fractions.sh
# Add --dry-run to print the fully-resolved command without running it.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/common/env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY_RUN=1; else ARGS+=("$a"); fi
done

CMD=(python "$HERE/1_recompute_annotation_fractions.py" "${ARGS[@]+"${ARGS[@]}"}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
