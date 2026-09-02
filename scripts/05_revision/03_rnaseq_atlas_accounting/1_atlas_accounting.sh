#!/bin/bash
# Revision 03 step 1 - reconcile the 3,053 induction RNA-seq tracks from the targets sheet.
#
# Run locally (no scheduler needed):
#   bash scripts/05_revision/03_rnaseq_atlas_accounting/1_atlas_accounting.sh
# or submit:
#   scripts/common/submit.sh --profile cpu scripts/05_revision/03_rnaseq_atlas_accounting/1_atlas_accounting.sh
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

CMD=(python "$HERE/1_atlas_accounting.py" "${ARGS[@]+"${ARGS[@]}"}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
