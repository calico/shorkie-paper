#!/bin/bash
# Revision 09 step 2 - test dependency against a distance-matched background.
#
# Run locally (no scheduler needed):
#   bash scripts/05_revision/09_dependency_maps/2_quantify_dependency.sh
# or submit:
#   scripts/common/submit.sh --profile cpu scripts/05_revision/09_dependency_maps/2_quantify_dependency.sh
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

CMD=(python "$HERE/2_quantify_dependency.py" "${ARGS[@]+"${ARGS[@]}"}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
