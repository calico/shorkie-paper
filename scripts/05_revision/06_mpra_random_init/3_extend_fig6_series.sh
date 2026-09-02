#!/bin/bash
# Revision 06 step 3 - add Shorkie_Random_Init as a third series to the Figure 6 panels.
#
# Run locally (no scheduler needed):
#   bash scripts/05_revision/06_mpra_random_init/3_extend_fig6_series.sh
# or submit:
#   scripts/common/submit.sh --profile cpu scripts/05_revision/06_mpra_random_init/3_extend_fig6_series.sh
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

CMD=(python "$HERE/3_extend_fig6_series.py" "${ARGS[@]+"${ARGS[@]}"}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
