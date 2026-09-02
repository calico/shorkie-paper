#!/bin/bash
# Revision 05 step 3 - verify and correct the Figure 5 D/E panel labelling.
#
# Run locally (no scheduler needed):
#   bash scripts/05_revision/05_figure_and_caption_fixes/3_fig5_caption_fix.sh
# or submit:
#   scripts/common/submit.sh --profile cpu scripts/05_revision/05_figure_and_caption_fixes/3_fig5_caption_fix.sh
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

CMD=(python "$HERE/3_fig5_caption_fix.py" "${ARGS[@]+"${ARGS[@]}"}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
