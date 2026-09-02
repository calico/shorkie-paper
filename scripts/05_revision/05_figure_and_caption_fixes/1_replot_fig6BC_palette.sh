#!/bin/bash
# Revision 05 step 1 - redraw Figure 6B/6C with an accessible, luminance-separated palette.
#
# Run locally (no scheduler needed):
#   bash scripts/05_revision/05_figure_and_caption_fixes/1_replot_fig6BC_palette.sh
# or submit:
#   scripts/common/submit.sh --profile cpu scripts/05_revision/05_figure_and_caption_fixes/1_replot_fig6BC_palette.sh
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

CMD=(python "$HERE/1_replot_fig6BC_palette.py" "${ARGS[@]+"${ARGS[@]}"}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
