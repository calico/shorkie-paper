#!/bin/bash
# Revision 08 step 2 - recompute Figure 2E embeddings under length-controlled schemes (GPU).
#
# Run locally (no scheduler needed):
#   bash scripts/05_revision/08_tsne_length_control/2_embed_variants.sh
# or submit:
#   scripts/common/submit.sh --profile gpu scripts/05_revision/08_tsne_length_control/2_embed_variants.sh
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

CMD=(python "$HERE/2_embed_variants.py" "${ARGS[@]+"${ARGS[@]}"}")
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
