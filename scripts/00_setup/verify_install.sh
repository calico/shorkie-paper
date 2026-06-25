#!/usr/bin/env bash
# Quick install check for the shorkie package + reproducibility env.
# Run after `pip install -e .` (env yeast_ml). Exits non-zero on any failure.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO"

echo "== import shorkie =="
python -c "import shorkie; from shorkie import config; from shorkie.models import ensemble; \
print('shorkie OK  | repo:', config.repo_root()); print('ensemble.NUM_FEATURES =', ensemble.NUM_FEATURES)"

echo "== config resolves the released models =="
python -c "from shorkie import config; \
[print(' ', k, '->', config.path(k)) for k in ('models.shorkie_lm','models.shorkie_finetuned','models.shorkie_random_init')]"

echo "== pytest smoke tests =="
python -m pytest -q tests/

echo "== OK =="
