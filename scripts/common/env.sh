#!/usr/bin/env bash
# Export shorkie path variables resolved from config/paths.yaml.
#
#   source scripts/common/env.sh
#
# Uses the installed `shorkie` package (single source of truth for paths, with
# ${...} interpolation and env-var overrides). Pipeline scripts reference the
# exported variables (e.g. ${DATA_ROOT}, ${BASKERVILLE_SCRIPTS}) instead of
# hardcoding absolute paths.
eval "$(python - <<'PY'
import shlex
from shorkie import config
mapping = {
    "WORK_ROOT":           "work_root",
    "DATA_ROOT":           "data_root",
    "SEQ_EXPERIMENT_ROOT": "seq_experiment_root",
    "LM_EXPERIMENT_ROOT":  "lm_experiment_root",
    "EXPERIMENTS_ROOT":    "experiments_root",
    "RELEASE_ROOT":        "release_root",
    "BASKERVILLE_SCRIPTS": "external.baskerville_scripts",
    "WESTMINSTER_SCRIPTS": "external.westminster_scripts",
}
for var, key in mapping.items():
    val = config.get(key)
    if val is not None:
        print(f"export {var}={shlex.quote(str(val))}")
PY
)"
