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
    "WORK_ROOT":                 "work_root",
    "DATA_ROOT":                 "data_root",
    "SEQ_EXPERIMENT_ROOT":       "seq_experiment_root",
    "LM_EXPERIMENT_ROOT":        "lm_experiment_root",
    "EXPERIMENTS_ROOT":          "experiments_root",
    "RELEASE_ROOT":              "release_root",
    "BASKERVILLE_SCRIPTS":       "external.baskerville_scripts",
    "WESTMINSTER_SCRIPTS":       "external.westminster_scripts",
    # Legacy data-build / genome-eval working roots (LM-corpus build + genome eval).
    "CORPUS_BUILD_DATA_ROOT":    "corpus_build_data_root",
    "CORPUS_BUILD_RESULTS_ROOT": "corpus_build_results_root",
    "SSM_RESULTS_ROOT":          "ssm_results_root",
    "YEAST_SEQNN_EVAL_ROOT":     "yeast_seqnn_eval_root",
    # External-tool binaries / library files (RepeatMasker, RMRB, dashing2, MEME db).
    "REPEATMASKER_LIB":          "tools.repeatmasker_lib",
    "RMRB_LIB":                  "tools.rmrb_lib",
    "DASHING2_BIN":              "tools.dashing2_bin",
    "MOTIF_DB_DIR":              "motif_db_dir",
}
for var, key in mapping.items():
    val = config.get(key)
    if val is not None:
        print(f"export {var}={shlex.quote(str(val))}")
PY
)"
