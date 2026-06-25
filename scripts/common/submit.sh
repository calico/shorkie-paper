#!/usr/bin/env bash
# Portable SLURM submit wrapper for shorkie-paper.
#
# Emits #SBATCH directives from config/slurm.yaml (keyed by --profile), so the
# pipeline scripts carry no site-specific scheduler config. The per-script
# job-name / array spec stay with the caller.
#
# Usage:
#   scripts/common/submit.sh --profile {gpu|cpu|bigmem} \
#       [--job-name NAME] [--array SPEC] [--time T] [--mem M] script.sh [args...]
#
# Run locally WITHOUT a scheduler (e.g. on a workstation or inside a container):
#   SHORKIE_LOCAL=1 scripts/common/submit.sh --profile gpu script.sh [args...]
set -euo pipefail

PROFILE=gpu; JOBNAME=""; ARRAY=""; TIME=""; MEM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)  PROFILE="$2"; shift 2;;
    --job-name) JOBNAME="$2"; shift 2;;
    --array)    ARRAY="$2";   shift 2;;
    --time)     TIME="$2";    shift 2;;
    --mem)      MEM="$2";     shift 2;;
    --)         shift; break;;
    *)          break;;
  esac
done
[[ $# -ge 1 ]] || { echo "submit.sh: missing script to run" >&2; exit 2; }
SCRIPT="$1"; shift

# Resolve the profile + global mail/env settings from config/slurm.yaml.
eval "$(python - "$PROFILE" <<'PY'
import sys, shlex
from shorkie import config
c = config.load()
prof = c.get(f"profiles.{sys.argv[1]}") or {}
def out(k, v): print(f"{k}={shlex.quote(str(v))}")
out("P_PARTITION", prof.get("partition", ""))
out("P_ACCOUNT",   prof.get("account", ""))
out("P_GRES",      prof.get("gres", ""))
out("P_TIME",      prof.get("time", c.get("default_time", "48:00:00")))
out("P_MEM",       prof.get("mem",  c.get("default_mem", "32G")))
out("P_NTASKS",    prof.get("ntasks_per_node", 1))
out("P_EMAIL",     c.get("email", ""))
out("P_MAILTYPE",  c.get("mail_type", "end"))
PY
)"
TIME="${TIME:-$P_TIME}"; MEM="${MEM:-$P_MEM}"

# Local fallback: just run the script (no scheduler).
if [[ "${SHORKIE_LOCAL:-0}" == "1" ]]; then
  exec bash "$SCRIPT" "$@"
fi

ARGS=(--partition="$P_PARTITION" --account="$P_ACCOUNT" --time="$TIME" --mem="$MEM"
      --nodes=1 --ntasks-per-node="$P_NTASKS" --export=ALL)
[[ -n "$P_GRES"   ]] && ARGS+=(--gres="$P_GRES")
[[ -n "$JOBNAME"  ]] && ARGS+=(--job-name="$JOBNAME")
[[ -n "$ARRAY"    ]] && ARGS+=(--array="$ARRAY")
[[ -n "$P_EMAIL"  ]] && ARGS+=(--mail-user="$P_EMAIL" --mail-type="$P_MAILTYPE")
exec sbatch "${ARGS[@]}" "$SCRIPT" "$@"
