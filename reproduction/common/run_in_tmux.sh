#!/bin/bash
# Launch a long-running command in a DETACHED tmux session so it keeps running
# even if the interactive/Claude session disconnects. Writes the command's stdout+stderr
# to <logfile> and appends a final "DONE_EXIT_<code>" line on completion (poll for it).
#
# Usage:
#   reproduction/common/run_in_tmux.sh <session_name> <logfile> <command string...>
# Example:
#   reproduction/common/run_in_tmux.sh fig03_exec /tmp/fig03.log \
#     "conda run -n yeast_ml jupyter nbconvert --to notebook --execute --inplace \
#      --ExecutePreprocessor.startup_timeout=240 --ExecutePreprocessor.timeout=3600 \
#      --ExecutePreprocessor.kernel_name=yeast_ml reproduction/figure_03/reproduce_figure_03.ipynb"
#
# Check progress:  tmux attach -t <session_name>   (Ctrl-b d to detach)
#                  tail -f <logfile>
#                  grep DONE_EXIT <logfile>         (present == finished)
set -euo pipefail
SESS="$1"; LOG="$2"; shift 2
CMD="$*"
# Persist the command to a script so tmux quoting can't mangle it.
CMDFILE="$(mktemp /tmp/tmux_${SESS}_XXXX.sh)"
{
  echo '#!/bin/bash'
  echo "cd '$(pwd)'"
  echo "source /home/kchao10/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true"
  echo "$CMD"
  echo 'echo "DONE_EXIT_$?"'
} > "$CMDFILE"
chmod +x "$CMDFILE"
# Kill any stale session of the same name, then launch detached.
tmux kill-session -t "$SESS" 2>/dev/null || true
tmux new-session -d -s "$SESS" "bash '$CMDFILE' > '$LOG' 2>&1"
echo "[tmux] launched session '$SESS' -> log $LOG (cmd: $CMDFILE)"
echo "[tmux] watch: tail -f $LOG  | done when: grep -q DONE_EXIT $LOG"
