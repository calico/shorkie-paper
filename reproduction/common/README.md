# reproduction/common/

Shared helpers used by the per-figure `recheck/` builders. Not run directly.

| Script | What |
|--------|------|
| `compare.py` | Numeric published-vs-reproduced comparison; writes the `verify_figNN.csv` verdict rows. |
| `env_check.py` | Reports the resolved config/environment a figure run actually used. |
| `extract_panels.py` | Crops panels out of a published figure render. |
| `run_in_tmux.sh` | Launches a long recompute in a detached tmux session. |

See [`../README.md`](../README.md) for the reproduction layout.
