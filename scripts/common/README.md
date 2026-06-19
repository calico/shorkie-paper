# scripts/common — shared shell infrastructure

Two small, site-agnostic helpers that every pipeline stage sources or submits
through, so the per-stage scripts carry no hardcoded paths or scheduler config.

| File | Purpose |
|------|---------|
| `env.sh` | `source` it to export path roots resolved from `config/paths.yaml` via the installed `shorkie` package (e.g. `WORK_ROOT`, `DATA_ROOT`, `CORPUS_BUILD_DATA_ROOT`, `CORPUS_BUILD_RESULTS_ROOT`, `BASKERVILLE_SCRIPTS`, `WESTMINSTER_SCRIPTS`, `MOTIF_DB_DIR`, tool binaries). Scripts reference `${...}` instead of absolute paths. |
| `submit.sh` | Portable SLURM submit wrapper. Emits `#SBATCH` directives from `config/slurm.yaml` keyed by `--profile {gpu\|cpu\|bigmem}`, then `exec sbatch`s the target script. Set `SHORKIE_LOCAL=1` to skip the scheduler and run the script directly (workstation / container fallback). |

## Usage

```bash
# Export path roots into the current shell.
source scripts/common/env.sh

# Submit a stage script to SLURM with a profile (job-name/array stay with the caller).
scripts/common/submit.sh --profile gpu --job-name train --array 0-7 some_stage.sh [args...]

# Run the same script with no scheduler.
SHORKIE_LOCAL=1 scripts/common/submit.sh --profile gpu some_stage.sh [args...]
```

## Config keys read

- `env.sh` calls `shorkie.config.get(...)` for the path keys mapped at the top of
  the file (`work_root`, `data_root`, `corpus_build_data_root`,
  `external.baskerville_scripts`, `motif_db_dir`, etc.). Copy
  `config/paths.example.yaml` → `config/paths.yaml` and edit for your site.
- `submit.sh` reads `email`, `mail_type`, `default_time`, `default_mem`, and the
  `profiles.{gpu,cpu,bigmem}` blocks (`partition`, `account`, `gres`, `time`,
  `mem`, `ntasks_per_node`) from `config/slurm.yaml`. Copy
  `config/slurm.example.yaml` → `config/slurm.yaml` and edit. Blank `email`
  disables `--mail-user`.

No external bioinformatics tools are invoked here; both helpers depend only on
`bash`, the `shorkie` Python package, and (for non-local runs) `sbatch`.
