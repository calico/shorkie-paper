# `reproduction/` — paper figure reproduction

Code and reference crops to regenerate every panel of the Shorkie paper's seven main-text figures
and check them against the published figures. (The manuscript PDFs are not redistributed in this repo; each figure's `published/` holds the reference crops rendered from them.) The user-facing entry point for each figure
is its notebook in [`../notebooks/`](../notebooks/) (`fig01`–`fig07`); this directory holds the
single-source panel builders those notebooks call, plus the published crops and the numeric checks.

## Layout

```
reproduction/
├── common/                  shared helpers (panel extraction, numeric compare, env checks)
├── recheck/                 CROSS-figure audit layer: recompute_recheck.py independently re-derives
│                            headline numbers straight from the on-disk eval artifacts (8/8), plus
│                            determinism.csv (a fresh headless re-execution log) and coverage tables
└── figure_01/ … figure_07/
    ├── README.md            what the figure shows + how to run it
    ├── recheck/             panel builders (build_*.py) — single source of truth, called by the notebook
    ├── panels/              CLI scripts for heavy / GPU panels
    ├── published/           reference panel crops rendered from the paper's figures
    └── reproduced/          regenerated panels + verify_figNN.csv (reproduced-vs-published checks)
```

## Running

- conda env `yeast_ml` (`../environment.yml`); `pip install -e ..` for the `shorkie` package.
- `cp ../config/paths.example.yaml ../config/paths.yaml` and set `work_root`.
- Open the figure's notebook in `../notebooks/`; it resolves paths via `shorkie.config` and calls
  this figure's `recheck/build_*.py`. Heavy external-tool / GPU panels run via `figure_NN/panels/*`.
- `python common/env_check.py` reports any missing tools (e.g. `mummer`, `mash`, `ete4`) before you start.

Scope: the 7 main-text figures (~64 panels). Schematic panels are reproduced from source where one
exists and otherwise noted as hand-drawn; heavy external-tool panels are recomputed from on-disk genomes.
