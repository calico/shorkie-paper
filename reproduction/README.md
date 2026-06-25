# `reproduction/` — paper figure reproduction

Code and reference crops to regenerate every panel of the Shorkie paper's seven main-text figures
and check them against the published PDFs in `../paper/`. The user-facing entry point for each figure
is its notebook in [`../notebooks/`](../notebooks/) (`fig01`–`fig07`); this directory holds the
single-source panel builders those notebooks call, plus the published crops and the numeric checks.

## Layout

```
reproduction/
├── common/                  shared helpers (panel extraction, numeric compare, env checks)
└── figure_01/ … figure_07/
    ├── README.md            what the figure shows + how to run it
    ├── recheck/             panel builders (build_*.py) — single source of truth, called by the notebook
    ├── panels/              CLI scripts for heavy / GPU panels
    ├── published/           reference panel crops from ../paper/Figures/Figure_N.pdf
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
