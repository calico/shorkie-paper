# `reproduction/` — manuscript figure reproduction audit

A manuscript-figure-anchored layer that audits, traces, reproduces, and **verifies** every panel of the Shorkie paper's main-text figures against the published PDFs (`../paper/`). It is the **deep audit counterpart** to the user-facing notebooks in `../notebooks/` (`fig01`–`fig07`, one per main-text figure): each figure's notebook lives there and calls this layer's `recheck/build_*.py` builders. This layer is organized by **manuscript figure → panel** and adds the published-panel crops + a reproduced-vs-published verification step.

## Layout

```
reproduction/
├── AUDIT_MATRIX.md      ← master: every figure → panel → claim → script → data+config-key → env → GPU → notebook → status
├── common/
│   ├── extract_panels.py   rasterize/crop published panels from ../paper/Figures/Figure_N.pdf
│   ├── compare.py          numeric verification helpers (reported vs reproduced → verdict CSV)
│   └── env_check.py        assert tools (mummer/mash/ete/...) + config keys + data presence
└── figure_01/ … figure_07/
    ├── README.md           Phase-1 Discovery (per panel) + Phase-3 Verification report
    ├── recheck/            panel builders (build_*.py) + verify (build_verify_*.py) + DISCREPANCIES.md
    ├── panels/             CLI scripts for heavy/GPU panels (run_mummer.sh, run_coverage.py, run_ism_*.py)
    ├── published/          reference panel crops from the PDF
    └── reproduced/         regenerated panels + reproduced-vs-reported verify_figNN.csv
```
The **user-facing notebook** for each figure is `../notebooks/fig0N_<topic>.ipynb`; it calls this
figure's `recheck/build_*.py` (single source of truth) and renders the panels.

## Method (4 phases per figure)

1. **Discovery** — for each panel: the exact scientific claim (from the caption), the generating script (traced to this repo + its `Yeast_ML` origin), the input artifact + `shorkie.config` key, the conda env, GPU need, and on-disk availability. → `figure_NN/README.md`.
2. **Reproduction** — the user-facing notebook `../notebooks/fig0N_<topic>.ipynb` (paths via `shorkie.config`, helpers from `shorkie.*`, `yeast_ml` kernel) regenerates each panel by invoking this figure's `recheck/build_*.py` builders; heavy external-tool / GPU panels run via `panels/*`. Changes made to legacy scripts are logged in the figure README.
3. **Verification** — reproduced panel vs published crop (visual), and recomputed summary stats vs the values reported in the manuscript (numeric → `reproduced/verify_figNN.csv`); discrepancies logged with likely cause.
4. **Delivery** — finalize, update `AUDIT_MATRIX.md` status, commit.

## Prerequisites

- conda env `yeast_ml` (`../environment.yml`); `pip install -e ..` for the `shorkie` package.
- `cp ../config/paths.example.yaml ../config/paths.yaml` and set `work_root`.
- Heavy tools for Figure 1: `mummer` (MUMmer4), `mash`, `ete3`/`ete4` (most via bioconda).
- Run `python common/env_check.py` to see what's missing before reproducing a figure.

## Scope

This pass covers the **7 main-text figures** (~64 panels). Supplementary figures S1–S29 are catalogued in `AUDIT_MATRIX.md` but not reproduced here. Schematic panels (architecture/pipeline diagrams) are reproduced from their programmatic source where one exists and otherwise documented as hand-drawn; heavy-external panels are fully recomputed from on-disk genomes.
