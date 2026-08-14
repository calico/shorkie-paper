# Figure 7 — Shorkie accurately predicts cis-eQTL variant effects

> *"Shorkie accurately predicts cis-eQTL variant effects."*

Reproduces main-text **Figure 7** of the Shorkie paper. Published reference: [`published/Figure_7_full.png`](published/Figure_7_full.png) — rendered from Figure 7 of the paper
([bioRxiv preprint](https://doi.org/10.1101/2025.09.19.677475); the manuscript PDFs themselves are not redistributed in this repo).

- **Run:** [`notebooks/fig07_eqtl_variant_effects.ipynb`](../../notebooks/fig07_eqtl_variant_effects.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig07.csv`](reproduced/verify_fig07.csv) (all PASS).
