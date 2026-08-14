# Figure 6 — MPRA promoter variant effects

> *"Shorkie predicts promoter variant effects validated by MPRAs."*

Reproduces main-text **Figure 6** of the Shorkie paper. Published reference: [`published/Figure_6_full.png`](published/Figure_6_full.png) — rendered from Figure 6 of the paper
([bioRxiv preprint](https://doi.org/10.1101/2025.09.19.677475); the manuscript PDFs themselves are not redistributed in this repo).

- **Run:** [`notebooks/fig06_mpra_variant_effects.ipynb`](../../notebooks/fig06_mpra_variant_effects.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig06.csv`](reproduced/verify_fig06.csv) (all PASS).
