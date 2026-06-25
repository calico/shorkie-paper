# Figure 7 — Shorkie accurately predicts cis-eQTL variant effects

> *"Shorkie accurately predicts cis-eQTL variant effects."*

Reproduces main-text **Figure 7** of the Shorkie paper. Published reference:
[`../../paper/Figures/Figure_7.pdf`](../../paper/Figures/Figure_7.pdf) (rendered to `published/Figure_7_full.png`).

- **Run:** [`notebooks/fig07_eqtl_variant_effects.ipynb`](../../notebooks/fig07_eqtl_variant_effects.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig07.csv`](reproduced/verify_fig07.csv) (all PASS).
