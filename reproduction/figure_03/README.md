# Figure 3 — Shorkie architecture and RNA-seq prediction performance

> *"Shorkie architecture and RNA-seq prediction performance across multiple scales."*

Reproduces main-text **Figure 3** of the Shorkie paper. Published reference: [`published/Figure_3_full.png`](published/Figure_3_full.png) — rendered from Figure 3 of the paper
([bioRxiv preprint](https://doi.org/10.1101/2025.09.19.677475); the manuscript PDFs themselves are not redistributed in this repo).

- **Run:** [`notebooks/fig03_supervised_rnaseq_prediction.ipynb`](../../notebooks/fig03_supervised_rnaseq_prediction.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig03.csv`](reproduced/verify_fig03.csv) (all PASS).
