# Figure 1 — Datasets, preprocessing, architecture & performance of the fungal LM

> *"Overview of datasets, preprocessing pipeline, model architecture, and performance metrics for the fungal language model (Shorkie LM)."*

Reproduces main-text **Figure 1** of the Shorkie paper. Published reference: [`published/Figure_1_full.png`](published/Figure_1_full.png) — rendered from Figure 1 of the paper
([bioRxiv preprint](https://doi.org/10.1101/2025.09.19.677475); the manuscript PDFs themselves are not redistributed in this repo).

- **Run:** [`notebooks/fig01_fungal_lm_corpus_architecture.ipynb`](../../notebooks/fig01_fungal_lm_corpus_architecture.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig01.csv`](reproduced/verify_fig01.csv) (all PASS).
