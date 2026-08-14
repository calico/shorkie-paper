# Figure 4 — Promoter & splicing motifs learned during pretraining

> *"Shorkie uses promoter and splicing motifs learned during pretraining."*

Reproduces main-text **Figure 4** of the Shorkie paper. Published reference: [`published/Figure_4_full.png`](published/Figure_4_full.png) — rendered from Figure 4 of the paper
([bioRxiv preprint](https://doi.org/10.1101/2025.09.19.677475); the manuscript PDFs themselves are not redistributed in this repo).

- **Run:** [`notebooks/fig04_promoter_splicing_motifs.ipynb`](../../notebooks/fig04_promoter_splicing_motifs.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig04.csv`](reproduced/verify_fig04.csv) (all PASS).
