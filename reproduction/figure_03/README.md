# Figure 3 — Shorkie architecture and RNA-seq prediction performance

> *"Shorkie architecture and RNA-seq prediction performance across multiple scales."*

Reproduces main-text **Figure 3** of the Shorkie paper. Published reference:
[`../../paper/Figures/Figure_3.pdf`](../../paper/Figures/Figure_3.pdf) (rendered to `published/Figure_3_full.png`).

- **Run:** [`notebooks/fig03_supervised_rnaseq_prediction.ipynb`](../../notebooks/fig03_supervised_rnaseq_prediction.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig03.csv`](reproduced/verify_fig03.csv) (all PASS).
