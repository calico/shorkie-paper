# Figure 4 — Promoter & splicing motifs learned during pretraining

> *"Shorkie uses promoter and splicing motifs learned during pretraining."*

Reproduces main-text **Figure 4** of the Shorkie paper. Published reference:
[`../../paper/Figures/Figure_4.pdf`](../../paper/Figures/Figure_4.pdf) (rendered to `published/Figure_4_full.png`).

- **Run:** [`notebooks/fig04_promoter_splicing_motifs.ipynb`](../../notebooks/fig04_promoter_splicing_motifs.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig04.csv`](reproduced/verify_fig04.csv) (all PASS).
