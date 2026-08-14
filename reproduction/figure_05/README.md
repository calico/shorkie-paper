# Figure 5 — Time-course stress-responsive TF induction (MSN2 & MSN4)

> *"Time-course analysis of stress-responsive transcription factor induction."*

Reproduces main-text **Figure 5** of the Shorkie paper. Published reference: [`published/Figure_5_full.png`](published/Figure_5_full.png) — rendered from Figure 5 of the paper
([bioRxiv preprint](https://doi.org/10.1101/2025.09.19.677475); the manuscript PDFs themselves are not redistributed in this repo).

- **Run:** [`notebooks/fig05_timecourse_tf_induction.ipynb`](../../notebooks/fig05_timecourse_tf_induction.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig05.csv`](reproduced/verify_fig05.csv) (all PASS).
