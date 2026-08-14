# Figure 2 — Shorkie LM identifies conserved TF motifs across fungal genomes

> *"Shorkie LM identifies conserved transcription factor binding motifs across fungal genomes."*

Reproduces main-text **Figure 2** of the Shorkie paper. Published reference: [`published/Figure_2_full.png`](published/Figure_2_full.png) — rendered from Figure 2 of the paper
([bioRxiv preprint](https://doi.org/10.1101/2025.09.19.677475); the manuscript PDFs themselves are not redistributed in this repo).

- **Run:** [`notebooks/fig02_lm_conserved_motifs.ipynb`](../../notebooks/fig02_lm_conserved_motifs.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig02.csv`](reproduced/verify_fig02.csv) (all PASS).
