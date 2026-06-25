# Figure 2 — Shorkie LM identifies conserved TF motifs across fungal genomes

> *"Shorkie LM identifies conserved transcription factor binding motifs across fungal genomes."*

Reproduces main-text **Figure 2** of the Shorkie paper. Published reference:
[`../../paper/Figures/Figure_2.pdf`](../../paper/Figures/Figure_2.pdf) (rendered to `published/Figure_2_full.png`).

- **Run:** [`notebooks/fig02_lm_conserved_motifs.ipynb`](../../notebooks/fig02_lm_conserved_motifs.ipynb) (env `yeast_ml`). It delegates the
  panel building to the single-source builders in [`recheck/`](recheck/).
- **Output:** regenerated panels in [`reproduced/`](reproduced/); the published-vs-reproduced numeric
  checks are in [`reproduced/verify_fig02.csv`](reproduced/verify_fig02.csv) (all PASS).
