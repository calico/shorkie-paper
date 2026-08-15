# others/

Supporting analyses that do not belong to a single model variant.

| Subdirectory | What |
|--------------|------|
| `phylogenetic_tree/` | Builds the *Saccharomycetales* species tree shown in **Figure 1B**: resolve NCBI taxon IDs → plot → clean labels → generate iTOL annotations. |

`reproduction/figure_01/panels/build_tree.sh` and `notebooks/fig01_fungal_lm_corpus_architecture.ipynb`
both call into this directory, so it is load-bearing for Figure 1 despite the generic name.
