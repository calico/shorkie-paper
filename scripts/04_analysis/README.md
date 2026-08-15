# 04_analysis — downstream analyses

> **Advanced / reproduction-only.** These produce the interpretation and benchmark results behind
> Figures 2 and 4–7.

| Subdirectory | Analyses |
|--------------|----------|
| `shorkie/` | Supervised-model work: ISM motif discovery, and the [`eqtl/`](shorkie/eqtl) and MPRA variant-effect benchmarks (Figures 6–7). |
| `shorkie_lm/` | Language-model interpretation: TF-MoDISco motifs, attention/UMAP embeddings, SMT3 visualisations (Figure 2). |
| `shorkie_scratch/` | The Shorkie_Random_Init ablation counterpart, for side-by-side comparison. |
| `others/` | Supporting utilities — currently the phylogenetic-tree build used by Figure 1B. |

**Naming note:** `shorkie_scratch/` refers to the model released as **Shorkie_Random_Init**
(`models.shorkie_random_init`). The directory name predates the published name; see
[`../02_train/README.md`](../02_train/README.md).

Most stages read `results.*` config keys pointing at work-directory outputs that are not in the public
release. The reproduction-critical subsets (eQTL and MPRA scores) *are* released — fetch with
`data/download.sh --eqtl` / `--mpra` and Figures 6–7 reproduce on CPU.
