# 04_analysis/others — phylogeny & architecture schematic

Two standalone supplementary figures: a phylogenetic species tree of the
training-corpus fungi built from NCBI taxonomy, and a schematic of the
Shorkie-LM / supervised model architecture. Numbered-step prefixes encode
execution order; each compute `*.py` keeps its paired `*.sh` runner where one
exists. Paths are read from `config/paths.yaml` via `shorkie.config` (e.g.
`work_root`) and `scripts/common/env.sh` (e.g. `WORK_ROOT`,
`BASKERVILLE_SCRIPTS`).

## `phylogenetic_tree/`

Builds and annotates the species tree for the fungal training corpus.

| Step | File | Purpose |
|------|------|---------|
| 1 | `1_get_taxo_id.py` | Read NCBI Taxon IDs from the cleaned corpus species CSV (`species_fungi_1385_gtf.cleaned.csv`). |
| 2 | `2_plot_tree.sh` | Query NCBI via `ete4 ncbiquery --tree` to emit `species_tree.nwk`. |
| 3 | `3_fix_tree.py` | Parse/clean the Newick tree, shorten leaf labels, write `new_tree.nwk` and render `tree.png` (`3_fix_tree_test.py` is a test variant). |
| 4 | `4_generate_annotation.py` | Filter Saccharomycetales species and write an iTOL `DATASET_COLORSTRIP` annotation file. |
| 5 | `5_generate_collapse_annotation.py` | Emit an iTOL `COLLAPSE` annotation collapsing clades with no highlighted species. |

External tools: `ete4` (NCBI taxonomy query + tree handling); annotation
outputs are for [iTOL](https://itol.embl.de/).

## `viz_shorkie_lm_arch/`

| File | Purpose |
|------|---------|
| `viz_lm.py` | Build the `SeqNN` model from `params.json` + `model_best.h5` and render a `model.png` layer diagram via Keras `plot_model`. |
| `viz_lm.sh` | SLURM wrapper invoking `hound_model_viz.py` (`${BASKERVILLE_SCRIPTS}`) over a trained fold checkpoint under `${WORK_ROOT}`. |

External tools: `baskerville` / `hound_model_viz.py` (Calico
baskerville-yeast), TensorFlow/Keras (`plot_model`, needs Graphviz/`pydot`).
