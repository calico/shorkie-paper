# 04_analysis/others — phylogeny & architecture schematic

A phylogenetic species tree of the training-corpus fungi built from NCBI
taxonomy (Figure 1B), plus a note on where the Shorkie-LM architecture schematic
(Figure 1A) is now reproduced. Numbered-step prefixes encode execution order.
Paths are read from `config/paths.yaml` via `shorkie.config` (e.g. `work_root`).

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

## Architecture schematic (Figure 1A)

There is no script here for the Shorkie-LM architecture schematic, and it is **not
reproduced**: Figure 1A is a hand-drawn schematic in the paper. The architecture itself is
fully specified by the released LM `params.json` (`models.shorkie_lm`): `unet_small_bert_drop`
= a multi-resolution U-Net trunk + transformer blocks + masked-token (MLM) head. The earlier
`viz_shorkie_lm_arch/{viz_lm.py,viz_lm.sh}` fragments (a non-functional Keras `plot_model` /
baskerville `hound_model_viz.py` wrapper) were removed.
