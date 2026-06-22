# Genome-distance pipeline (Figure 1C / 1D)

Quantifies how far each genome in the LM corpus sits from the *S. cerevisiae* R64
reference. Two complementary views, both shown in **Figure 1**:

- **1C — nucmer dot plots** (`1_nucmer_aln_genome.sh` → `2_show_coords.sh` → `3_mummerplot.sh`):
  whole-genome alignments of R64 (x-axis) against one representative genome per tier
  (y-axis). Near-continuous synteny for close strains (e.g. YJM195), fragmentation for
  more distant taxa, negligible alignment for outgroups.
- **1D — mash distance** (`mash/`) and, as an alternative metric, **dashing2 similarity**
  (`dashing2/`): a sorted distance/similarity of R64 vs *every* genome in a tier.

All paths resolve through `config/paths.yaml` (via `shorkie.config`) — there are **no
hardcoded filesystem fragments**. The shared resolver lives in **`_genome_dist_env.sh`**
(sourced by every step):

| What | Config key | Notes |
|---|---|---|
| Input genomes | `datasets.lm_corpus_split_root` | `data_<data_type>/fasta/*.cleaned.fasta`; reference is `data_r64_gtf/.../GCA_000146045_2.cleaned.fasta` |
| Output root | `corpus_build_results_root` | legacy scratch root — override in `config/paths.yaml`, or export `GD_OUTPUT_ROOT`, for your machine |
| dashing2 binary | `tools.dashing2_bin` | dashing2 is not on bioconda; build it and point this key at your copy |

Outputs land in `<results-root>/ensembl_fungi_59/<data_type>/genome_dist/<tool>/`.

`<data_type>` is the corpus tier suffix: **`r64_gtf`**, **`strains_gtf`**,
**`saccharomycetales_gtf`**, or **`fungi_1385_gtf`** (the dir read is `data_<data_type>`).

## Run

```bash
conda activate yeast_ml                       # shorkie importable; config/paths.yaml set

# Figure 1C — dot plots for one tier (e.g. the 80 strains)
bash 1_nucmer_aln_genome.sh strains_gtf       # nucmer  -> .delta
bash 2_show_coords.sh        strains_gtf       # show-coords -lcr -> .txt (human-readable coords)
bash 3_mummerplot.sh         strains_gtf       # mummerplot + gnuplot -> .png dot plots

# Figure 1D — mash distance (and the dashing2 alternative)
bash mash/1_mash_genome.sh     saccharomycetales_gtf
python mash/2_mash_genome_viz.py saccharomycetales_gtf
bash dashing2/1_dashing2_genome.sh strains_gtf
python dashing2/2_dashing2_genome_viz.py strains_gtf
```

Every `*.sh` accepts **`--dry-run`** (prints the resolved paths + commands without
running the tools — useful to confirm config before a heavy run) and honours
`GD_THREADS` (nucmer `-t`, default 8) and `GD_OUTPUT_ROOT` (results root override).

## Tool prerequisites

`nucmer`, `show-coords`, `mummerplot` (MUMmer4) and `gnuplot`, plus `mash`. On this
cluster MUMmer + mash live in the conda **base** env; `_genome_dist_env.sh` appends
`$(conda info --base)/bin` to `PATH` only when a tool is missing (keeping yeast_ml's
python first for `shorkie`). `dashing2` is built separately (`tools.dashing2_bin`).

## Relationship to the figure reproduction

The runnable figure-reproduction notebook renders the **same** nucmer/mash outputs with
matplotlib (a scheduler-/gnuplot-free path) via
`reproduction/figure_01/panels/run_mummer.sh` + `run_mash.sh` and
`reproduction/figure_01/reproduce_figure_01.ipynb`. These `scripts/` are the original
production pipeline (mummerplot/gnuplot dot plots), cleaned to the repo conventions.
