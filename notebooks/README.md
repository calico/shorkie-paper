# notebooks/ — figure-reproduction notebooks

**One notebook per main-text Shorkie-paper figure** (`fig01`–`fig07`, matching the
paper's figure numbers). Each notebook **imports from the installed `src/shorkie`
package** (helpers, config, model ensemble) — it never redefines utilities or
hardcodes machine paths — resolves every input through `shorkie.config`, and states
at the top which figure it reproduces and which upstream `scripts/` stage must have
run first. All are pinned to the `yeast_ml` kernel.

Each notebook is the **user-facing entry point** for its figure; it delegates the
panel-by-panel work to that figure's builders under
[`../reproduction/figure_NN/recheck/`](../reproduction). The
[`../reproduction/`](../reproduction) tree is the **deep audit counterpart**: it adds
the published-panel crops, the reproduced-vs-published verification (`verify_figNN.csv`),
and per-figure `DISCREPANCIES.md`. Run the notebook to regenerate a figure; read
`reproduction/figure_NN/` to audit how each number was checked.

## Setup

```bash
conda env create -f ../environment.yml      # creates the `yeast_ml` env
conda activate yeast_ml
pip install -e ..                            # makes `import shorkie` work
cp ../config/paths.example.yaml ../config/paths.yaml   # then edit work_root / results.* for your machine
python -m ipykernel install --user --name yeast_ml     # register the pinned kernel
jupyter lab                                  # open any figNN_*.ipynb
```

> **Env note:** older `yeast_ml` lockfiles shipped a `jsonschema`/`referencing`
> vs `attrs` mismatch that broke `import nbformat` (and therefore the Jupyter
> stack). The refreshed [`../environment.yml`](../environment.yml) pins
> `attrs>=23` + `referencing` + `jsonschema` alongside `jupyterlab`/`ipykernel`/
> `nbconvert`/`nbformat`/`papermill`/`logomaker`/`umap-learn`, so a fresh env
> works out of the box. (On an older env, `pip install -U attrs referencing
> jsonschema` repairs it in place.)

## Released-data vs. gated figures

- **✅ Runs from released data** — needs only artifacts in [`../data/manifest.json`](../data/manifest.json)
  (model weights on `gs://seqnn-share`, genome/bigwigs on `gs://shorkie-paper`)
  fetched via [`../data/download.sh`](../data/download.sh). GPU recommended where noted.
- **⬚ Gated** — the figure consumes a large **intermediate** (ISM/MoDISco `.h5`,
  score TSVs, training logs, embeddings) that is **not** in the released manifest.
  You must run the cited upstream `scripts/` stage first, then point the relevant
  `results.*` key in `config/paths.yaml` at your output. The notebook resolves the
  path, loads the artifact, and renders the figure (load-and-plot).

The `results.*` keys for the gated intermediates are defined and documented in
[`../config/paths.example.yaml`](../config/paths.example.yaml).

## Index

| Notebook | Figure | Source data | Upstream stage (run first) | Inputs (via `config`) |
|---|---|:---:|---|---|
| `fig01_fungal_lm_corpus_architecture.ipynb` | Fig 1 — fungal LM corpus, phylogeny, architecture & LM performance | ⬚ | `01_data_build/lm_corpus/` + `02_train/shorkie_lm/` + `03_eval/lm/lm_model_eval/`; heavy panels (phylogeny/MUMmer/Mash) via `reproduction/figure_01/panels/` | `lm_experiment_root`, `results.lm_eval_logs`, `datasets.lm_corpus_split_root` |
| `fig02_lm_conserved_motifs.ipynb` | Fig 2 — conserved TF motifs (SMT3 logos, motif→TSS, t-SNE) | ⬚ | `04_analysis/shorkie_lm/{lm_SMT3_viz, motif_analysis/motif_lm, umap_cluster_promoter}/` | `results.modisco_lm`, `results.umap`, `genome.gtf` |
| `fig03_supervised_rnaseq_prediction.ipynb` | Fig 3 — architecture + RNA-seq prediction (violin, scatter, coverage) | ⬚ / ✅ GPU | `03_eval/supervised/track_prediction_eval/`; coverage via `reproduction/figure_03/panels/run_coverage.py` (GPU) | `datasets.supervised_root`, `results.train_logs`, `models.shorkie_finetuned`, `datasets.bigwigs` |
| `fig04_promoter_splicing_motifs.ipynb` | Fig 4 — promoter & splicing ISM motifs (panels A–C/E–G; D & H not reproduced) | ⬚ | `04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/` (`ism_run` + `2_modisco_analysis`) | `results.ism_scores`, `results.modisco_ism`, `genome.fasta`, `datasets.targets_sheet` |
| `fig05_timecourse_tf_induction.ipynb` | Fig 5 — time-course MSN2/MSN4 TF induction | ⬚ | `04_analysis/shorkie/ism_motif/motif_shorkie__time_series/`; ATG42 ISM via `reproduction/figure_05/panels/run_atg42_ism.sbatch` (GPU) | `results.ism_scores`, `genome.fasta`, `datasets.targets_sheet` |
| `fig06_mpra_variant_effects.ipynb` | Fig 6 — MPRA promoter variant effects (Shorkie vs DREAM) | ⬚ | `04_analysis/shorkie/mpra/` (`2_hound_mpra_run`→`3_process_hdf5_logsed`→`5_mpra_viz`) | `results.mpra_viz`, `datasets.mpra` |
| `fig07_eqtl_variant_effects.ipynb` | Fig 7 — cis-eQTL variant effects (ROC/PR, AUPRC-by-distance, ISM) | ⬚ / ✅ GPU | `04_analysis/shorkie/eqtl/` (scoring → `3_visualization/`); ISM/coverage via `reproduction/figure_07/panels/` (GPU) | `results.eqtl_scores`, `results.mpra_eval`, `models.shorkie_finetuned`, `datasets.eqtl` |

## Conventions (for adding notebooks)

- Name `figNN_<topic>.ipynb`; **one main-text figure per notebook** (figure number = paper number).
- Import helpers from `shorkie` (`shorkie.config`, `shorkie.models.ensemble`,
  `shorkie.helpers.yeast_helpers`, `shorkie.viz.load_cov`) — do **not** copy code
  or hardcode paths. Resolve the repo root with `shorkie.config.repo_root()` so the
  notebook runs from any working directory.
- Top markdown cell: **Reproduces / Upstream / Requires / Source script** lines.
- Delegate panel work to the figure's `reproduction/figure_NN/recheck/build_*.py`
  builders (the single source of truth); the notebook adds the narrative + display layer.
- Some gated notebooks additionally need `logomaker` (`pip install logomaker`).
