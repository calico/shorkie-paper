# notebooks/ — figure-reproduction notebooks

One notebook per major Shorkie-paper figure. Each notebook **imports from the
installed `src/shorkie` package** (helpers, config, model ensemble) — it never
redefines utilities or hardcodes machine paths — resolves every input through
`shorkie.config`, and states at the top which figure it reproduces and which
upstream `scripts/` stage must have run first. All are pinned to the `yeast_ml`
kernel.

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
| `fig01_lm_architecture.ipynb` | Shorkie-LM architecture schematic (`unet_small_bert_drop`) | ✅ CPU | — (reads released params) | `models.shorkie_lm` |
| `fig02_lm_genome_eval.ipynb` | LM perplexity / loss across corpus tiers & architectures | ⬚ | `scripts/03_eval/lm/lm_model_eval/` | `results.lm_eval_logs` |
| `fig03_lm_motifs.ipynb` | LM TF-MoDISco motif logos | ⬚ | `scripts/04_analysis/shorkie_lm/motif_analysis/motif_lm/` (`1_search_motif`→`2_modisco_script`) | `results.modisco_lm`, `motif_db_dir` |
| `fig04_cross_species_motifs.ipynb` | Cross-species motif generalization (5 held-out tiers) | ⬚ | `…/motif_analysis/motif_lm__unseen_species/` (per tier) | `results.modisco_unseen` |
| `fig05_promoter_umap.ipynb` | Promoter / feature embedding clustering of LM representations | ⬚ | `…/umap_cluster_promoter/1_predict_seqs_LM.py` | `results.umap`, `genome.gtf` |
| `fig06_attention_map.ipynb` | LM self-attention over a gene locus | ✅ GPU | — (recomputes from released LM weights) | `models.shorkie_lm`, `genome.fasta`, `genome.gtf` |
| `fig07_smt3_dependency.ipynb` | SMT3 case study: PWM logo + nucleotide dependency maps | ⬚ | `…/motif_lm/` (PWM); dependency maps from the 3rd-party `…/lm_SMT3_viz/dependency_map/` notebook | `results.modisco_lm`, `models.shorkie_lm` |
| `fig08_track_prediction.ipynb` | Supervised track prediction: predicted vs observed coverage | ✅ GPU | — (released ensemble + bigwig) | `models.shorkie_finetuned`, `datasets.bigwigs`, `datasets.targets_sheet`, `genome.{fasta,gtf}` |
| `fig09_track_eval_metrics.ipynb` | Supervised eval: train/valid curves + bin/gene-level metrics | ⬚ | `scripts/02_train/` + `scripts/03_eval/supervised/track_prediction_eval/` | `results.train_logs` |
| `fig10_variant_effect_logSED.ipynb` | Variant-effect logSED saliency track | ✅ GPU | — (released ensemble + R64 genome) | `models.shorkie_finetuned`, `genome.{fasta,gtf}`, `datasets.targets_sheet` |
| `fig11_eqtl_benchmark.ipynb` | cis-eQTL variant-effect benchmark (ROC/PR, AUC by distance) | ⬚ | `scripts/04_analysis/shorkie/eqtl/` (scoring → `3_visualization/0_parse_eqtl_res.py`) | `results.eqtl_scores`, `results.mpra_eval` |
| `fig12_mpra_benchmark.ipynb` | MPRA DREAM-Challenge benchmark (Shorkie logSED vs. observed) | ⬚ | `scripts/04_analysis/shorkie/mpra/` (`2_hound_mpra_run`→`3_process_hdf5_logsed`) | `results.mpra_viz`, `datasets.mpra` |
| `fig13_ism_motifs.ipynb` | Shorkie ISM saliency + motif logos (RP/TSS genes) | ⬚ | `…/ism_motif/motif_shorkie__RP_TSS/ism_run/` + `2_modisco_analysis/` | `results.ism_scores`, `results.modisco_ism`, `genome.fasta`, `datasets.targets_sheet` |
| `fig14_ablations.ipynb` | Ablations: LM-pretraining impact (scratch vs. finetuned; arch/lr search) | ⬚ | `scripts/02_train/{shorkie_finetuned,shorkie_scratch}/` + `scripts/04_analysis/shorkie_scratch/` | `results.train_logs` |

## Conventions (for adding notebooks)

- Name `figNN_<topic>.ipynb`; one figure / panel group per notebook.
- Import helpers from `shorkie` (`shorkie.config`, `shorkie.models.ensemble`,
  `shorkie.helpers.yeast_helpers`, `shorkie.viz.load_cov`) — do **not** copy code
  or hardcode paths.
- Top markdown cell: **Reproduces / Upstream / Requires / Source script** lines.
- Cite the `scripts/…` file the plotting logic was ported from; these notebooks
  add a narrative layer and do not replace the pipeline scripts.
- Some gated notebooks additionally need `logomaker` (`pip install logomaker`).
