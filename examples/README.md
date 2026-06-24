# examples/ — load, run, and fine-tune the Shorkie models

Hands-on examples for the two released models — **Shorkie_LM** (masked DNA language model) and
**Shorkie** (supervised 8-fold coverage predictor). They reuse the installable package
(`shorkie.models.ensemble`, `shorkie.helpers.yeast_helpers`) and `shorkie.config`, so they run
unchanged on the training cluster and (after `data/download.sh`) on any machine.

## Prerequisites
```bash
# 1. environment + package
conda env create -f environment.yml && conda activate yeast_ml   # or use containers/
pip install -e .                                                 # makes `import shorkie` work
git submodule update --init                                      # external/baskerville-yeast + westminster
pip install -e external/baskerville-yeast -e external/westminster

# 2. weights (public bucket gs://seqnn-share)
data/download.sh --models all          # shorkie_lm + shorkie (8-fold) + shorkie_random_init
#   ^ then either set config/paths.yaml:work_root, or `export SHORKIE_MODELS=<dest>/models`
#     so the notebooks find the checkpoints (default: they resolve via shorkie.config).

# 3. a reference genome FASTA + GTF (config keys genome.fasta / genome.gtf) — S. cerevisiae R64.
```
The notebooks resolve model directories with `shorkie.config` (`models.shorkie_lm`,
`models.shorkie_finetuned`). To point them at downloaded weights instead, set the
`SHORKIE_MODELS` env var to your local `models/` dir. The released `params.json` + 5215-track
targets sheet are committed at `minimal_example/{params.json,sheet.txt}` (examples 3–4 use those).

## The examples
| # | File | Model | Shows | GPU? |
|---|---|---|---|---|
| 1 | `1_lm_load_and_inference.ipynb` | Shorkie_LM | load the LM; masked-token prediction `P(A,C,G,T)` at a position; argmax-vs-genome | CPU ok (GPU faster) |
| 2 | `2_lm_embeddings.ipynb` | Shorkie_LM | extract the 1st self-attention layer (`multihead_attention`) embeddings; PCA | CPU ok |
| 3 | `3_shorkie_load_and_predict.ipynb` | Shorkie (8-fold) | load the ensemble; predict 5215-track coverage `(1,1,896,5215)`; plot a track | CPU ok (slow) / GPU |
| 4 | `4_shorkie_variant_effect.ipynb` | Shorkie (8-fold) | ref/alt **logSED** variant scoring + per-track effects | CPU ok (slow) / GPU |
| 5 | `5_finetune_lm_on_rnaseq.sh` | LM → Shorkie | **fine-tune** Shorkie_LM on RNA-seq/ChIP-exo/MNase tracks (`westminster --restore`) | **GPU/SLURM** (documented, not auto-run) |

Notes:
- **Input encoding:** Shorkie takes `(16384, 170)` tensors — channels 0–3 are DNA one-hot, 4–169 are
  species identity (column 114 = *S. cerevisiae*). `shorkie.models.ensemble.make_input` builds this
  (and masks a position for the LM via `mask_pos`).
- **logSED** = `log2(Σ_alt_bins + 1) − log2(Σ_ref_bins + 1)` over gene-body bins — the variant-effect
  metric used in the eQTL/MPRA benchmarks (`reproduction/figure_07`, `figure_06`).
- The CLI form of example 4 is `minimal_example/run_shorkie_variant.py`.
- Fine-tuning on **your own** RNA-seq tracks: build a targets sheet + 8-fold TFRecords with
  `scripts/01_data_build/supervised_tracks/`, point `datasets.supervised_data` at them, and run
  example 5 (the LM trunk transfers via `--restore`; only the supervised head is new).
