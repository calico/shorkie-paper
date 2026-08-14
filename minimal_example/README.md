# Shorkie — Minimal Variant Effect Prediction Example

Score a SNP with Shorkie using **logSED** (log₂ Sequence Effect Difference).  
`--model_dir` is the only required flag: `params.json` and `sheet.txt` ship next to this script,
and the genome FASTA/GTF resolve through `shorkie.config` (`genome.fasta` / `genome.gtf`).

---

## Quick Start

Fetch the 8-fold weights and the reference genome, then point `--model_dir` at the weights:

```bash
bash ../data/download.sh --minimal            # -> ./my_shorkie/train/f{0..7}c0/train/model_best.h5
bash ../data/download.sh --genome -u PROJECT  # -> reference FASTA (+.fai) and GTF
python run_shorkie_variant.py --model_dir ./my_shorkie
```

`--genome` is on the requester-pays data bucket, so pass `-u <your-gcp-project>`. If you keep the
genome somewhere else, point at it directly instead:

```bash
python run_shorkie_variant.py --model_dir ./my_shorkie \
  --fasta_file /path/to/GCA_000146045_2.cleaned.fasta \
  --gtf_file   /path/to/GCA_000146045_2.59.gtf
```

On the training cluster you can instead use the config-resolved model path:
```bash
python run_shorkie_variant.py \
  --model_dir "$(python -c 'from shorkie import config; print(config.path("models.shorkie_finetuned"))')"
```

This runs on a built-in example variant (`chrI:124373 T>C`, gene `YAL016C-B`). For a notebook walkthrough
(load → predict → variant effect), see [`../examples/`](../examples).

### Supply your own variant

```bash
python run_shorkie_variant.py \
  --model_dir /path/to/self_supervised_unet_small_bert_drop \
  --chrom chrI --pos 124373 --ref T --alt C --gene YAL016C-B
```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | *(required)* | Root dir of trained model (contains `train/f0c0/…/model_best.h5`, etc.) |
| `--num_folds` | `8` | Number of fold models to ensemble |
| `--chrom` | `chrI` | Chromosome, FASTA naming (`chrI`…`chrXVI`) |
| `--pos` | `124373` | 1-based SNP position |
| `--ref` | `T` | Reference allele |
| `--alt` | `C` | Alternate allele |
| `--gene` | `YAL016C-B` | Gene name (must exist in GTF) |
| `--params_file` | `minimal_example/params.json` | `params.json` for model architecture |
| `--targets_file` | `minimal_example/sheet.txt` | Tab-separated track metadata |
| `--gtf_file` | config `genome.gtf` | Yeast genome annotation (GTF; `I`…`XVI` naming) |
| `--fasta_file` | config `genome.fasta` | Indexed yeast genome FASTA (`chrI`…`chrXVI` naming) |

> **Chromosome naming.** The released FASTA uses `chrI`…`chrXVI` while the GTF uses `I`…`XVI`;
> the script handles each correctly. Fetch both with `data/download.sh --genome` — a genome
> downloaded straight from Ensembl will not match these names.

---

## Model directory structure

```
self_supervised_unet_small_bert_drop/
├── params.json
└── train/
    ├── f0c0/train/model_best.h5
    ├── f1c0/train/model_best.h5
    ⋮
    └── f7c0/train/model_best.h5
```

`bash data/download.sh --minimal` fetches the released 8-fold model into exactly this
layout (or download each fold from `gs://seqnn-share/shorkie_models/shorkie/` — see *Model Availability*
in the top-level [`README`](../README.md)).

---

## What is logSED?

```
logSED = log2(Σ_alt_bins + 1) − log2(Σ_ref_bins + 1)
```

Sums predicted read coverage over gene-body output bins, comparing the reference
and alternate allele sequences. Positive = alt increases predicted expression;
negative = decreases.

---

## Input encoding (for reference)

Shorkie takes `(16384, 170)` inputs:  
- Channels 0–3: DNA one-hot (A/C/G/T)  
- Channels 4–169: species identity; column 114 = 1 for *S. cerevisiae*

---

## Files

| File | Description |
|------|-------------|
| `run_shorkie_variant.py` | Main script (~160 lines) |
| `params.json` | Model architecture config (5215-track Shorkie) |
| `sheet.txt` | Track metadata (5215 ChIP/RNA-seq tracks) |
| `run_example.sh` | SLURM wrapper showing a fully-explicit invocation |
| `README.md` | This file |
