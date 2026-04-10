# Shorkie — Minimal Variant Effect Prediction Example

Score a SNP with Shorkie using **logSED** (log₂ Sequence Effect Difference).  
Only `--model_dir` is required; all other arguments have sensible defaults.

---

## Quick Start

```bash
python run_shorkie_variant.py \
  --model_dir /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/\
exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop
```

This runs on a built-in example variant (`chrI:124373 T>C`, gene `YAL016C-B`).

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
| `--chrom` | `chrXI` | Chromosome (Roman numeral notation, e.g. `chrXI`) |
| `--pos` | `128987` | 1-based SNP position |
| `--ref` | `A` | Reference allele |
| `--alt` | `G` | Alternate allele |
| `--gene` | `YKL152C` | Gene name (must exist in GTF) |
| `--params_file` | *(cluster default)* | `params.json` for model architecture |
| `--targets_file` | *(cluster default)* | Tab-separated track metadata |
| `--gtf_file` | *(cluster default)* | Yeast genome annotation (GTF) |
| `--fasta_file` | *(cluster default)* | Indexed yeast genome FASTA |

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

The canonical model on the Salzberg cluster:
```
$ROOT/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/
    self_supervised_unet_small_bert_drop/
```
where `ROOT=/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML`.

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
| `run_shorkie_variant.py` | Main script (~130 lines) |
| `cleaned_sheet.txt` | Track metadata (5215 ChIP/RNA-seq tracks) |
| `README.md` | This file |
