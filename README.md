<p align="center">
    <img
    src="./shorkie_logo.png"
    alt="Shorkie logo"
    style="display:block; margin-inline:auto; width:30%; height:auto;"
    />
</p>

# Shorkie - Predicting dynamic expression patterns in budding yeast with a fungal DNA language model

Shorkie is a semi-supervised sequence-to-expression model for yeast: a masked DNA language model pretrained on hundreds of closely related fungal genomes and fine-tuned on thousands of epigenomic and transcriptomic profiles—including a large set of transcriptional-regulator induction RNA-seq experiments generated for this study—to predict RNA-seq coverage and variant effects.


This repository contains shell scripts, notebooks, and command snippets used to reproduce the analyses in the **Shorkie** paper. These analyses invoke functionality from the <a href="https://github.com/calico/baskerville-yeast" target="_blank"><strong>baskerville-yeast</strong></a>, and <a href="https://github.com/calico/westminster" target="_blank"><strong>westminster</strong></a> repositories. Please visit those repositories for installation and environment setup instructions.

Contact *[drk (at) @calicolabs.com](mailto:drk@calicolabs.com)*, *[jlinder (at) @calicolabs.com](mailto:jlinder@calicolabs.com)*, or *[kuanhao.chao (at) @gmail.com](mailto:kuanhao.chao@gmail.com)* for questions.

---

## Reproducing the paper

### Quickstart

```bash
git clone --recurse-submodules git@github.com:calico/shorkie-paper.git
cd shorkie-paper
conda env create -f environment.yml && conda activate yeast_ml      # env name: yeast_ml
pip install -e external/baskerville-yeast -e external/westminster -e .   # model code + this package
cp config/paths.example.yaml config/paths.yaml                      # then edit `work_root`
bash data/download.sh --minimal                                     # 8 finetuned folds for the minimal example
python -m ipykernel install --user --name yeast_ml                  # for the notebooks
```

`data/download.sh` also takes `--models` (all 9 `.h5` + sidecars), `--lm-corpus <tier>`, and
`--supervised` (verified against [`data/manifest.json`](./data/manifest.json)). All filesystem
paths resolve through `config/paths.yaml` — there are no hardcoded machine paths in the scripts.

### Repository layout

| Path | What |
|---|---|
| [`src/shorkie/`](./src/shorkie) | installable helper package — `config` (paths), `models.ensemble` (8-fold loader + `logSED`), `helpers.yeast_helpers`, `viz.load_cov` |
| [`scripts/`](./scripts) | all pipelines, staged `00_setup → 01_data_build → 02_train → 03_eval → 04_analysis` (+ `common/`) |
| [`notebooks/`](./notebooks) | 13 figure-reproduction notebooks (import from `shorkie`, pinned to the `yeast_ml` kernel) |
| [`config/`](./config) | `paths.example.yaml`, `slurm.example.yaml` templates |
| [`data/`](./data) | committed reference files + `manifest.json` + `download.sh` (large data is on GCS) |
| [`external/`](./external) | pinned `baskerville-yeast` + `westminster` submodules |
| [`minimal_example/`](./minimal_example) | self-contained logSED variant scorer • [`containers/`](./containers) scheduler-free image |

### Reproducibility matrix

| Variant | Data build | Train | Eval | Analysis |
|---|---|---|---|---|
| **Shorkie LM** (masked DNA LM) | [`01_data_build/lm_corpus/`](./scripts/01_data_build/lm_corpus) | [`02_train/shorkie_lm/`](./scripts/02_train/shorkie_lm) | [`03_eval/lm/`](./scripts/03_eval/lm) | [`04_analysis/shorkie_lm/`](./scripts/04_analysis/shorkie_lm) |
| **Shorkie** (fine-tuned) | [`01_data_build/supervised_tracks/`](./scripts/01_data_build/supervised_tracks) | [`02_train/shorkie_finetuned/`](./scripts/02_train/shorkie_finetuned) | [`03_eval/supervised/`](./scripts/03_eval/supervised) | [`04_analysis/shorkie/`](./scripts/04_analysis/shorkie) |
| **Shorkie scratch** (random-init ablation) | *(same supervised set)* | [`02_train/shorkie_scratch/`](./scripts/02_train/shorkie_scratch) | [`03_eval/supervised/`](./scripts/03_eval/supervised) | [`04_analysis/shorkie_scratch/`](./scripts/04_analysis/shorkie_scratch) |

The only difference between *finetuned* and *scratch* is the `--restore` flag + learning rate
(see [`scripts/02_train/README.md`](./scripts/02_train/README.md)).

### Figures → notebooks → upstream stage

Each paper figure has a notebook in [`notebooks/`](./notebooks); ✅ = runs end-to-end from released
data (`data/download.sh`), ⬚ = load-and-plot from a gated intermediate produced by the cited stage.
See [`notebooks/README.md`](./notebooks/README.md) for the full artifact + `config` key index.

| Notebook | Figure | Runs from released data? | Upstream `scripts/` stage |
|---|---|:--:|---|
| `fig02_lm_genome_eval` | LM perplexity / loss | ⬚ | `03_eval/lm/lm_model_eval/` |
| `fig03_lm_motifs` | LM MoDISco motif logos | ⬚ | `04_analysis/shorkie_lm/motif_analysis/motif_lm/` |
| `fig04_cross_species_motifs` | cross-species motifs | ⬚ | `…/motif_lm__unseen_species/` |
| `fig05_promoter_umap` | promoter embedding UMAP | ⬚ | `04_analysis/shorkie_lm/umap_cluster_promoter/` |
| `fig06_attention_map` | LM self-attention | ✅ GPU | *(recompute from LM weights)* |
| `fig07_smt3_dependency` | SMT3 PWM + dependency maps | ⬚ | `04_analysis/shorkie_lm/lm_SMT3_viz/` |
| `fig08_track_prediction` | predicted vs observed coverage | ✅ GPU | `03_eval/supervised/track_prediction_eval/` |
| `fig09_track_eval_metrics` | track-prediction metrics | ⬚ | `02_train/` + `03_eval/supervised/` |
| `fig10_variant_effect_logSED` | variant-effect logSED track | ✅ GPU | `04_analysis/shorkie/eqtl/` (+ `minimal_example/`) |
| `fig11_eqtl_benchmark` | cis-eQTL ROC/PR benchmark | ⬚ | `04_analysis/shorkie/eqtl/` |
| `fig12_mpra_benchmark` | MPRA DREAM benchmark | ⬚ | `04_analysis/shorkie/mpra/` |
| `fig13_ism_motifs` | Shorkie ISM saliency + motifs | ⬚† | `04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/` |
| `fig14_ablations` | LM-pretraining ablations | ⬚ | `04_analysis/shorkie_scratch/` + figs 11–12 |

† `fig13` ships code-validated but not executed (its ISM `scores.h5` intermediate is not retained on disk); regeneration steps are in the notebook.

---

## Model Availability

The model weights can be downloaded as .h5 files from the URLs below. We are releasing both Shorkie LM-DNA language model and Shorkie, fine-tuned with thousands of epigenomic and transcriptomic profiles. 

- [Shorkie LM](https://storage.googleapis.com/seqnn-share/shorkie_lm/train/model_best.h5)
- Shorkie (`gs://seqnn-share/shorkie/`)
    - [f0](https://storage.googleapis.com/seqnn-share/shorkie/f0/model_best.h5) | [f1](https://storage.googleapis.com/seqnn-share/shorkie/f1/model_best.h5) | [f2](https://storage.googleapis.com/seqnn-share/shorkie/f2/model_best.h5) | [f3](https://storage.googleapis.com/seqnn-share/shorkie/f3/model_best.h5) | [f4](https://storage.googleapis.com/seqnn-share/shorkie/f4/model_best.h5) | [f5](https://storage.googleapis.com/seqnn-share/shorkie/f5/model_best.h5) | [f6](https://storage.googleapis.com/seqnn-share/shorkie/f6/model_best.h5) | [f7](https://storage.googleapis.com/seqnn-share/shorkie/f7/model_best.h5)

---

## Training Data Availability

### Shorkie LM

Shorkie LM was pretrained on the **165_Saccharomycetales** corpus.  
To support reproducibility and the Shorkie LM variants introduced in the paper, we also release three companion corpora—**R64**, **80_strains**, and **1341_Fungus**—each with raw genomes and matched TFRecords. These corpora span different phylogenetic distances and were used to train additional DNA language model variants.

- R64: [genomes] `gs://shorkie-paper/data/unsupervised/genome/R64/` | [tfrecord] `gs://shorkie-paper/data/unsupervised/processed/R64/`
- 80_strains: [genomes] `gs://shorkie-paper/data/unsupervised/genome/80_strains/` | [tfrecord] `gs://shorkie-paper/data/unsupervised/processed/80_strains/`
- 165_Saccharomycetales: [genomes] `gs://shorkie-paper/data/unsupervised/genome/165_Saccharomycetales/` | [tfrecord] `gs://shorkie-paper/data/unsupervised/processed/165_Saccharomycetales/`
- 1341_Fungus: [genomes] `gs://shorkie-paper/data/unsupervised/genome/1341_Fungus/` | [tfrecord] `gs://shorkie-paper/data/unsupervised/processed/1341_Fungus/`

- The training script is at [`scripts/02_train/shorkie_lm/`](./scripts/02_train/shorkie_lm).

### Shorkie

Shorkie was fine-tuned from the Shorkie LM using large-scale transcriptomic and epigenomic datasets from S. cerevisiae.

- **Induction Dynamics Gene Expression Atlas (IDEA)**: RNA-seq **induction time-point** samples from the *Induction Dynamics Gene Expression Atlas (IDEA)*. New datasets generated by Calico Life Sciences LLC (related to IDEA 1.0; Hackett, S.R. et al., *Mol Syst Biol*, 2020).  
    - [Coverage tracks (BigWig)] `gs://shorkie-paper/data/supervised/bigwigs/`
    - [Processed TFRecords] `gs://shorkie-paper/data/supervised/processed/`

- **Yeast strain RNA-seq**: RNA-seq datasets across diverse *S. cerevisiae* strains (Caudal, É. et al., *Nat Genet*, 2024).

- **ChIP-exo** & **ChIP-MNase**: (Rossi, M.J. et al., *Nature*, 2021).

- The training script is at [`scripts/02_train/shorkie_finetuned/`](./scripts/02_train/shorkie_finetuned).


---

## Benchmark data availability

This section lists external benchmark datasets used to evaluate **Shorkie**, along with their sources and primary references.

### MPRA (Random Promoter DREAM Challenge)

- **Dataset**: Random Promoter DREAM Challenge MPRA (held-out set; 71,103 promoter sequences spanning eight categories: native promoters, random 80-bp oligos, high-expression, low-expression, “challenging” sequences, SNV perturbations, motif perturbations, and motif tiling).  
- **Primary reference**: Rafi, A. M. *et al.* “A community effort to optimize sequence-based deep learning models of gene regulation.” *Nat Biotechnol* (2024).  
- **Notes**: We evaluated Shorkie by replacing MPRA constructs into genomic context upstream of TSSs (details in the paper).

### *cis*-eQTL benchmarks

We evaluate Shorkie and compare to DREAM models on two independent yeast *cis*-eQTL resources:

1) **Caudal *et al.* pan-transcriptome**  
   - **Data portal**: 1002 Yeast Genomes project  
     - [GWAS summary stats](http://1002genomes.u-strasbg.fr/files/RNAseq)  
     - [gVCF (1011 isolates)](http://1002genomes.u-strasbg.fr/files/)
   - **Primary reference**:  
     - Caudal, É. *et al.* “Pan-transcriptome reveals a large accessory genome contribution to gene expression variation in yeast.” *Nat Genet* 56, 1278–1287 (2024).  
     - Peter, J. *et al.* “Genome evolution across 1,011 *Saccharomyces cerevisiae* isolates.” *Nature* 556, 339–344 (2018).  
   - **Notes**: We benchmarked 1,901 local *cis*-eQTLs from ~1,000 isolates; negative controls were noncoding SNPs matched by allele, TSS distance, and MAF.

2) **Kita *et al.* high-resolution eQTLs**  
   - [**Supplementary table**](https://www.pnas.org/doi/suppl/10.1073/pnas.1717421114/suppl_file/pnas.1717421114.sd01.txt)
   - **Primary reference**: Kita, R. *et al.* “High-resolution mapping of *cis*-regulatory variation in budding yeast.” *PNAS* 114 (2017).  
   - **Notes**: We benchmarked 683 variants, stratified into Promoter, UTR5, UTR3, and ORF categories.


---

## Minimal Example: Variant Effect Prediction with Shorkie

The [`minimal_example/`](./minimal_example/) directory contains a self-contained
script that demonstrates how to load Shorkie and compute a **logSED** (log₂ Sequence 
Effect Difference) score for a single SNP — no fine-tuning required.

### Setup

1. **Download model weights** (8 folds). Easiest: `bash data/download.sh --minimal`
   (fetches into the `my_shorkie/train/f{i}c0/train/model_best.h5` layout below and
   verifies MD5s against [`data/manifest.json`](./data/manifest.json)). Or manually:
   ```bash
   mkdir -p my_shorkie/train
   for i in 0 1 2 3 4 5 6 7; do
     mkdir -p my_shorkie/train/f${i}c0/train
     wget -O my_shorkie/train/f${i}c0/train/model_best.h5 \
       https://storage.googleapis.com/seqnn-share/shorkie/f${i}/model_best.h5
   done
   ```

2. **Provide a yeast genome FASTA + GTF** (e.g. *S. cerevisiae* R64).

### Run

```bash
python minimal_example/run_shorkie_variant.py \
  --model_dir  my_shorkie \
  --params_file  minimal_example/params.json \
  --targets_file minimal_example/sheet.txt \
  --fasta_file   /path/to/genome.fasta \
  --gtf_file     /path/to/genome.gtf \
  --chrom chrI --pos 124373 --ref T --alt C --gene YAL016C-B
```

### Output

```
==================================================
  Variant  : chrI:124373 T>C
  Gene     : YAL016C-B
  logSED   : +0.0557
==================================================
  logSED > 0 → alt increases predicted expression
  logSED < 0 → alt decreases predicted expression
```

See [`minimal_example/README.md`](./minimal_example/README.md) for full documentation.
