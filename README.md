<p align="center">
    <img
    src="./shorkie_logo.png"
    alt="Shorkie logo"
    style="display:block; margin-inline:auto; width:30%; height:auto;"
    />
</p>

# Shorkie - Predicting dynamic expression patterns in budding yeast with a fungal DNA language model

Shorkie is a semi-supervised sequence-to-expression model for yeast: a masked DNA language model pretrained on hundreds of closely related fungal genomes and fine-tuned on thousands of epigenomic and transcriptomic profiles—including a large set of transcriptional-regulator induction RNA-seq experiments generated for this study—to predict RNA-seq coverage and variant effects.


This repository lets you **run Shorkie on your own yeast sequences and variants** — and reproduce the
analyses in the **Shorkie** paper. The model framework lives in the
<a href="https://github.com/calico/baskerville-yeast" target="_blank"><strong>baskerville-yeast</strong></a>
and <a href="https://github.com/calico/westminster" target="_blank"><strong>westminster</strong></a>
repositories (pinned as submodules under `external/`); this repo adds an installable helper package
(`src/shorkie`), the released model/data catalogue, runnable examples, and the figure notebooks.

Please open a [GitHub issue](https://github.com/calico/shorkie-paper/issues) for bugs or questions. For other inquiries, contact *[drk (at) calicolabs.com](mailto:drk@calicolabs.com)*, *[jlinder (at) calicolabs.com](mailto:jlinder@calicolabs.com)*, or *[kuanhao.chao (at) gmail.com](mailto:kuanhao.chao@gmail.com)*.

---

## Requirements

- **OS:** Linux. **Python:** 3.9. **TensorFlow:** ~2.15 (exact pins in [`environment.yml`](./environment.yml)).
- **CPU** is enough to run inference, score variants, and reproduce the figures from released data.
- **GPU** (CUDA-capable) is needed only for training / fine-tuning and the GPU-marked figure panels — those
  also need `tensorrt==8.6.1` and a CUDA-enabled TensorFlow build. [`containers/`](./containers) ships a
  known-good Docker / Apptainer image for a scheduler-free run.

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/calico/shorkie-paper.git   # (or git@github.com:calico/shorkie-paper.git with SSH access)
cd shorkie-paper
conda env create -f environment.yml && conda activate yeast_ml      # env name: yeast_ml
pip install -e external/baskerville-yeast -e external/westminster -e .   # model code + this package
cp config/paths.example.yaml config/paths.yaml                      # then edit `work_root`
bash data/download.sh --minimal                                     # 8 Shorkie folds for the example below
```

`data/download.sh` takes `--models [lm|finetuned|random_init|all]` — all three model variants (Shorkie LM,
the Shorkie 8-fold ensemble, and the Shorkie_Random_Init 8-fold ablation) are **live** on the public bucket
`gs://seqnn-share` — plus `--lm-corpus <tier>`, `--supervised`, `--eqtl`, `--mpra` (all verified against
[`data/manifest.json`](./data/manifest.json)). Every filesystem path resolves through `config/paths.yaml`
— there are no hardcoded machine paths.

Approximate download sizes: `--minimal` (8 Shorkie folds) ≈ 0.46 GB; `--models all` (LM + both 8-fold
ensembles) ≈ 1.4 GB. The LM corpora (`--lm-corpus`) and `--supervised` bigwigs/TFRecords are large (tens to
hundreds of GB — e.g. the supervised bigwigs are ~93 GB); see the `size_bytes` / `approx_size` fields in
[`data/manifest.json`](./data/manifest.json) for exact figures.

## Using Shorkie on your own data

- **[`examples/`](./examples)** — step-by-step notebooks: load each model, run inference, score variant
  effects for Shorkie / Shorkie_LM, and fine-tune the LM on your own RNA-seq tracks. Start here.
- **[`minimal_example/`](./minimal_example)** — a self-contained CLI that scores one SNP end-to-end
  (see [Minimal Example](#minimal-example-variant-effect-prediction-with-shorkie) below).
- **[`containers/`](./containers)** — Docker / Apptainer image for a scheduler-free run.
- **[`src/shorkie/`](./src/shorkie)** — the importable helper package: `config` (path resolution),
  `models.ensemble` (8-fold loader + `logSED`), `helpers.yeast_helpers`, `viz.load_cov`.

---

## Model Availability

The model weights are downloaded as .h5 files from the URLs below (or with
`data/download.sh --models all`). **Shorkie LM**, **Shorkie** (8-fold), and **Shorkie_Random_Init** (8-fold)
are all live on the public bucket `gs://seqnn-share` and catalogued (with md5s) in
[`data/manifest.json`](./data/manifest.json).

- **(live)** [Shorkie LM](https://storage.googleapis.com/seqnn-share/shorkie_lm/train/model_best.h5)
- **(live)** Shorkie (`gs://seqnn-share/shorkie/`)
    - [f0](https://storage.googleapis.com/seqnn-share/shorkie/f0/model_best.h5) | [f1](https://storage.googleapis.com/seqnn-share/shorkie/f1/model_best.h5) | [f2](https://storage.googleapis.com/seqnn-share/shorkie/f2/model_best.h5) | [f3](https://storage.googleapis.com/seqnn-share/shorkie/f3/model_best.h5) | [f4](https://storage.googleapis.com/seqnn-share/shorkie/f4/model_best.h5) | [f5](https://storage.googleapis.com/seqnn-share/shorkie/f5/model_best.h5) | [f6](https://storage.googleapis.com/seqnn-share/shorkie/f6/model_best.h5) | [f7](https://storage.googleapis.com/seqnn-share/shorkie/f7/model_best.h5)
- **(live)** Shorkie_Random_Init (from-scratch ablation, lr 5e-4, 8-fold; `gs://seqnn-share/shorkie_random_init/`)
    - [f0](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f0/model_best.h5) | [f1](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f1/model_best.h5) | [f2](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f2/model_best.h5) | [f3](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f3/model_best.h5) | [f4](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f4/model_best.h5) | [f5](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f5/model_best.h5) | [f6](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f6/model_best.h5) | [f7](https://storage.googleapis.com/seqnn-share/shorkie_random_init/f7/model_best.h5)

See [`examples/`](./examples) for runnable notebooks on each model — loading, inference, variant-effect
prediction, and fine-tuning the LM on your own RNA-seq tracks.

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

> **Released.** The reproduction-critical subsets — the per-SNP eQTL score TSVs (Caudal/Kita/Renganaath,
> Shorkie / Shorkie_LM / Shorkie_Random_Init) and the MPRA ground-truth expression + cached Shorkie/DREAM
> scores — are catalogued in [`data/manifest.json`](./data/manifest.json) and live on
> `gs://shorkie-paper/{eqtl,mpra}/`. Fetch them with `data/download.sh --eqtl` / `--mpra` (requester-pays;
> pass `-u PROJECT`) and Figures 6–7 reproduce on CPU without re-scoring. The large raw inputs (the
> 1011-genomes GVCF, the full DREAM Challenge sequences, the DREAM-RNN/PrixFixe weights) are third-party
> and obtained from their original sources below (not re-hosted).

### MPRA (Random Promoter DREAM Challenge)

- **Dataset**: Random Promoter DREAM Challenge MPRA (held-out set; 71,103 promoter sequences spanning eight categories: native promoters, random 80-bp oligos, high-expression, low-expression, “challenging” sequences, SNV perturbations, motif perturbations, and motif tiling).  
- **Primary reference**: Rafi, A. M. *et al.* “A community effort to optimize sequence-based deep learning models of gene regulation.” *Nat Biotechnol* (2024).  
- **Notes**: We evaluated Shorkie by replacing MPRA constructs into genomic context upstream of TSSs (details in the paper).

### *cis*-eQTL benchmarks

We evaluate Shorkie and compare to DREAM models on three independent yeast *cis*-eQTL resources:

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

3) **Renganaath *et al.* MPRA-validated *cis*-regulatory variants**
   - [**Article (eLife 2020)**](https://elifesciences.org/articles/62669)
   - **Primary reference**: Renganaath, K., Chong, R., Day, L., Kosuri, S., Kruglyak, L. & Albert, F. W. “Systematic identification of *cis*-regulatory variants that cause gene expression differences in a yeast cross.” *eLife* 9, e62669 (2020).
   - **Notes**: We benchmarked 142 core-promoter variants (Figure 7, panel G).


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

---

## Reproducing the paper figures

Each main-text figure has one notebook in [`notebooks/`](./notebooks) (`fig01`–`fig07`). A notebook
either runs end-to-end from released data (`data/download.sh`) or loads a gated intermediate produced by
the cited `scripts/` stage, then renders the panels by calling that figure's builders under
[`reproduction/figure_NN/`](./reproduction). See [`notebooks/README.md`](./notebooks/README.md) for the
full figure → artifact → `config`-key index, and [`reproduction/`](./reproduction) for the per-figure
panel builders, published crops, and reproduced-vs-published checks (`verify_figNN.csv`).

The end-to-end pipelines live in [`scripts/`](./scripts), staged
`00_setup → 01_data_build → 02_train → 03_eval → 04_analysis`, for all three model variants:

| Variant | Train | Analysis |
|---|---|---|
| **Shorkie LM** (masked DNA LM) | [`02_train/shorkie_lm/`](./scripts/02_train/shorkie_lm) | [`04_analysis/shorkie_lm/`](./scripts/04_analysis/shorkie_lm) |
| **Shorkie** (fine-tuned) | [`02_train/shorkie_finetuned/`](./scripts/02_train/shorkie_finetuned) | [`04_analysis/shorkie/`](./scripts/04_analysis/shorkie) |
| **Shorkie_Random_Init** (random-init ablation, lr 5e-4, 8-fold) | [`02_train/shorkie_scratch/`](./scripts/02_train/shorkie_scratch) | [`04_analysis/shorkie_scratch/`](./scripts/04_analysis/shorkie_scratch) |

The only difference between *finetuned* and *random-init* is the `--restore` flag + learning rate
(see [`scripts/02_train/README.md`](./scripts/02_train/README.md)).

---

## Citation

The manuscript and supplements are in [`paper/`](./paper) — [`shorkie.pdf`](./paper/shorkie.pdf),
[`shorkie_supplemental_figures.pdf`](./paper/shorkie_supplemental_figures.pdf), and
[`shorkie_supplemental_tables.pdf`](./paper/shorkie_supplemental_tables.pdf).

If you use Shorkie, please cite:

> Chao, K.-H., Magzoub, M. M., Stoops, E., Hackett, S. R., Linder, J., & Kelley, D. R. (2025).
> *Predicting dynamic expression patterns in budding yeast with a fungal DNA language model.* bioRxiv.
> <https://doi.org/10.1101/2025.09.19.677475>

```bibtex
@article{chao2025shorkie,
  title   = {Predicting dynamic expression patterns in budding yeast with a fungal DNA language model},
  author  = {Chao, Kuan-Hao and Magzoub, Majed M. and Stoops, E. and Hackett, Sean R. and Linder, Johannes and Kelley, David R.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.09.19.677475},
  url     = {https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1}
}
```

---

## License

Code in this repository is released under the **Apache License 2.0** — see [`LICENSE`](./LICENSE). The
released model weights (`gs://seqnn-share`) and the third-party benchmark datasets carry their own terms;
see the original sources cited above.
