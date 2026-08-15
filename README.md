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

📖 **Full documentation: [khchao.com/shorkie](https://khchao.com/shorkie/)** — installation, usage guides
for every model, a [figure-by-figure analysis gallery](https://khchao.com/shorkie/content/gallery.html),
a [dataset catalogue](https://khchao.com/shorkie/content/data_resources.html), and
[troubleshooting](https://khchao.com/shorkie/content/troubleshooting.html).

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

`data/download.sh` takes `--minimal`, `--models [lm|finetuned|random_init|lm-variants|all]` — all three
model variants (Shorkie LM, the Shorkie 8-fold ensemble, and the Shorkie_Random_Init 8-fold ablation) are
**live** on the public bucket `gs://seqnn-share` — plus `--genome`, `--lm-corpus <tier>`, `--supervised`,
`--eqtl`, `--mpra` (all verified against [`data/manifest.json`](./data/manifest.json)). Every filesystem
path resolves through `config/paths.yaml` — there are no hardcoded machine paths.

To actually run the models you also need the **R64 reference genome**:

```bash
bash data/download.sh --genome -u <your-gcp-project>   # FASTA + GTF + .fai -> <dest>/genome/R64/
```

> **Use this copy, not a fresh Ensembl download.** The chromosome naming is load-bearing and differs
> between the two files — the FASTA uses `chrI`…`chrXVI`, the GTF uses `I`…`XVI`. A genome pulled
> straight from Ensembl or SGD will not match, and every example will fail to find its sequence.

Approximate download sizes: `--minimal` (8 Shorkie folds) ≈ 0.46 GB; `--models all` (LM + both 8-fold
ensembles) ≈ 0.97 GB; `--genome` ≈ 22 MB. The LM corpora (`--lm-corpus`) and `--supervised`
bigwigs/TFRecords are large (the 165_Saccharomycetales tier is ~3.7 GB, the supervised bigwigs ~93 GB);
see [Data availability](#data-availability) below, or the `size_bytes` / `approx_size` fields in
[`data/manifest.json`](./data/manifest.json), for exact figures.

Then check your install:

```bash
pytest -q                       # release-integrity + smoke tests (fast, offline, no weights needed)
```

## Using Shorkie on your own data

- **[`examples/`](./examples)** — step-by-step notebooks: load each model, run inference, score variant
  effects for Shorkie / Shorkie_LM, and fine-tune the LM on your own RNA-seq tracks. Start here.
- **[`examples/6_finetune_minidemo.sh`](./examples/6_finetune_minidemo.sh)** — before committing GPU-days
  to fine-tuning, run this: the real `--restore` pipeline on a tiny slice of the released data,
  ~96 MB and about a minute on CPU. Its output is deliberately not a usable model — it proves the
  mechanics work.
- **[`minimal_example/`](./minimal_example)** — a self-contained CLI that scores one SNP end-to-end
  (see [Minimal Example](#minimal-example-variant-effect-prediction-with-shorkie) below).
- **[`containers/`](./containers)** — Docker / Apptainer image for a scheduler-free run.
- **[`src/shorkie/`](./src/shorkie)** — the importable helper package: `config` (path resolution),
  `models.ensemble` (8-fold loader + `logSED`), `helpers.yeast_helpers`, `viz.load_cov`.

Stuck? The [troubleshooting guide](https://khchao.com/shorkie/content/troubleshooting.html) is keyed on
the exact error messages people actually hit.

---

## Model Availability

The model weights are downloaded as .h5 files from the URLs below (or with
`data/download.sh --models all`). **Shorkie LM**, **Shorkie** (8-fold), and **Shorkie_Random_Init** (8-fold)
are all live on the public bucket `gs://seqnn-share` and catalogued (with md5s) in
[`data/manifest.json`](./data/manifest.json).

- **(live)** [Shorkie LM](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_lm/train/model_best.h5)
- **(live)** Shorkie (`gs://seqnn-share/shorkie_models/shorkie/`)
    - [f0](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f0/model_best.h5) | [f1](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f1/model_best.h5) | [f2](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f2/model_best.h5) | [f3](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f3/model_best.h5) | [f4](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f4/model_best.h5) | [f5](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f5/model_best.h5) | [f6](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f6/model_best.h5) | [f7](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f7/model_best.h5)
- **(live)** Shorkie_Random_Init (from-scratch ablation, lr 5e-4, 8-fold; `gs://seqnn-share/shorkie_models/shorkie_random_init/`)
    - [f0](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f0/model_best.h5) | [f1](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f1/model_best.h5) | [f2](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f2/model_best.h5) | [f3](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f3/model_best.h5) | [f4](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f4/model_best.h5) | [f5](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f5/model_best.h5) | [f6](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f6/model_best.h5) | [f7](https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie_random_init/f7/model_best.h5)

See [`examples/`](./examples) for runnable notebooks on each model — loading, inference, variant-effect
prediction, and fine-tuning the LM on your own RNA-seq tracks.

### LM checkpoints for the other corpus tiers

Catalogued as `models.lm_variants` (fetch with `data/download.sh --models lm-variants`): the four
`unet_small` runs behind the Figure 1F/G corpus-scaling comparison, plus 1341_Fungus at
`unet_small_bert_drop`.

| Variant | Corpus | Architecture | `num_features` |
|---|---|---|---|
| `R64_yeast__unet_small` | R64 (1 genome) | `unet_small` | 6 |
| `80_strains__unet_small` | 80_strains | `unet_small` | 85 |
| `165_Saccharomycetales__unet_small` | 165_Saccharomycetales | `unet_small` | 170 |
| `1341_Fungus__unet_small` | 1341_Fungus | `unet_small` | 1366 |
| `1341_Fungus__unet_small_bert_drop` | 1341_Fungus | `unet_small_bert_drop` | 1366 |

> **Not drop-in replacements for Shorkie LM.** The released Shorkie LM is 165_Saccharomycetales +
> `unet_small_bert_drop`; the four `unet_small` runs are a *different architecture*, used for the
> corpus-scaling figure, and are meaningful compared against each other rather than against the released
> model. The one genuine alternative is `1341_Fungus__unet_small_bert_drop` — same architecture, largest
> corpus.
>
> **`num_features` is not stored in `params.json`** and must be set at load time; it is
> `4 (DNA) + num_species + 1`. Passing the wrong value raises
> `Error loading weights by name: axes don't match array`.

---

## Data availability

Everything we curated for this study, with how to get it. All of it is catalogued — with sizes, MD5s and
destinations — in [`data/manifest.json`](./data/manifest.json), which is what `data/download.sh` reads.
A browsable version is at
[khchao.com/shorkie → Datasets](https://khchao.com/shorkie/content/data_resources.html).

**Two buckets:** model weights are on the **public** `gs://seqnn-share` (plain HTTPS works, no account).
Datasets are on `gs://shorkie-paper`, which is **requester-pays** — it needs `gsutil` and a
billing-enabled GCP project via `-u PROJECT`; you pay egress only.

| Dataset | Size | Get it with |
|---|---|---|
| **Model weights** (LM + Shorkie 8-fold + Random_Init) | 0.97 GB | `data/download.sh --models all` |
| **R64 reference genome** (FASTA + GTF + `.fai`) | 22 MB | `data/download.sh --genome -u PROJECT` |
| **LM corpus — R64** (1 genome) | 23 MB | `data/download.sh --lm-corpus R64 -u PROJECT` |
| **LM corpus — 80_strains** (80 genomes) | 1.5 GB | `data/download.sh --lm-corpus 80_strains -u PROJECT` |
| **LM corpus — 165_Saccharomycetales** ⭐ (165 genomes) | 3.7 GB | `data/download.sh --lm-corpus 165_Saccharomycetales -u PROJECT` |
| **LM corpus — 1341_Fungus** (1,361 genomes) | 42.5 GB | `data/download.sh --lm-corpus 1341_Fungus -u PROJECT` |
| **Supervised tracks** — BigWigs | ~93 GB | `data/download.sh --supervised bigwigs -u PROJECT` |
| **Supervised tracks** — 8-fold TFRecords | ~10 GB | `data/download.sh --supervised tfrecords -u PROJECT` |
| **cis-eQTL benchmark** (scores + DREAM baselines) | ~64 MB | `data/download.sh --eqtl -u PROJECT` |
| **MPRA benchmark** (ground truth + cached scores) | ~1.6 GB | `data/download.sh --mpra all -u PROJECT` |

⭐ = the corpus Shorkie_LM was actually pretrained on.

> **Cannot use Google Cloud?** Requester-pays needs a billing-enabled GCP project, which is impractical
> in some regions. `scripts/00_setup/zenodo_upload.py` publishes the models and corpora to Zenodo — see
> [`data/README.md`](./data/README.md).

### The pretraining corpora

Four tiers of increasing phylogenetic breadth, each with raw genomes and matched 16,384 bp **ZLIB**
TFRecords. Shorkie LM was pretrained on **165_Saccharomycetales**; the others back the ablations.

| Tier | Genomes | Train seqs | `gs://shorkie-paper/data/unsupervised/…` |
|---|---|---|---|
| R64 | 1 | 1,201 | `{genome,processed}/R64/` |
| 80_strains | 80 | 102,315 | `{genome,processed}/80_strains/` |
| 165_Saccharomycetales ⭐ | 165 | 385,551 | `{genome,processed}/165_Saccharomycetales/` |
| 1341_Fungus | 1,361 | 625,355 | `{genome,processed}/1341_Fungus/` |

All four share one held-out split, drawn from *S. cerevisiae* R64 only and split by **whole chromosome**
— valid: chrXI, chrXIII, chrXV; test: chrXII, chrXIV, chrXVI, with chrXI–XVI excluded from training
everywhere. Keep that split if you want numbers comparable to ours. Per-tier species lists, with NCBI
accessions, are committed in [`data/species_lists/`](./data/species_lists).

> **`1341_Fungus` nests one level deeper.** Its TFRecords live under an extra `1342_Fungus/`
> subdirectory (yes, a different number), unlike the other three tiers — so
> `--lm-corpus 1341_Fungus` yields `…/processed/1341_Fungus/1342_Fungus/*.tfr` and a fixed-depth glob
> will miss them. The public label is also historical: the cleaned corpus holds **1,361** assemblies.

Training script: [`scripts/02_train/shorkie_lm/`](./scripts/02_train/shorkie_lm).

### The supervised tracks

5,215 tracks on *S. cerevisiae* R64 at 16 bp resolution, which Shorkie is fine-tuned on:

- **Induction Dynamics Gene Expression Atlas (IDEA)** — RNA-seq induction time-course samples, generated
  for this study by Calico Life Sciences (related to IDEA 1.0; Hackett, S.R. *et al.*, *Mol Syst Biol*, 2020)
- **Yeast strain RNA-seq** across diverse *S. cerevisiae* isolates (Caudal, É. *et al.*, *Nat Genet*, 2024)
- **ChIP-exo** and **ChIP-MNase** (Rossi, M.J. *et al.*, *Nature*, 2021)

`gs://shorkie-paper/data/supervised/{bigwigs,processed}/`. The targets sheet is also committed at
[`minimal_example/sheet.txt`](./minimal_example/sheet.txt) so you can inspect track metadata without
downloading anything. Training script:
[`scripts/02_train/shorkie_finetuned/`](./scripts/02_train/shorkie_finetuned).

---

## Benchmark data availability

External benchmark datasets used to evaluate **Shorkie**, with sources and primary references.

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
       https://storage.googleapis.com/seqnn-share/shorkie_models/shorkie/f${i}/model_best.h5
   done
   ```

2. **Fetch the R64 reference genome**: `bash data/download.sh --genome -u <your-gcp-project>`
   (FASTA + GTF + `.fai`). See the naming caveat under [Quickstart](#quickstart) — use this copy
   rather than a fresh Ensembl download.

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
  logSED   : +0.0643
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

If you use Shorkie, please cite our preprint:

> Chao, K.-H., Magzoub, M. M., Stoops, E. H., Hackett, S. R., Linder, J., & Kelley, D. R. (2025).
> *Predicting dynamic expression patterns in budding yeast with a fungal DNA language model.* bioRxiv.
> <https://doi.org/10.1101/2025.09.19.677475>

```bibtex
@article{chao2025shorkie,
  title   = {Predicting dynamic expression patterns in budding yeast with a fungal DNA language model},
  author  = {Chao, Kuan-Hao and Magzoub, Majed M. and Stoops, Emily H. and Hackett, Sean R. and Linder, Johannes and Kelley, David R.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.09.19.677475},
  url     = {https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1}
}
```

---

## License

| What | Terms |
|---|---|
| **Code** in this repository | **Apache License 2.0** — see [`LICENSE`](./LICENSE) |
| **Model weights** (`gs://seqnn-share/shorkie_models/`) and the data we derived (TFRecords, cached scores) | **CC BY 4.0** — free to use and redistribute with attribution; please cite the preprint |
| **Genome assemblies** in the LM corpora | Third-party public data (Ensembl Fungi release 59 / NCBI GenBank), redistributed here for reproducibility, under their original providers' terms. Accessions are in [`data/species_lists/`](./data/species_lists) |
| **Benchmark datasets** (DREAM Challenge MPRA, the eQTL studies, 1002 Yeast Genomes) | Their own terms — see the original sources cited above |

Please cite both the Shorkie preprint and the original data sources when you use the released data.
