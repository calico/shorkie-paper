# Figure 1 — Datasets, preprocessing, architecture & performance of the fungal LM

> *"Overview of datasets, preprocessing pipeline, model architecture, and performance metrics for the fungal language model (Shorkie LM)."*

Reproduction package for **main-text Figure 1**. Published reference: [`../../paper/Figures/Figure_1.pdf`](../../paper/Figures/Figure_1.pdf) (rendered to `published/Figure_1_full.png`).

- **Reproduce:** [`reproduce_figure_01.ipynb`](reproduce_figure_01.ipynb) (panels A, E, F, G + orchestrates the heavy panels) + [`panels/`](panels/) CLI scripts (B phylogeny, C MUMmer, D Mash).
- **Verify:** the [Verification](#phase-3--verification) section below + `reproduced/verify_fig01.csv`.

---

## Phase 1 — Discovery

Figure 1 has **7 panels (A–G)** spanning every difficulty tier. The scientific thesis: among four evolutionary scopes, the **165_Saccharomycetales (order-level) corpus is the "sweet spot"** that minimizes held-out *S. cerevisiae* perplexity — confirmed quantitatively by panels F (validation loss) and G (test perplexity).

| Panel | Claim | Type | Generating script | Input + config key | GPU |
|---|---|---|---|---|---|
| **A** | Shorkie-LM architecture: multi-resolution U-Net (1→128 bp) + 8 transformer blocks, masked-token (MLM) head | schematic | `scripts/04_analysis/others/viz_shorkie_lm_arch/viz_lm.py` (TF `plot_model` fragment) | released LM `params.json` → `models.shorkie_lm` | no |
| **B** | Phylogeny of the 4 datasets (R64 species / 80_strains / 165_Saccharomycetales order / 1341_Fungal kingdom), Saccharomycetales highlighted green | heavy-external | `scripts/04_analysis/others/phylogenetic_tree/{1_get_taxo_id.py → 3_fix_tree.py → 4_generate_annotation.py → 5_generate_collapse_annotation.py}` + `2_plot_tree.sh` | `data/species_lists/species_*_gtf.cleaned.csv` (Taxon ID col) + NCBI taxonomy via `ete4 ncbiquery --tree` → `species_tree.nwk`; final circular tree styled in **iTOL (web)** | no |
| **C** | MUMmer dot plots, R64 (x) vs one representative per dataset (y) — spectrum of genomic preservation | heavy-external | `scripts/03_eval/lm/genome_evaluation/3_genome_dist/{1_nucmer_aln_genome.sh, 2_show_coords.sh, 3_mummerplot.sh}` | genome FASTAs under `datasets.lm_corpus_split_root/data_{tier}_gtf/fasta/*.cleaned.fasta`; tools `nucmer`/`mummerplot`/`gnuplot` | no |
| **D** | Mash distance, R64 vs all 165_Saccharomycetales (top) and all 80_Strains (bottom), sorted ascending | heavy-external | `scripts/03_eval/lm/genome_evaluation/3_genome_dist/mash/{1_mash_genome.sh, 2_mash_genome_viz.py}` | same FASTAs; tool `mash dist` (parses field `[-3]` = distance) | no |
| **E** | Data-preprocessing pipeline: genome chunking (16384 bp, 4096 overlap) → one-hot tensors + region loss weights → homolog removal + 7% repeat-fraction threshold → train/valid/test chr split | schematic | transforms in `scripts/01_data_build/lm_corpus/` (`1_generate_sequences_bed.py` + `bed_helper.py`, `write_data*.py`) | one example 16,384 bp window | no |
| **F** | Validation-loss curves over training batches for the 4 corpora; legend = min valid loss | computational | `scripts/03_eval/lm/lm_model_eval/3_dataset_comparison.py` | `lm_experiment_root/test_…/{lm_r64_gtf/lm_r64_gtf_unet_small, lm_strains_gtf/lm_strains_gtf_unet_small, LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small, LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small}/train/train.out` | no |
| **G** | Test perplexity, gene vs intergenic, across the 4 corpora | computational | `scripts/03_eval/lm/lm_model_eval/2_model_arch_comparison_test_eval_loss_perplexity.py` | `…/test_testset_perplexity_region/test_testset_perplexity_region.out` → `results.lm_eval_logs` | no |

### Datasets / model variants (panel B labels)
| Tier | Label | Level | #species | corpus dir |
|---|---|---|---|---|
| r64 | R64_yeast | species | 1 | `data_r64_gtf` |
| strains | 80_Strains | strain | 80 | `data_strains_gtf` |
| saccharomycetales | 165_Saccharomycetales | order | 165 | `data_saccharomycetales_gtf` |
| fungi_1385 | 1341_Fungal / 1342_Fungus | kingdom | 1361 | `data_fungi_1385_gtf` |

### Panel C representative target genomes (from the published figure)
| Dot plot | Species / strain | Accession | Tier dir | On disk |
|---|---|---|---|---|
| species (self) | *S. cerevisiae* R64-1.1 | GCA_000146045.2 | data_r64_gtf | ✅ |
| strain | *S. cerevisiae* **YJM195** | GCA_000975585.2 | data_strains_gtf | ✅ exact (deep-recheck fix) |
| order | *Nakaseomyces glabratus* CBS138 | GCA_000002545.2 | data_saccharomycetales_gtf | ✅ |
| order | *Candida albicans* SC5314 | GCA_000182965.3 | data_saccharomycetales_gtf | ✅ |
| kingdom | *Neurospora crassa* OR74A | GCA_000182925.2 | data_fungi_1385_gtf | ✅ |
| kingdom | *Schizosaccharomyces pombe* 972h | GCA_000002945.2 | data_fungi_1385_gtf | ✅ |

### Conda env
All panels run in **`yeast_ml`**. Extra CLI tools: `mummer` (C), `mash` (D), `ete3`/`ete4` (B) — install via bioconda (`mummer4`, `mash`); the project env already has `ete3`. `pdftoppm` (panel extraction) + `PIL` for verification.

### Data availability (verified on disk)
Every input is present under `WORK = /scratch4/ssalzbe1/khchao/Yeast_ML` (resolved via `shorkie.config`):
- genomes: R64 (1), 80_strains (80), 165_Sacc (165), 1341_Fungal (1361) `*.cleaned.fasta` under `data_{tier}_gtf/fasta/`.
- 1F: `train.out` for all 4 tiers (`unet_small`), 117 / 170 / 5001 / 3991 Epoch lines.
- 1G: `test_testset_perplexity_region.out` for all 4 tiers.
- species lists committed at `data/species_lists/`.

### Numeric anchors (extracted from the on-disk `.out`; for Phase-3 verification)
**1G — perplexity (gene / intergenic):** R64 3.7561 / 3.7386 · 80_Strains 3.7342 / 3.7225 · 165_Sacc **3.5488** / 3.6360 · 1342_Fungus 3.6043 / 3.6851. Overall PPL: 3.7458 / 3.7257 / **3.5853** / 3.6380. CE loss: 1.3206 / 1.3153 / **1.2768** / 1.2914. → 165_Sacc lowest; beats 1341_Fungal (paper's claim ✓).
**1F — min validation loss (from published legend):** R64 0.4181 · 80_Strains 0.4154 · 165_Sacc **0.4018** · 1341 0.4055.

### Changes to legacy scripts (tracked)
- **1F parser bug:** `3_dataset_comparison.py::parse_epoch_line` does `line.split(":",1)` then takes the first two floats. For the actual log line
  `Epoch 0 - 455s - train_loss: 0.4322 - steps: 150 - valid_loss: 0.4401 - steps: 64 - best!`
  that yields `(train=0.4322, valid=150)` — **wrong** (grabs the `steps` count). The reproduction notebook uses explicit regex `train_loss:\s*([\d.]+)` / `valid_loss:\s*([\d.]+)` instead. Documented, not edited in place.
- **1C/1D path root:** the shell scripts read `corpus_build_data_root` (`/scratch4/khc/yeast_ssm/data`, a legacy build root absent on this machine). The notebook/panel scripts point instead at the on-disk genomes under `datasets.lm_corpus_split_root`. Documented as a path override.
- **1C/1D output `data_{tier}` vs `data_{tier}_gtf`:** the legacy scripts use `data_{tier}`; the on-disk dirs are `data_{tier}_gtf`. Handled by the override.

---

## Phase 2 — Reproduction (status)

All 7 panels reproduced from on-disk data via `reproduce_figure_01.ipynb` (executed, 0 error cells, 8 embedded figures) + `panels/{run_mummer,run_mash,build_tree}.sh`. Reproduced panels in `reproduced/`:
`Figure_1A_reproduced.png` (arch block-stack), `panelB_tree/Figure_1B_reproduced.png` (circular tree), `panelC_mummer/Figure_1C_reproduced.png` (6 dot plots), `Figure_1D_reproduced.png` (mash bars), `Figure_1E_reproduced.png` (one-hot + loss-weight worked example), `Figure_1F_reproduced.png` (val-loss curves), `Figure_1G_reproduced.png` (gene/intergenic perplexity).

Tooling installed for the full recompute: `mash` v2.3 (fresh `mash_env`, symlinked into `yeast_ml`); MUMmer (`nucmer`/`show-coords`, conda base); `ete4 ncbiquery` (NCBI taxonomy → newick).

## Phase 3 — Verification

**Numeric — `reproduced/verify_fig01.csv`: 12/12 PASS (all Δ = 0.0).** The computational panels reproduce the manuscript values exactly:
- **1F** min validation loss: R64 0.4181 · 80_Strains 0.4154 · 165_Sacc **0.4018** · 1342 0.4055 — identical to the published legend.
- **1G** gene / intergenic perplexity: R64 3.7561/3.7386 · 80_Strains 3.7342/3.7225 · 165_Sacc **3.5488**/3.6360 · 1342 3.6043/3.6851 — identical to the on-disk `.out` anchors. 165_Saccharomycetales lowest → the paper's "sweet spot" claim holds.

**Visual — reproduced vs published** (`published/Figure_1_full.png`, rendered from `paper/Figures/Figure_1.pdf`):
- **1C** dot plots reproduce the homology spectrum exactly — aligned-fraction vs R64: species ≈1.19 (self) → strain ≈1.18 (**YJM195**, near-continuous synteny) → *N. glabratus* 0.017 → *C. albicans* 0.003 → *N. crassa* 0.0003 → *S. pombe* 0.001 (clean diagonal degrading to sparse points, matching the panel). Coords re-run from the FASTAs are **byte-identical**.
- **1D** mash: 80_Strains distances 0–0.008 (near-identical), 165_Saccharomycetales 0→1 — matching the published sorted-bar distributions.
- **1B** circular tree highlights the Saccharomycetales clade (green) with R64 starred — same nested-dataset structure as the published (iTOL-styled) panel.
- **1A / 1E** reproduced programmatically (architecture block-stack from `params.json`; one-hot + region-loss-weight worked example) — conceptual/structural match to the hand-drawn schematics.

### Discrepancy log
| Panel | Discrepancy | Cause | Impact |
|---|---|---|---|
| 1A, 1E | reproduced as programmatic schematic, not the publication's illustrator art | panels are hand-drawn diagrams | conceptual match only (expected for schematics) |
| 1B | tree has 987 leaves vs 1361 input taxids; Saccharomycetales = 111 tips highlighted by dataset taxid (not by NCBI lineage string) | NCBI merged/retired some taxids; NCBI's 2023 taxonomy split "Saccharomycetales" into several orders so the lineage string undercounts; final figure additionally iTOL-styled | topology + highlighted clade faithful (newick byte-identical on re-run); cosmetic styling differs |
| 1C | **RESOLVED** — strain panel now uses *S. cerevisiae* **YJM195** (GCA_000975585.2), the genome the published figure actually shows ("TGT: … YJM195"; manuscript "YJM195 at Mash ≈ 0.01"). Earlier notes wrongly said the published strain was "YPF136" and the repro used YJM1078 — both incorrect | YJM195 **is** in the 80_Strains corpus; fixed `run_mummer.sh` + notebook | now an **exact representative match** (coords byte-identical on re-run); see `recheck/DISCREPANCIES.md` |
| 1F | x-axis = `n_epochs × 64` → 165_Sacc reaches ~320k (= published ~300k); a prior "stops ~250k" note was a crop mis-read. Parser fix vs legacy `3_dataset_comparison.py` retained | legacy `split(':',1)` mis-reads the `train.out` format (grabs `steps` as valid_loss) | corrected `valid_loss:` regex → exact match (Δ=0); panel restyled to published (dashed) |
| 1G | **provenance closed** — exact perplexities appear only in the figure, not the manuscript text | published bar heights pixel-fit to the reproduced values: R²=0.999998, max resid 0.0002 ppl | reproduced numbers **are** the published bars; see `recheck/DISCREPANCIES.md` |
| 1C/1D | input genomes read from `datasets.lm_corpus_split_root` instead of the legacy `corpus_build_data_root` | legacy build root absent on this machine | none (same FASTAs) |

**Verdict: Figure 1 fully reproduced.** Computational panels (F, G) numerically exact (12/12, Δ=0); heavy-external panels (B, C, D) recomputed from genomes (byte-identical on re-run) with the published trends reproduced; schematics (A, E) reproduced programmatically.

### Deep recheck (`recheck/`)
A second, stricter pass verified every result against the **published** figure and corrected two stale errors. Highlights — full detail in [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md):
- **12/12 numbers independently recomputed** from the raw `train.out` / perplexity `.out` (`recompute_fig01.py`), Δ=0; 1F tied to the published legend.
- **1G provenance closed:** the published bar heights pixel-fit the reproduced perplexities (R²=0.999998, max resid 0.0002 ppl; `measure_panelG.py` → `panelG_barheight_fit.csv`).
- **1C strain rep corrected** YJM1078/"YPF136" → **YJM195** (GCA_000975585.2; the genome the published panel shows) → exact match; stale YJM1078 coords removed.
- **Determinism:** mash tables + all MUMmer coords + the ete4 newick re-run **byte-identical** (`diff_heavy_panels.py` → `determinism_fig01.csv`).
- **1F x-axis** confirmed (`n_epochs×64` → 165_Sacc ~320k = published ~300k).
- **Exact-render pass (D/F/G):** panels D, F, G re-rendered to match the published crops in style **and**
  panel proportions (`restyle_panels_DFG.py`; pure render of byte-identical data, numbers unchanged 12/12).
  1D now reproduces the published portrait/broken-axis/brace styling (a manual Illustrator enhancement of
  the plain released bar chart); 1F/1G widened to the published wide aspect. A/B/C/E left as-is.
- Per-panel `recheck/panel_{B,C,D,F,G}_sidebyside.png` + `recheck/Figure_1_published_vs_reproduced.png` give clean [published | reproduced] composites.
