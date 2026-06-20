# Shorkie manuscript — figure reproduction AUDIT MATRIX

Master panel-level traceability for the **7 main-text figures** (~64 panels). Each panel maps to its scientific claim, generating script (in this repo), input artifact + `shorkie.config` key, conda env, GPU need, any existing topic-notebook that already covers it, and on-disk data availability.

**Roots:** public repo = `shorkie-paper/`; private origin/data = `WORK = /scratch4/ssalzbe1/khchao/Yeast_ML` (resolved via `shorkie.config`). Paths below are relative to one of these.

**Type legend:** `comp` = computational load/recompute (CPU) · `gpu` = needs a model forward pass/ISM · `heavy` = heavy external tool (MUMmer/Mash/ETE3) · `schem` = hand-drawn schematic (regenerate programmatic source where one exists, note manual styling).

**Repro status legend:** ✅ reproduced+verified · 🟡 in progress · ⬚ catalogued (not yet reproduced) · ⚠️ documented gap.

Supplementary figures S1–S29 are out of scope for this pass (see manuscript `paper/shorkie_supplemental_figures.pdf`); several reuse the same pipelines (e.g. S3/S4 genome eval, S5/S13 ablations, S18 attention, S19/S20 motifs).

---

## Figure 1 — Datasets, preprocessing, architecture & performance of the fungal LM
Package: [`figure_01/`](figure_01/)

| Panel | Claim | Type | Generating script (repo) | Input + config key | Env | GPU | Notebook | Data on disk | Status |
|---|---|---|---|---|---|---|---|---|---|
| A | Shorkie-LM architecture schematic (multi-res U-Net + 8 transformer blocks, masked-token) | schem | `scripts/04_analysis/others/viz_shorkie_lm_arch/viz_lm.py` (fragment; notebook draws it) | released `params.json` → `models.shorkie_lm` | yeast_ml | no | `fig01` | present | ✅ |
| B | 4-dataset phylogeny (R64 / 80_strains / 165_Sacc / 1341_Fungal), Saccharomycetales highlighted | heavy | `scripts/04_analysis/others/phylogenetic_tree/{1_get_taxo_id,3_fix_tree,4_generate_annotation,5_generate_collapse_annotation}.py + 2_plot_tree.sh` | `data/species_lists/*.csv` + NCBI taxonomy (ete4 ncbiquery) → newick; iTOL web-styled | ete4/ete3 | no | none | species lists present | ✅ |
| C | MUMmer dot plots: R64 vs R64-1.1, YPF136, *N. glabratus* CBS138 (GCA_000002545.2), *C. albicans* SC5314 (GCA_000182965.3), *N. crassa* OR74A (GCA_000182925.2), *S. pombe* 972h (GCA_000002945.2) | heavy | `scripts/03_eval/lm/genome_evaluation/3_genome_dist/{1_nucmer_aln_genome,2_show_coords,3_mummerplot}.sh` | genome FASTAs under `datasets.lm_corpus_split_root/data_{tier}_gtf/fasta` | yeast_ml + MUMmer | no | none | 5/6 targets present (YPF136 strain → representative) | ✅ |
| D | Mash distance: R64 vs all 165_Sacc (top) + all 80_Strains (bottom), sorted | heavy | `scripts/03_eval/lm/genome_evaluation/3_genome_dist/mash/{1_mash_genome.sh,2_mash_genome_viz.py}` | 80 + 165 FASTAs (same root) | yeast_ml + mash | no | none | 80 + 165 FASTAs present | ✅ |
| E | Data-preprocessing pipeline (chunking 16384/4096, tensors + region loss weights, homolog & 7% repeat removal, chr split) | schem | transforms in `scripts/01_data_build/lm_corpus/` | one example window | yeast_ml | no | none | corpus present | ✅ |
| F | Validation-loss curves over training batches, 4 tiers (legend min: R64 0.4181 / 80_Strains 0.4154 / 165_Sacc 0.4018 / 1341 0.4055) | comp | `scripts/03_eval/lm/lm_model_eval/3_dataset_comparison.py` (**parser bug for this log format — fix valid_loss regex**) | `lm_{tier}.../{unet_small}/train/train.out` → `lm_experiment_root` | yeast_ml | no | partial `fig02` | present (117–5001 Epoch lines/tier) | ✅ |
| G | Test perplexity gene vs intergenic, 4 tiers (165_Sacc lowest 3.585; beats 1341 3.638) | comp | `scripts/03_eval/lm/lm_model_eval/2_model_arch_comparison_test_eval_loss_perplexity.py` | `test_testset_perplexity_region.out` → `results.lm_eval_logs` | yeast_ml | no | `fig02` | present | ✅ |

Fig 1 numeric anchors (verified on disk): perplexity gene/intergenic — R64 3.756/3.739, 80_Strains 3.734/3.723, 165_Sacc 3.549/3.636, 1342_Fungus 3.604/3.685; CE loss 1.3206/1.3153/1.2768/1.2914.

---

## Figure 2 — Shorkie LM identifies conserved TF motifs across fungal genomes
Package: [`figure_02/`](figure_02/)

| Panel | Claim | Type | Generating script (repo) | Input + config key | GPU | Notebook | Data on disk | Status |
|---|---|---|---|---|---|---|---|---|
| A | SMT3 promoter prediction (chrIV:1,469,090-198) Shorkie-LM vs SpeciesLM; poly(dA:dT),Cbf1,Tye7,Reb1 | comp | `04_analysis/shorkie_lm/lm_SMT3_viz/2_viz_dna_pwm_shorkie_lm.py` | modisco `.h5` → `results.modisco_lm` | no | `fig07` | present (173M .h5) | ⚠️ partial |
| B | PPM reconstruction from DNA via Shorkie LM | comp | `04_analysis/shorkie_lm/motif_analysis/motif_lm/4_viz_motif.py` | modisco `.h5` → `results.modisco_lm` | no | `fig03` | present | ⚠️ GPU |
| C | Motif summary across 6 datasets (TF-MoDISco: TATA, splice donor, branch pt, Cbf1p, Reb1.1, Snf1.1, Mcm1.1, Rap1.1, Sfp1.2, Abf1.1, Dot6) | comp | `…/motif_lm/{1_search_motif.py,2_modisco_script.sh,3_modisco_report.sh}` | modisco `.h5` + MEME DB → `results.modisco_lm`, `motif_db_dir` | no | `fig03`/`fig04` | present (report HTML) | ✅ |
| D | Histograms: motif enrichment upstream of TSS + splice/branch within genes | comp | `…/motif_lm/4_motif_to_tss_dist/3_plot_tss_dist_freq.py` | `motif_tss_distances.csv`, `background_tss_distances.csv` (pipeline output) | no | none | present (~80 PNGs + CSVs) | ✅ |
| E | t-SNE of genomic elements from 1st self-attention layer | comp | `04_analysis/shorkie_lm/umap_cluster_promoter/2_viz_clusters_LM.py` | `embeddings_chr*.h5` (16) → `results.umap`, `genome.gtf` | no | `fig05` | present (~292M) | ✅ |

---

## Figure 3 — Shorkie architecture and RNA-seq prediction performance
Package: [`figure_03/`](figure_03/)

| Panel | Claim | Type | Generating script (repo) | Input + config key | GPU | Notebook | Data on disk | Status |
|---|---|---|---|---|---|---|---|---|
| A | Shorkie arch (U-Net + 8 transformer blocks + heads) | schem | none (params in `02_train/shorkie_finetuned/params.json`) | `models.shorkie_finetuned` | no | none | n/a | ⬚ |
| B | β-estradiol induction experimental schematic | schem | none | `datasets.bigwigs` | no | none | n/a | ⬚ |
| C | Bin-level Pearson R distribution by track type (Shorkie vs Random_Init) | comp | `03_eval/supervised/track_prediction_eval/2_bin_gene_level_metrics/1_bin_level_freq_viz.py` | per-fold `eval/acc.txt` → `results.train_logs` | no | none | present (f0–7) | ⬚ |
| D | Scatter bin-level R (Random_Init x vs Shorkie y) | comp | `…/2_bin_gene_level_metrics/3_gene_level_score_dist_viz.py` (track level) | `eval/acc.txt` → `results.train_logs` | no | `fig09`(partial) | present | ⬚ |
| E | Scatter gene-level R | comp | `…/2_bin_gene_level_metrics/3_gene_level_score_dist_viz.py` (gene) | `gene_level_eval_rc/.../gene_acc.txt` → `results.train_logs` | no | `fig09` | present | ⬚ |
| F | Scatter qnorm mean-centered gene-level R | comp | `…/3_gene_level_score_dist_viz.py` (pearsonr_norm) | `gene_acc.txt` → `results.train_logs` | no | `fig09` | present | ⬚ |
| G | Gene-by-gene track-level R | comp | `…/2_bin_gene_level_metrics/4_track_level_score_diff_viz.py` | `gene_acc.txt` → `results.train_logs` | no | none | present | ⬚ |
| H | Coverage chrVII:362,180–366,023 (RPL7A): obs vs Shorkie vs Random_Init | gpu | `…/3_viz_rnaseq_tracks/2_yeast_rna_seq_models.py` | ensemble + bigwig → `models.shorkie_finetuned`,`datasets.bigwigs`,`genome.*` | yes | `fig08` | partial | ⬚ |
| I | Coverage chrIV:305,657–310,505 (RPS16B,RPL13A) | gpu | same | same | yes | `fig08` | partial | ⬚ |
| J | Coverage chrVII:495,374–499,965 (EFM5) | gpu | same | same | yes | `fig08` | partial | ⬚ |

Anchors: bin-R 0.78 vs 0.67; gene-R 0.88 vs 0.74; Shorkie>Random_Init in 87.8% of genes.

---

## Figure 4 — Promoter & splicing motifs learned during pretraining
Package: [`figure_04/`](figure_04/)

| Panel | Claim | Type | Generating script (repo) | Input + config key | GPU | Notebook | Data on disk | Status |
|---|---|---|---|---|---|---|---|---|
| A–C | Promoter ISM rows (LM logo / Shorkie ISM / Random_Init ISM / annot) for RPL26A, FUN12, KRE33 | gpu | `04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/1_plot_dna_logo_general.py` | ISM `scores.h5` → `results.ism_scores` | ISM-gen yes | `fig13` | present (`revision_experiments/motif_random_init_RP_TSS/gene_exp_motif_test_RP/f0c0/.../scores.h5`) | ⬚ |
| D | Canonical S. cerevisiae splicing motifs | schem | `…/motif_shorkie__RP_TSS/4_plot_dna_SS.py` | reference motifs | no | none | n/a | ⬚ |
| E–G | Splicing ISM maps for DTD1, MMS2, HOP2 | gpu | `…/4_plot_dna_SS.py` | ISM `scores.h5` (SS windows) → `results.ism_scores` | ISM-gen yes | `fig13` | present (`…/gene_exp_motif_test_SS/f0c0/.../scores.h5`) | ⬚ |
| H | TF-MoDISco motifs on Shorkie ISM (curated DB top / Shorkie-derived bottom) | comp | `…/2_modisco_analysis/4_viz_motif.py` | modisco `.h5` → `results.modisco_ism` | no | `fig13` | present (`…/1_modisco_analysis/.../modisco_results_10000_500.h5`) | ⬚ |

---

## Figure 5 — Time-course stress-responsive TF induction  ⚠️ NO existing notebook (largest gap)
Package: [`figure_05/`](figure_05/)

| Panel | Claim | Type | Generating script (repo) | Input + config key | GPU | Notebook | Data on disk | Status |
|---|---|---|---|---|---|---|---|---|
| A | MSN2@ATG42 (chrII:515,214-714) ISM logos × 7 time points | comp/gpu | `04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/3_timepoint_analysis/0_timepoint_viz_scores_h5.py` | `…/2_timepoint_analysis/gene_exp_motif_test_MSN2_targets/f0c0/part*/scores.h5` (`results.ism_scores`) | ISM-gen yes | none | present (T0–T90) | ⬚ |
| B | MSN2 exp vs predicted fold-change/time | comp | `…/3_timepoint_analysis/1_timepoint_viz_scores_h5_diff.py` | same scores.h5 | no | none | present | ⬚ |
| C | MSN2 pairwise Euclidean-dist heatmap | comp | `…/3_timepoint_analysis/2_timepoint_viz_scores_h5_pairwise.py` | same scores.h5 | no | none | present | ⬚ |
| D | MSN2 ΔT TF-MoDISco motifs | comp | `…/3_timepoint_analysis/modisco_analysis/4_viz_motif.py` | `…/modisco_results_10000_500_diff.h5` (7 T) | no | none | present | ⬚ |
| E | MSN2 normalized Pearson-R boxplot/time | comp | `…/3_timepoint_analysis/0_timepoint_viz_scores_h5.py` | scores.h5 | no | none | present | ⬚ |
| F–J | MSN4@TSL1 (chrXIII:70,173-673), analogous to A–E | comp/gpu | same scripts (MSN4 targets) | `…/gene_exp_motif_test_MSN4_targets/...` | ISM-gen yes | none | present | ⬚ |

Anchors: MSN2 genome-wide R 0.55–0.65; MSN4 0.45–0.70.

---

## Figure 6 — MPRA promoter variant effects
Package: [`figure_06/`](figure_06/)

| Panel | Claim | Type | Generating script (repo) | Input + config key | GPU | Notebook | Data on disk | Status |
|---|---|---|---|---|---|---|---|---|
| A | MPRA insertion schematic (100–200 bp upstream, 10 bp steps) | schem | none | none | no | none | n/a | ⬚ |
| B | AUROC high vs low expr × insertion site (3 quantile groups) | comp | `04_analysis/shorkie/mpra/4_mpra_high_low_seq/5_MPRA_classifier_avg.py` | NPZ → `results.mpra_viz` | no | `fig12` | partial | ⬚ |
| C | AUPRC same | comp | same | `results.mpra_viz` | no | `fig12` | partial | ⬚ |
| D | Shorkie logSED vs measured (native; r=0.644, ρ=0.660) | comp | `04_analysis/shorkie/mpra/5_mpra_viz/MPRA_scatter_regression_single.py` | NPZ + `data/MPRA/filtered_test_data_with_MAUDE_expression.txt` → `results.mpra_viz`,`datasets.mpra` | no | `fig12` | present | ⬚ |
| E | Same, challenging sequences | comp | same | `results.mpra_viz` | no | `fig12` | partial | ⬚ |
| F | SNV variant set (ref vs alt) | comp | `04_analysis/shorkie/eqtl/2_variant_scoring/` + `ism_motif/motif_shorkie_ism__snp/` | `results.eqtl_scores`/`results.mpra_eval` | no | none | partial | ⬚ |
| G | Motif perturbations (dual-seq) | comp | `04_analysis/shorkie/eqtl/2_variant_scoring/` | `results.eqtl_scores` | no | none | partial | ⬚ |
| H | Motif tiling constructs | comp | `04_analysis/shorkie/eqtl/2_variant_scoring/` | `results.eqtl_scores` | no | none | partial | ⬚ |
| I | Endogenous RNA-seq coverage (Shorkie vs DREAM-RNN) — domain specificity | heavy/gpu | `mpra/2_hound_mpra_run/` + `MPRA_RNASeq/8_Shorkie_scatter_plot_bigwig.py` | `datasets.bigwigs` | yes | none | present | ⬚ |

---

## Figure 7 — cis-eQTL variant effects
Package: [`figure_07/`](figure_07/)

| Panel | Claim | Type | Generating script (repo) | Input + config key | GPU | Notebook | Data on disk | Status |
|---|---|---|---|---|---|---|---|---|
| A | OMA1 eQTL (chrXI:603,195-604,232) alt reduces expr | gpu | `04_analysis/shorkie/eqtl/2_variant_scoring/score_variants_shorkie.py` | ensemble + eQTL TSV + `genome.fasta` → `models.shorkie_finetuned`,`datasets.eqtl` | yes | `fig10` | present (`revision_experiments/eQTL/caudal_etal_Shorkie/positive/results/`) | ⬚ |
| B | LAP3 eQTL (chrXIV:200,569-201,933) alt increases expr | gpu | same | same | yes | `fig10` | present | ⬚ |
| C | logFC computation schematic | schem | none | none | no | `fig10`(text) | n/a | ⬚ |
| D | Negative eQTL control generation (~1000 isolates, MAF≥5%) | comp | `04_analysis/shorkie/eqtl/0_data_generation/1_generate_negs.py` | gVCF+GTF → `datasets.eqtl`,`genome.gtf` | no | none | present (`viz_new/results/negset_{1..4}/`) | ⬚ |
| E–G | PR/ROC (Shorkie/Random_Init/Shorkie_LM/DREAM) for Caudal, Kita, Renganaath | comp | `04_analysis/shorkie/eqtl/3_visualization/1_roc_pr_shorkie_fold.py` | `*_scores.tsv` + DREAM TSVs → `results.eqtl_scores`,`results.mpra_eval` | no | `fig11` | present | ⬚ |
| H–I | AUPRC by TSS-distance bins (Caudal, Kita) | comp | `04_analysis/shorkie/eqtl/3_visualization/2_AUROC_AUPRC_by_dsitance.py` | scores TSVs (distance) → `results.eqtl_scores`,`results.mpra_eval` | no | `fig11` | present | ⬚ |
| J–O | ISM maps at eQTL SNPs (polyA create/loss, PAC, Reb1 weak/strong) | gpu | `04_analysis/shorkie/ism_motif/motif_shorkie_ism__snp/2_plot_dna_logo_general.py` | ISM `scores.h5` → `results.ism_scores` | yes | `fig13` | partial (`…/motif_shorkie_ism_snp/.../scores.h5`) | ⬚ |

Anchors: Caudal ~1,901 eQTLs; Kita 683; Renganaath 142; Shorkie > ablations & DREAM across datasets and TSS-distance bins.
