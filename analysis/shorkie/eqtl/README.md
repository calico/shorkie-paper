# eQTL Benchmarking and Analysis Pipeline

This directory contains the generalized, essential scripts used to benchmark and compare the performance of various models (including `Shorkie`, `Shorkie_LM`, `SpeciesLM`, and MPRA-trained architectures) on eQTL datasets. The scripts have been parameterized to be portable and applicable across multiple datasets (e.g. Renganaath et al., Kita et al., Caudal et al.).

The overall pipeline is structured into five main steps:

## `0a_gwas_preprocessing/`
Contains generalized scripts to filter and extract eQTL information from raw GWAS outputs and subset a reference GVCF.
- `1_snp_position.py`: Annotates GWAS summary stats with local/distant distances.
- `2_extract_eQTL.py`: Intersects GWAS variants with a background GVCF file.
- `3_fix_vcf.py`: Normalizes chromosome headers within output VCF files.
- `4_plot_gwas.py`: Generates manhattan plots of the parsed GWAS data.

## `0_data_generation/`
Contains scripts used to generate and validate negative eQTL sets. These scripts are generalized using `argparse` to handle slightly different variant formatting from the input data formats.
- `1_generate_negs.py`: Generates distance-matched negative control eQTLs by taking positive eQTLs and matching their distance-to-TSS with non-coding common variants from a gVCF background. Specify the required dataset target format via `--dataset {caudal,kita,renganaath}`.
- `2_covert_negs_to_vcf.py`: Formats the generated TSV negative sets into `.vcf` formats for processing models. Takes argument `--negative_tsv_prefix` indicating where `1_generate_negs` outputted.
- `3_predict_neg_eqtl.sh`: Array SLURM script executing bash inference for the resulting sequence combinations. Uses `$ROOT_DIR` and `$NEG_TSV_PREFIX` environment variables to control path parameters cleanly.
- `4_pos_neg_cmp_viz_by_tss.py`: Visualizes performance between negative and positive eQTL score distributions using argparse flags to dynamically locate model results.
- `2_covert_negs_to_vcf.py`: Converts generated TSV negative sets into VCF format.
- `3_predict_neg_eqtl.sh`: Bash wrapper to predict effects of negatives.
- `4_pos_neg_cmp_viz_by_tss.py`: Visualization script comparing the distance-to-TSS distribution of the positive and generated negative sets to ensure they match well.

## `1_baseline_mpra_eval/`
Contains scripts used to evaluate baseline models consisting of PrixFixe architectures trained solely on MPRA (Massively Parallel Reporter Assay) data, evaluated on the eQTL task.
- `1_create_seq_pos.py` & `3_create_seq_neg.py`: Converts variants into structured HDF5/fasta datasets. Requires `--dataset {kita,renganaath,caudal}`.
- `2_predict_seq_pos.py` & `4_predict_seq_neg.py`: Runs baseline inference.
- `5_classifier_eval.py`: Computes AUROC and AUPRC metrics for the baseline models.

## `2_variant_scoring/`
Contains the core scripts for computing zero-shot or few-shot variant effect scores using the Shorkie foundation models. Set required arguments to pass paths mapping to FASTA/GTF references and model `.h5` files.
- `score_variants_shorkie.py`: Scores positive eQTLs using the standard `Shorkie` model architecture (masked language modeling over sequence).
- `score_negative_variants_shorkie.py`: Same as above for negative sets.
- `score_variants_LM.py`: Scores variants using the `Shorkie_LM` architecture (causal/autoregressive formulation).
- `score_negative_variants_LM.py`: Same as above for negative sets.

> **Note:** These scripts are model entry points meant to be wrapped in SLURM iteration scripts for combinations of Models $\times$ Datasets.

## `3_visualization/`
Contains the scripts that aggregate predictions across all evaluated models and datasets to generate the paper figures. Most scripts accept a `--root_dir` argument dynamically mapping back to standard data locations.
- `0_parse_eqtl_res.py`: Consolidates all model variant scores into standard merged output dataframes. 
- `1_roc_pr_shorkie_fold.py`: Generates base AUROC and AUPRC comparison curves.
- `2_AUROC_AUPRC_by_dsitance.py`: Stratifies model performance by the distance of the variant to the Transcription Start Site.
- `3_Shorkie_MPRA_model_quantile.py`: Compares predictions between the Shorkie foundation models and the supervised MPRA baselines.
- `4_interpretability.py`: In-silico mutagenesis (ISM) and sequence attribution mapping scripts.
