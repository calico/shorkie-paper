#!/usr/bin/env python3
"""
Refactored Step 2 script that:
1) Reads the transcript overlaps from Step 1 (0_transcript_fold_overlaps.csv).
2) For each transcript overlap, determines the correct fold_param (e.g. fold4 -> f4c0).
3) Loads the relevant supervised/self-supervised models (cached) once per fold_param.
4) Extracts coverage in each overlap range and generates coverage plots.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import pysam
import pyranges as pr
import tensorflow as tf

# From your code snippet or local modules
import matplotlib.pyplot as plt
import h5py
# custom modules
from optparse import OptionParser

# Baskerville / custom code
from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dataset
from baskerville import layers
from yeast_helpers import *
from load_cov import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

RC = True
################################################################################
# 1) Create a mapping from "foldX" → "fXc0"
################################################################################
fold_mapping = {
    'fold0': 'f0c0',
    'fold1': 'f1c0',
    'fold2': 'f2c0',
    'fold3': 'f3c0',
    'fold4': 'f4c0',
    'fold5': 'f5c0',
    'fold6': 'f6c0',
    # add more as needed
}

################################################################################
# 2) Helper functions to parse group/time from target descriptions
################################################################################

def parse_group(desc: str) -> str:
    """Classify coverage tracks by group based on description text."""
    if "Chip-exo" in desc or "_pos_logFE" in desc:
        return "Chip-exo"
    elif "Chip-MNase" in desc:
        return "Chip-MNase"
    elif "1000 strains RNAseq" in desc:
        return "1000-RNA-seq"
    elif "RNAseq" in desc:
        return "RNA-seq"
    else:
        return "Other"

def parse_time_group(identifier: str) -> str:
    """
    Extract time group from a track identifier (e.g. "T0", "T5", etc.)
    Adjust logic as needed.
    """
    if "_T0_" in identifier:
        return "T0"
    elif "_T5_" in identifier:
        return "T5"
    elif "_T10_" in identifier:
        return "T10"
    elif "_T15_" in identifier:
        return "T15"
    elif "_T20_" in identifier:
        return "T20"
    elif "_T25_" in identifier:
        return "T25"
    elif "_T30_" in identifier:
        return "T30"
    elif "_T35_" in identifier:
        return "T35"
    elif "_T40_" in identifier:
        return "T40"
    elif "_T45_" in identifier:
        return "T45"
    elif "_T50_" in identifier:
        return "T50"
    elif "_T55_" in identifier:
        return "T55"
    elif "_T60_" in identifier:
        return "T60"
    elif "_T65_" in identifier:
        return "T65"
    elif "_T70_" in identifier:
        return "T70"
    elif "_T75_" in identifier:
        return "T75"
    elif "_T80_" in identifier:
        return "T80"
    elif "_T85_" in identifier:
        return "T85"
    elif "_T90_" in identifier:
        return "T90"
    elif "_T120_" in identifier:
        return "T120"
    elif "_T180_" in identifier:
        return "T180"
    else:
        return "Other"

def map_chromosome_to_roman(chromosome: str) -> str:
    """
    Example converter from "chromosome1" to "chrI".
    Adjust or remove if not needed.
    """
    roman_mapping = {
        'chromosome1': 'chrI',
        'chromosome2': 'chrII',
        'chromosome3': 'chrIII',
        'chromosome4': 'chrIV',
        'chromosome5': 'chrV',
        'chromosome6': 'chrVI',
        'chromosome7': 'chrVII',
        'chromosome8': 'chrVIII',
        'chromosome9': 'chrIX',
        'chromosome10': 'chrX',
        'chromosome11': 'chrXI',
        'chromosome12': 'chrXII',
        'chromosome13': 'chrXIII',
        'chromosome14': 'chrXIV',
        'chromosome15': 'chrXV',
        'chromosome16': 'chrXVI',
    }
    return roman_mapping.get(chromosome, chromosome)

################################################################################
# 3) Load or restore models for a given fold_param
################################################################################

def get_models_for_fold(fold_param, seq_len, target_index, root_dir):
    """
    Build and return both supervised and self-supervised models for a given fold_param.
    
    Args:
      fold_param: e.g. 'f4c0', 'f5c0', etc.
      seq_len:    (int) length of input sequences to each model
      target_index: DataFrame index for the subset of coverage tracks (targets)
      root_dir:   path to your environment directory

    Returns:
      (models_supervised, models_selfsupervised)
    """
    models_supervised = []
    models_selfsupervised = []

    for model_type in ["supervised", "self_supervised"]:
        tf.keras.backend.clear_session()
        print(f"Loading {model_type} model for fold_param={fold_param}")

        if model_type == "supervised":
            params_file = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/{model_type}_unet_small_bert_drop_variants/learning_rate_0.0005/train/{fold_param}/train/params.json'
            model_file  = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/{model_type}_unet_small_bert_drop_variants/learning_rate_0.0005/train/{fold_param}/train/model_best.h5'
        else:
            params_file = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/{model_type}_unet_small_bert_drop/train/{fold_param}/train/params.json'
            model_file  = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/{model_type}_unet_small_bert_drop/train/{fold_param}/train/model_best.h5'

        # Read JSON params
        with open(params_file) as params_open:
            params = json.load(params_open)
            params_model = params['model']
            params_train = params['train']

        # Adjust model features
        if model_type == 'self_supervised':
            # For example, your environment might need 165 + 5 = 170 features
            num_species = 165
            params_model["num_features"] = num_species + 5
        else:
            # supervised
            params_model["num_features"] = 4

        params_model["seq_length"] = seq_len

        # Initialize & restore
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, trunk=False)
        seqnn_model.build_slice(target_index)
        seqnn_model.build_ensemble(RC, [0])

        if model_type == 'supervised':
            models_supervised.append(seqnn_model)
        else:
            models_selfsupervised.append(seqnn_model)

    return models_supervised, models_selfsupervised

################################################################################
# 4) Coverage plotting function (your existing Step 2 logic with minor changes)
################################################################################

def plot_cov_plot(
    search_gene,
    gene,               # A bgene.Gene object from transcriptome, or None
    gene_start,
    gene_end,
    center_pos,
    chrom,
    seq_len,
    seqnn_model,
    models_supervised,
    models_selfsupervised,
    fasta_open,
    targets_df,
    gene_pr,
    dataset_type,
    exon_count,
    fold_param,
    region_start,
    region_end,
    transform
):
    """
    Plot coverage for a given gene or region, using the loaded supervised and
    self-supervised models. This function is adapted from your Step 2 code.

    Args:
      search_gene: string name for the transcript/gene
      gene:  gene object (or None) if you want to retrieve exons, etc.
      gene_start, gene_end: from the transcript
      center_pos: integer around which we center a seq_len window
      chrom:  e.g. 'chrIV'
      seq_len: input length for the model
      seqnn_model, models_supervised, models_selfsupervised: pre-loaded models
      fasta_open: pysam handle to reference .fasta
      targets_df: DataFrame with track metadata
      gene_pr: pyranges object with GTF annotation
      dataset_type: e.g. "RNA-seq"
      exon_count: number of exons
      fold_param: e.g. "f4c0"
      region_start, region_end: the specific overlapping sub-range

    Returns:
      None. Saves coverage plots to disk in a structured directory.
    """

    print(f"[*] Plotting coverage for {search_gene} at {chrom}:{region_start}-{region_end}, fold={fold_param}")

    # 1) Determine output directories
    #    e.g.  {out_dir}/f4c0/chrIV/YDR424C_mRNA/exonCount_3/RNA-seq/range_1319387_1319841
    save_dir = f'{out_dir}/{fold_param}/exonCount_{exon_count}/{chrom}/{search_gene}/{dataset_type}/range_{region_start}_{region_end}'
    os.makedirs(save_dir, exist_ok=True)

    # 2) Set up the coordinate window for model input
    start = center_pos - seq_len // 2
    end   = center_pos + seq_len // 2

    print("* Determined sequence window: ", start, end)

    # 3) Suppose we want to compute the gene slice for annotation
    #    We'll do so only if `gene` is provided
    gene_slice = None
    if gene is not None:
            # gene_slice = gene.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)

        seq_out_start = start + seqnn_model.model_strides[0]*seqnn_model.target_crops[0]
        seq_out_len = seqnn_model.model_strides[0]*seqnn_model.target_lengths[0]

        #Determine output positions of gene exons
        gene_slice = gene.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)
        print("gene_slice: ", gene_slice)

    # 4) Predict coverage with supervised models
    sequence_one_hot_wt_supervised, _ = process_sequence(
        fasta_open, chrom, start, end, gene_pr, model_type='supervised'
    )
    y_wt_supervised = predict_tracks(models_supervised, sequence_one_hot_wt_supervised)
    print("Supervised prediction shape: ", y_wt_supervised.shape)

    # 5) Predict coverage with self-supervised models
    sequence_one_hot_wt_selfsupervised, _ = process_sequence(
        fasta_open, chrom, start, end, gene_pr, model_type='self_supervised'
    )
    y_wt_selfsupervised = predict_tracks(models_selfsupervised, sequence_one_hot_wt_selfsupervised)
    print("Self-supervised prediction shape: ", y_wt_selfsupervised.shape)

    if transform == "sum_sqrt":
        y_wt_supervised = (y_wt_supervised + 1)**2 - 1
        y_wt_selfsupervised = (y_wt_selfsupervised + 1)**2 - 1

    # 6) Extract ground truth coverage from bigWig
    #    Typically, you'd want coverage from [start+1024 : end-1024], or something
    #    consistent with your Step 2 code. Adjust as you see fit.
    pad = 64
    region_cov_start = start + 1024  # example offset
    region_cov_end   = end   - 1024

    y_cov_list = []
    # Some logic to pick coverage tracks relevant to 'dataset_type'
    # E.g., filter your targets_df again here, or assume it's already filtered.
    print("Number of rows in targets_df: ", len(targets_df))
    for idx, row_tg in targets_df.iterrows():
        bw_f = row_tg['file']  # path to bigWig
        cov_values = read_coverage(bw_f, chrom, region_cov_start, region_cov_end)
        y_ref_cov = seq_norm(cov_values)  # shape ~ (length,)
        y_cov_list.append(y_ref_cov)
    # The dimension is (#tracks, length) I want to convert it to (length, #tracks)
    y_ref_cov = np.array(y_cov_list)  # shape ~ (#tracks, length)
    y_ref_cov = y_ref_cov.T  # shape ~ (length, #tracks)
    y_ref_cov = np.expand_dims(y_ref_cov, axis=0)  # -> (1, #tracks, length)
    y_ref_cov = np.expand_dims(y_ref_cov, axis=1)  # -> (1,1,length, #tracks) depends on usage

    # 7) Now call your custom function that actually does the 3-panel coverage plotting
    #    "plot_3_coverage" was in your snippet. Let's assume it's still imported or defined in your code base.
    plot_3_coverage(
        y_gt = y_ref_cov,               # the bigWig ground truth (1,1,length,#chans)
        y_wt_sup = y_wt_supervised,     # supervised preds
        y_wt_selfsup = y_wt_selfsupervised,  # self-supervised preds
        chrom = chrom,
        start = start,
        center_pos = center_pos,
        gene_start = gene_start,
        gene_end   = gene_end,
        plot_window = 16384 - 2 * pad * 16,  # or whatever your step2 logic had
        bin_size   = 16,
        pad        = pad,
        region_mode = 'full',
        invert_16bp_sum = False,
        save_figs  = True,
        save_suffix = f'_{chrom}_{start}_{end}',
        save_dir   = save_dir,
        gene_slice = gene_slice,
        gene_strand= '+',     # adjust if needed
        search_gene= search_gene,
        gene_color = 'deepskyblue'
    )

    print(f"[Done] Coverage plot saved to: {save_dir}")

################################################################################
# 5) Main script logic
################################################################################

def main():
    """
    Main function that:
    1. Reads the step1 overlap file (0_transcript_fold_overlaps.csv).
    2. Reads coverage targets file (cleaned_sheet.txt).
    3. For each row in the overlap file:
       - Identify the fold (fold4 -> f4c0).
       - Identify each overlapping region.
       - Load the relevant models (cached).
       - Plot coverage in that region.
    """

    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option("--dataset_type",
                      dest="dataset_type",
                      default=None,
                      help="Group of dataset to evaluate: 'Chip-exo', 'Chip-MNase', 'RNA-seq', etc.")
    parser.add_option("--time_group",
                      dest="time_group",
                      default="T0",
                      help="Time group to evaluate: e.g. T0, T5, T10, etc.")
    parser.add_option("--root_dir",
                      dest="root_dir",
                      default="../../..",
                      help="Root directory pointing to Yeast_ML paths")
    parser.add_option("--out_dir",
                      dest="out_dir",
                      default="viz_tracks",
                      help="Output directory for generated plots")
    (options, args) = parser.parse_args()

    # ~~~~ Modify these paths as needed ~~~~
    root_dir      = options.root_dir
    overlaps_file = f"{root_dir}/experiments/SUM_data_process/results/0_transcript_fold_overlaps_3.csv"   # Step 1 output
    targets_file  = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet.txt'
    gtf_file      = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf'
    fasta_file    = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'

    dataset_type = options.dataset_type  # e.g. "RNA-seq", "Chip-exo"
    time_group   = options.time_group
    out_dir      = options.out_dir

    seq_len = 16384  # or 32768, etc.

    # 1) Load the Step 1 overlap file
    df_overlaps = pd.read_csv(overlaps_file, sep="\t")
    # Typically has columns: transcript_id, chrom, transcript_start, transcript_end,
    # exon_count, overlapping_fold, overlapping_range

    # 2) Load coverage targets
    targets_df = pd.read_csv(targets_file, sep='\t', index_col=0)
    targets_df["group"] = targets_df["description"].apply(parse_group)

    print("Loaded targets_df shape: ", targets_df.shape)

    if dataset_type is not None:
        print(f"Filtering for dataset_type = {dataset_type}")
        old_count = len(targets_df)
        targets_df = targets_df[targets_df["group"] == dataset_type]
        new_count = len(targets_df)
        print(f"Filtered targets from {old_count} -> {new_count}")

    # If time grouping is relevant for RNA-seq, do it
    if dataset_type == "RNA-seq":
        targets_df["time_group"] = targets_df["identifier"].apply(parse_time_group)
        old_count = len(targets_df)
        targets_df = targets_df[targets_df["time_group"] == time_group]
        new_count = len(targets_df)
        print(f"Filtered RNA-seq time_group from {old_count} -> {new_count}")

    target_index = targets_df.index
    print("target_index: ", target_index)
    print("Final target_df shape: ", targets_df.shape)

    # 3) Load GTF with pyranges
    gene_pr = pr.read_gtf(gtf_file)
    gene_pr = gene_pr[gene_pr.Feature.isin(['gene','exon','five_prime_UTR','three_prime_UTR'])]
    # Optionally build a transcriptome object
    transcriptome = bgene.Transcriptome(gtf_file)

    # 4) Initialize FASTA
    fasta_open = pysam.Fastafile(fasta_file)

    # 5) A dictionary to cache models so we don't reload them repeatedly
    model_cache = {}

    # 6) Iterate over each transcript from Step 1
    for i, row in df_overlaps.iterrows():
        transcript_id = row["transcript_id"]    # e.g. """YDR424C_mRNA"""
        chrom         = row["chrom"]            # e.g. chrIV
        t_start       = row["transcript_start"]
        t_end         = row["transcript_end"]
        exon_count    = row["exon_count"]

        overlapping_fold_str  = row["overlapping_fold"]   # "fold4,fold4,fold4" or "None"
        overlapping_range_str = row["overlapping_range"]  # "1319387-1319841,1319387-1319628,..." or "None"

        if overlapping_fold_str == "None" or overlapping_range_str == "None":
            # No overlap
            continue

        # 6a) Split the folds/ranges
        folds_list  = overlapping_fold_str.split(",")   # e.g. ["fold4","fold4","fold4"]
        ranges_list = overlapping_range_str.split(",")  # e.g. ["1319387-1319841","1319387-1319628","..."]

        # Optionally remove duplicates if needed:
        # folds_ranges = list(set(zip(folds_list, ranges_list)))

        # 7) Get a Gene object from the transcriptome (optional)
        #    Transcript ID might be something like """YDR424C_mRNA"""
        #    If you want the "gene key", adapt how you parse transcript_id
        #    This example tries removing quotes and _mRNA
        gene_name_clean = transcript_id.replace('"','').replace('_mRNA','')

        # Attempt to find the gene in transcriptome
        # Some transcripts might have multiple "keys" that match partially
        matched_gene_keys = [
            k for k in transcriptome.genes.keys() if gene_name_clean in k
        ]
        if matched_gene_keys:
            gene_obj = transcriptome.genes[matched_gene_keys[0]]
        else:
            gene_obj = None  # fallback

        # 8) For each fold-range pair, load the correct model and plot coverage
        for fold_str, range_str in zip(folds_list, ranges_list):
            if fold_str not in fold_mapping:
                print(f"Warning: {fold_str} not in fold_mapping. Skipping.")
                continue

            fold_param = fold_mapping[fold_str]  # e.g. "fold4" -> "f4c0"

            # parse the region string "1319387-1319628"
            try:
                region_start, region_end = range_str.split("-")
                region_start = int(region_start)
                region_end   = int(region_end)
            except:
                print(f"Warning: could not parse {range_str} as start-end. Skipping.")
                continue

            center_pos = (region_start + region_end)//2

            # Load models if not cached
            if fold_param not in model_cache:
                msup, mselfsup = get_models_for_fold(fold_param, seq_len, target_index, root_dir)
                model_cache[fold_param] = (msup, mselfsup)
            else:
                msup, mselfsup = model_cache[fold_param]

            # Grab one model from msup if you want a "primary" reference
            seqnn_model = msup[0] if len(msup) > 0 else None

            # 9) Call your coverage plotting function
            plot_cov_plot(
                search_gene       = gene_name_clean,
                gene              = gene_obj,
                gene_start        = t_start,
                gene_end          = t_end,
                center_pos        = center_pos,
                chrom             = chrom,
                seq_len           = seq_len,
                seqnn_model       = seqnn_model,
                models_supervised = msup,
                models_selfsupervised = mselfsup,
                fasta_open        = fasta_open,
                targets_df        = targets_df,
                gene_pr           = gene_pr,
                dataset_type      = dataset_type if dataset_type else "All",
                exon_count        = exon_count,
                fold_param        = fold_param,
                region_start      = region_start,
                region_end        = region_end,
                transform         = None
                # transform         = "sum_sqrt"
            )

    print("[All Done] Processed overlaps from Step 1.")

################################################################################
if __name__ == '__main__':
    main()
