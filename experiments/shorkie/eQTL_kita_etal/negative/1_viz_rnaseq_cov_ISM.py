#!/usr/bin/env python3
import json
import os
import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pysam
import pyfaidx
import tensorflow as tf
import scipy.stats as stats
import pyranges as pr

from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dataset
from yeast_helpers_selfsupervised import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


##############################
# Argument parsing
##############################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot logSED for a single gene / SNP."
    )
    parser.add_argument(
        "--gene",
        required=True,
        help="Gene ID (e.g. YNL239W).",
    )
    parser.add_argument(
        "--center_pos",
        type=int,
        required=True,
        help="Center position of the SNP (genomic coordinate, 1-based).",
    )
    parser.add_argument(
        "--pos",
        type=int,
        required=True,
        help="Exact SNP position (should match center_pos).",
    )
    parser.add_argument(
        "--alt",
        choices=["A", "C", "G", "T"],
        required=True,
        help="Alternative allele (one of A/C/G/T).",
    )
    parser.add_argument(
        "--condition",
        default="test",
        help="Name of the condition/subdirectory under ./viz_tracks to save figures.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=16384,
        help="Sequence length to extract around center_pos (must be even).",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=8,
        help="Number of trained folds to ensemble.",
    )
    parser.add_argument(
        "--params_file",
        default="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/"
                "exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/"
                "self_supervised_unet_small_bert_drop/params.json",
        help="Path to model params JSON.",
    )
    parser.add_argument(
        "--targets_file",
        default="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/seq_experiment/"
                "exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/"
                "cleaned_sheet_RNA-Seq_T0.txt",
        help="Path to targets TSV (index_col=0, sep='\\t').",
    )
    parser.add_argument(
        "--gtf_file",
        default="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/"
                "ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/"
                "data_r64_gtf/gtf/GCA_000146045_2.59.gtf",
        help="Path to GTF (gene annotation).",
    )
    parser.add_argument(
        "--fasta_file",
        default="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/"
                "ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/"
                "data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta",
        help="Path to FASTA (genome sequence).",
    )
    parser.add_argument(
        "--output_dir",
        default="./viz_tracks",
        help="Base directory in which per-condition subfolder will be created.",
    )
    return parser.parse_args()

def plot_coverage_track_pair_bins_w_ref_zoomed(
    y_wt, y_mut, chrom, start,
    search_gene, center_pos, gene_start, gene_end, strand,
    poses, track_indices, track_names, track_scales,
    track_transforms, clip_softs, snpweight,
    y_ground_truth=None, ref_track_indices=None,
    log_scale=False, sqrt_scale=False, plot_mut=True,
    plot_window=4096, normalize_window=4096,
    bin_size=32, pad=16, rescale_tracks=True,
    normalize_counts=False, save_figs=False,
    save_suffix="default", save_dir="./",
    gene_slice=None, anno_df=None, margin=100
):
    """
    Zoomed coverage plotting over [min(center_pos, gene_start) - margin,
    max(center_pos, gene_end) + margin], in binned coordinates.
    """
    # 1) define the zoom region in genomic coords
    region_start = min(center_pos, gene_start) - margin
    region_end   = max(center_pos, gene_end)   + margin

    # 2) convert to bin indices (once)
    plot_start_bin = (region_start - start) // bin_size - pad
    plot_end_bin   = (region_end   - start) // bin_size - pad
    normalize_start = center_pos - normalize_window // 2
    normalize_end   = center_pos + normalize_window // 2
    normalize_start_bin = (normalize_start - start) // bin_size - pad
    normalize_end_bin   = (normalize_end - start) // bin_size - pad
    center_bin     = (center_pos - start)    // bin_size - pad
    gene_start_bin = (gene_start - start)    // bin_size - pad
    gene_end_bin   = (gene_end   - start)    // bin_size - pad
    mut_bin        = (poses[0]   - start)    // bin_size - pad

    # 3) extract annotation bins
    anno_bins = []
    if anno_df is not None:
        df = anno_df.query(
            f"chrom=='{chrom}' and position_hg38>={region_start} and position_hg38<{region_end}"
        )
        anno_bins = [(pos - start)//bin_size - pad for pos in df['position_hg38'].values]

    # 4) prepare mean‐across‐axes vectors (will undo transforms inside loop)
    #    note: we keep full per‐track tensors for transform, but collapse axes now
    wt_base = np.mean(y_wt, axis=(0,1,3))
    mut_base = np.mean(y_mut, axis=(0,1,3))
    gt_base = None
    if y_ground_truth is not None:
        gt_base = np.mean(y_ground_truth, axis=(0,1,3))

    logSED = None

    # 5) per‐track plotting
    for track_name, track_index, track_scale, track_transform, clip_soft in zip(
        track_names, track_indices, track_scales, track_transforms, clip_softs
    ):
        # Plot track densities (bins)
        y_wt_curr = np.array(np.copy(y_wt), dtype=np.float32)
        y_mut_curr = np.array(np.copy(y_mut), dtype=np.float32)

        if rescale_tracks:
            # undo scale
            y_wt_curr /= track_scale
            y_mut_curr /= track_scale

            # undo soft_clip
            if clip_soft is not None:
                y_wt_curr_unclipped = (y_wt_curr - clip_soft) ** 2 + clip_soft
                y_mut_curr_unclipped = (y_mut_curr - clip_soft) ** 2 + clip_soft

                unclip_mask_wt = y_wt_curr > clip_soft
                unclip_mask_mut = y_mut_curr > clip_soft

                y_wt_curr[unclip_mask_wt] = y_wt_curr_unclipped[unclip_mask_wt]
                y_mut_curr[unclip_mask_mut] = y_mut_curr_unclipped[unclip_mask_mut]

            # undo sqrt
            y_wt_curr = y_wt_curr ** (1.0 / track_transform)
            y_mut_curr = y_mut_curr ** (1.0 / track_transform)

        y_wt_curr = np.mean(y_wt_curr[..., track_index], axis=(0, 1, 3))
        y_mut_curr = np.mean(y_mut_curr[..., track_index], axis=(0, 1, 3))

        if normalize_counts:
            wt_count = np.sum(y_wt_curr[normalize_start_bin:normalize_end_bin])
            mut_count = np.sum(y_mut_curr[normalize_start_bin:normalize_end_bin])

            # Normalize to densities
            y_wt_curr /= wt_count
            y_mut_curr /= mut_count

            # Bring back to count space (wt reference)
            y_wt_curr *= wt_count
            y_mut_curr *= wt_count

        if gene_slice is not None:
            print("y_wt_curr[gene_slice].shape: ", y_wt_curr[gene_slice].shape)
            print("y_mut_curr[gene_slice].shape: ", y_mut_curr[gene_slice].shape)
            sum_wt = np.sum(y_wt_curr[gene_slice])
            sum_mut = np.sum(y_mut_curr[gene_slice])
            logSED = np.log2(sum_mut + 1) - np.log2(sum_wt + 1)
            print(" - sum_wt = " + str(round(sum_wt, 4)))
            print(" - sum_mut = " + str(round(sum_mut, 4)))
            print("logSED sum_mut / sum_wt : ", logSED)

        y_wt_curr = y_wt_curr[plot_start_bin:plot_end_bin]
        y_mut_curr = y_mut_curr[plot_start_bin:plot_end_bin]

        if log_scale:
            y_wt_curr = np.log2(y_wt_curr + 1.0)
            y_mut_curr = np.log2(y_mut_curr + 1.0)
        elif sqrt_scale:
            y_wt_curr = np.sqrt(y_wt_curr + 1.0)
            y_mut_curr = np.sqrt(y_mut_curr + 1.0)

        max_y_wt = np.max(y_wt_curr)
        max_y_mut = np.max(y_mut_curr)

        if plot_mut:
            max_y = max(max_y_wt, max_y_mut)
        else:
            max_y = max_y_wt

        print(" - max_y_wt = " + str(round(max_y_wt, 4)))
        print(" - max_y_mut = " + str(round(max_y_mut, 4)))
        
        
        gt = None
        # set up subplots
        if gt is not None:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,4), sharex=True)
        else:
            # fig, ax1 = plt.subplots(1,1, figsize=(18,2))
            fig, ax1 = plt.subplots(1,1, figsize=(16,1.7))
            ax2 = None

        x = np.arange(plot_start_bin, plot_end_bin)
        # highlight SNP ± margin
        highlight_start_bin = ((center_pos - 40) - start)//bin_size - pad
        highlight_end_bin   = highlight_start_bin + 6        # Highlight SNP region ±margin
        # draw shaded region behind everything
        ax1.axvspan(
            highlight_start_bin,
            highlight_end_bin,
            color='lightgrey',
            alpha=0.5,
            zorder=0,
            label="±(40bp) ISM region"
        )
        
        # bars
        ax1.bar(x, y_wt_curr, width=1, alpha=0.6, label="Ref")
        if plot_mut:
            ax1.bar(x, y_mut_curr, width=1, alpha=0.6, label="Alt")
        if ax2 is not None:
            ax2.bar(x, gt, width=1, alpha=0.6, label="Ground Truth")

        # variant & gene lines
        ax1.scatter([mut_bin], [0.05*max_y], s=80, marker='*', color='k', label="SNP")
        ax1.axvline(center_bin, color='k', linestyle=':', label=f"Variant ({center_pos})")
        start_line, end_line = (gene_start_bin, gene_end_bin) if strand=='+' else (gene_end_bin, gene_start_bin)
        ax1.axvline(start_line, color='g', linestyle='--', label=f"{search_gene} start ({gene_start})")
        ax1.axvline(end_line,   color='r', linestyle='-.', label=f"{search_gene} end ({gene_end})")

        # annotation lines
        for b in anno_bins:
            ax1.axvline(b, color='cyan', linewidth=1, alpha=0.5, linestyle='-')

        # labels and legend
        ax1.set_xlim(plot_start_bin, plot_end_bin)
        ax1.set_ylabel("Coverage")
        ax1.set_title(f"{search_gene} {track_name} logSED={logSED:.3f}", fontsize=14)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02,1), fontsize=8)
        if ax2:
            ax2.set_xlim(plot_start_bin, plot_end_bin)
            ax2.set_ylabel("GT Signal")
            ax2.set_xlabel(f"{chrom}:{region_start}-{region_end}bp")
            ax2.legend(loc="upper left", bbox_to_anchor=(1.02,1), fontsize=8)
        else:
            ax1.set_xlabel(f"{chrom}:{region_start}-{region_end}bp")

        plt.tight_layout()

        # optional saving
        if save_figs:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{search_gene}_{track_name}_{save_suffix}.png"), dpi=300)
            plt.close(fig)

    # return logSED


def plot_cov_plot(
    gene_id, gene_obj, gene_start, gene_end, strand,
    center_pos, chrom, poses, alts, models,
    fasta_open, condition, targets_df, seq_len, output_dir,
    margin=100
):
    seqnn_model = models[0]

    # Determine sequence window
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    seq_out_start = start + seqnn_model.model_strides[0] * seqnn_model.target_crops[0]
    seq_out_len = seqnn_model.model_strides[0] * seqnn_model.target_lengths[0]

    # Determine output positions for gene exons
    gene_slice = gene_obj.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)
    # gene_slice = gene_obj.output_slice(gene_start, gene_end-gene_start, seqnn_model.model_strides[0], False)
    print(f"seqnn_model.model_strides[0]: {seqnn_model.model_strides[0]}, seqnn_model.target_crops[0]: {seqnn_model.target_crops[0]}")
    print(f"seq_out_start: {seq_out_start}, seq_out_len: {seq_out_len}")
    print(f"gene_start: {gene_start}, gene_end: {gene_end}, gene_slice: {gene_slice}")

    sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)
    sequence_one_hot_mut = np.copy(sequence_one_hot_wt)

    # Introduce the SNP (mutation)
    for pos, alt in zip(poses, alts):
        alt_ix = -1
        if alt == 'A':
            alt_ix = 0
        elif alt == 'C':
            alt_ix = 1
        elif alt == 'G':
            alt_ix = 2
        elif alt == 'T':
            alt_ix = 3
        sequence_one_hot_mut[pos - start - 1, :4] = 0.
        sequence_one_hot_mut[pos - start - 1, alt_ix] = 1.
        
    # ---- Ensemble Prediction: Average across eight models (model averaging only) ----
    def predict_tracks_ensemble(model_list, sequence):
        preds = [predict_tracks([model], sequence) for model in model_list]
        return np.mean(np.array(preds), axis=0)  # retains shape (1,1,length, num_tracks)

    y_wt = predict_tracks_ensemble(models, sequence_one_hot_wt)
    y_mut = predict_tracks_ensemble(models, sequence_one_hot_mut)

    # ------------------------
    # 1) aggregated‐track logSED
    cov_wt_all = np.mean(y_wt, axis=(0,1,3))  # mean over folds/dummy & tracks
    cov_mt_all = np.mean(y_mut, axis=(0,1,3))
    sum_ref_all = cov_wt_all[gene_slice].sum()
    sum_alt_all = cov_mt_all[gene_slice].sum()
    logSED_agg  = np.log2(sum_alt_all + 1) - np.log2(sum_ref_all + 1)

    # ------------------------
    # 2) per‐track logSED & their average
    cov_wt = np.mean(y_wt, axis=(0,1))  # (L, T)
    cov_mt = np.mean(y_mut, axis=(0,1))
    logSED_tracks = []
    for t in range(cov_wt.shape[1]):
        sr = cov_wt[gene_slice, t].sum()
        sa = cov_mt[gene_slice, t].sum()
        logSED_tracks.append(np.log2(sa+1) - np.log2(sr+1))
    logSED_mean_pertrack = float(np.mean(logSED_tracks))

    print(f"* aggregated‐track logSED:          {logSED_agg:.6f}")
    print(f"* per‐track logSED (mean across T): {logSED_mean_pertrack:.6f}")


    save_dir = os.path.join(output_dir, condition)
    os.makedirs(save_dir, exist_ok=True)
    save_suffix = f"_chr{chrom}_{center_pos}_zoom"
    plot_window = 16384 - 2 * 64 * 16
    # Call zoomed plotting function
    plot_coverage_track_pair_bins_w_ref_zoomed(
        y_wt, 
        y_mut, 
        chrom, 
        start,
        gene_id, 
        center_pos, 
        gene_start, 
        gene_end, 
        strand,
        poses,
        track_indices=[range(y_wt.shape[-1])],
        track_names=["Average"],
        track_scales=[1],
        track_transforms=[1],
        clip_softs=[384.0],
        snpweight=0.0,
        plot_mut=True,
        plot_window=plot_window,
        normalize_window=plot_window,
        bin_size=16,
        pad=64,
        rescale_tracks=True,
        normalize_counts=False,
        save_figs=True,
        save_suffix=save_suffix,
        save_dir=save_dir,
        gene_slice=gene_slice,
        anno_df=None,
        margin=100
    )

    return logSED_agg, logSED_mean_pertrack


##############################
# Main execution
##############################
def main():
    args = parse_args()

    gene = args.gene
    center_pos = args.center_pos
    pos = args.pos
    alt = args.alt
    condition = args.condition
    seq_len = args.seq_len
    num_folds = args.num_folds

    # Load parameters, targets, transcriptome, FASTA
    with open(args.params_file) as pf:
        params = json.load(pf)
    params_model = params["model"]
    params_train = params["train"]
    num_species = 165
    params_model["num_features"] = num_species + 5

    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    target_index = targets_df.index
    print("T0 targets track number:", len(target_index))

    # Load one fold first, just to get model‐strides / target_crops / target_lengths
    models = []
    for fold in range(num_folds):
        fold_param = f"f{fold}c0"
        model_file = os.path.join(
            os.path.dirname(args.params_file),
            "train",
            fold_param,
            "train",
            "model_best.h5",
        )

        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, trunk=False, by_name=False)        
        seqnn_model.build_slice(target_index)
        seqnn_model.build_ensemble(True, [0])
        models.append(seqnn_model)

    plot_model = models[0]

    fasta_open = pysam.Fastafile(args.fasta_file)
    transcriptome = bgene.Transcriptome(args.gtf_file)

    # Prepare GTF‐derived coordinate table (unused below, but kept for completeness)
    gtf_df = pd.read_csv(
        args.gtf_file,
        sep="\t",
        skiprows=5,
        names=[
            "Chromosome",
            "havana_str",
            "feature",
            "Start",
            "End",
            "feat1",
            "Strand",
            "feat2",
            "id_str",
        ],
    )
    gtf_df = gtf_df.query("feature == 'gene'").copy().reset_index(drop=True)
    gtf_df = gtf_df.loc[gtf_df["id_str"].str.contains("gene_name")].copy().reset_index(drop=True)
    gtf_df["gene_id"] = gtf_df["id_str"].apply(lambda x: x.split('gene_id "')[1].split('";')[0].split(".")[0])
    gtf_df["gene_name"] = gtf_df["id_str"].apply(lambda x: x.split('gene_name "')[1].split('";')[0].split(".")[0])
    gtf_df = (
        gtf_df[["Chromosome", "Start", "End", "gene_id", "feat1", "Strand", "gene_name"]]
        .drop_duplicates(subset=["gene_id"], keep="first")
        .copy()
        .reset_index(drop=True)
    )
    gtf_df["Chromosome"] = "chr" + gtf_df["Chromosome"]
    gtf_df["Start"] -= 1

    # Identify the matching transcriptome gene object
    gene_keys = [gkey for gkey in transcriptome.genes.keys() if gene in gkey]
    if len(gene_keys) == 0:
        raise ValueError(f"Gene '{gene}' not found in transcriptome.")
    gene_obj = transcriptome.genes[gene_keys[0]]
    strand = gene_obj.strand
    gene_exons = gene_obj.get_exons()
    gene_start = gene_exons[0][0]
    gene_end = gene_exons[-1][1]

    chrom = "chr" + gene_obj.chrom
    poses = [pos]
    alts = [alt]
    print(f"gene_start: {gene_start}, gene_end: {gene_end}, chrom: {chrom}, poses: {poses}, alts: {alts}")
    print(f"gene_obj: {gene_obj}")

    # Run plotting and scoring
    logSED_agg, logSED_mean = plot_cov_plot(
        gene,
        gene_obj,
        gene_start,
        gene_end,
        strand,
        center_pos,
        chrom,
        poses,
        alts,
        models,
        fasta_open,
        condition,
        targets_df,
        seq_len,
        args.output_dir
    ) 

    # write out both scores
    out_dir = os.path.join(args.output_dir, args.condition)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"logSED_scores_{args.gene}_{chrom}_{args.center_pos}.txt"
    )
    with open(out_file, "w") as fw:
        fw.write(
            f"{args.gene}\t"
            f"{logSED_agg:.6f}\t"
            f"{logSED_mean:.6f}\n"
        )

    fasta_open.close()


if __name__ == "__main__":
    main()


    # # Write out single line to a text file
    # out_fname = os.path.join(
    #     args.output_dir, condition, f"avg_logSED_scores_{gene}_{chrom}_{center_pos}.txt"
    # )
    # with open(out_fname, "w") as fw_logSED:
    #     fw_logSED.write(f"{gene}\t{avg_logSED}\n")

    # ##############################
    # #Run ISM (gene-specific)
    # ##############################
    # # Determine sequence window
    # start = center_pos - seq_len // 2
    # end = center_pos + seq_len // 2
    # seq_out_start = start + seqnn_model.model_strides[0] * seqnn_model.target_crops[0]
    # seq_out_len = seqnn_model.model_strides[0] * seqnn_model.target_lengths[0]
    # print("start: ", start, "; end: ", end, "; seq_out_start: ", seq_out_start, "; seq_out_len: ", seq_out_len)

    # # Determine output positions for gene exons
    # gene_slice = gene_obj.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)
    # print("gene_slice: ", gene_slice)
    # sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)
    # sequence_one_hot_mut = np.copy(sequence_one_hot_wt)

    # # Introduce the SNP (mutation)
    # for pos, alt in zip(poses, alts):
    #     alt_ix = -1
    #     if alt == 'A':
    #         alt_ix = 0
    #     elif alt == 'C':
    #         alt_ix = 1
    #     elif alt == 'G':
    #         alt_ix = 2
    #     elif alt == 'T':
    #         alt_ix = 3
    #     sequence_one_hot_mut[pos - start - 1, :4] = 0.
    #     sequence_one_hot_mut[pos - start - 1, alt_ix] = 1.

    # #Get contribution scores (ISM)

    # track_index = np.arange(len(targets_df), dtype='int32')
    # ism_width = 80  
    # [pred_ism_wt, pred_ism_mut] = get_ism(
    #     models,
    #     [sequence_one_hot_wt, sequence_one_hot_mut],
    #     ism_start=(center_pos - start) - ism_width//2,
    #     ism_end=(center_pos - start) + ism_width//2,
    #     prox_bin_start=0,
    #     prox_bin_end=1,
    #     dist_bin_start=0,
    #     dist_bin_end=1,
    #     track_index=track_index,
    #     track_scale=1.,
    #     track_transform=1.,
    #     clip_soft=None,
    #     dist_bin_index=gene_slice.tolist(),
    #     use_mean=True,
    #     use_ratio=False,
    #     use_logodds=False,
    #     untransform_old=False,
    # )
    # # save the ISM results
    # ism_results = {
    #     'pred_ism_wt': pred_ism_wt,
    #     'pred_ism_mut': pred_ism_mut,
    #     'gene': gene,        
    #     'gene_slice': gene_slice,
    #     'start': start,
    #     'end': end,
    #     'center_pos': center_pos,
    #     'chrom': chrom,
    #     'poses': poses,
    #     'alts': alts
    # }
    # os.makedirs('ism_results', exist_ok=True)
    # # Save the ISM results to a file npz
    # np.savez(f'ism_results/{gene}_{chrom}_{center_pos}.npz', **ism_results)