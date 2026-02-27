#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import pysam
import pyranges as pr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

from modisco.visualization import viz_sequence

##########################################
# Helper function to plot ACGT letters   #
##########################################
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
        "UP": TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
        "DN": TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
        "(": TextPath((-0.25, 0), "(", size=1, prop=fp),
        ".": TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")": TextPath((-0.1, 0), ")", size=1, prop=fp)
    }
    COLOR_SCHEME = {
        'G': 'orange',
        'A': 'green',
        'C': 'blue',
        'T': 'red',
        'UP': 'green',
        'DN': 'red',
        '(': 'black',
        '.': 'black',
        ')': 'black'
    }
    text = LETTERS[letter]
    chosen_color = COLOR_SCHEME[letter] if color is None else color

    # Build transformation: scale the letter and move it to (x, y)
    t = mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)
    return p

##########################################
# Plot DNA logo using conservation score #
##########################################
def plot_dna_logo(pwm, sequence_template, figsize=(12, 3), logo_height=1.0, 
                  plot_start=0, plot_end=164, plot_sequence_template=False, 
                  save_figs=False, fig_name=None, annotate_positions=None, annotate_labels=None):
    # Slice the PWM and sequence template for the region to be plotted
    pwm = np.copy(pwm[plot_start: plot_end, :])
    sequence_template = sequence_template[plot_start: plot_end]

    # Normalize PWM rows to sum to 1
    pwm += 0.0001
    for j in range(pwm.shape[0]):
        pwm[j, :] /= np.sum(pwm[j, :])

    # Compute per-position entropy and conservation (2 - entropy)
    entropy = np.zeros(pwm.shape)
    entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    height_base = (1.0 - logo_height) / 2.

    # Plot each position
    for j in range(pwm.shape[0]):
        sort_index = np.argsort(pwm[j, :])
        for ii in range(4):
            i = sort_index[ii]
            # Map index to nucleotide
            if i == 0:
                nt = 'A'
            elif i == 1:
                nt = 'C'
            elif i == 2:
                nt = 'G'
            elif i == 3:
                nt = 'T'

            nt_prob = pwm[j, i] * conservation[j]
            color = None
            # If sequence_template is provided and not the placeholder '$', then use it
            if sequence_template[j] != '$':
                color = 'black'
                if plot_sequence_template and nt == sequence_template[j]:
                    nt_prob = 2.0
                else:
                    nt_prob = 0.0

            if ii == 0:
                dna_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, ax, color=color)
            else:
                prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
                dna_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax, color=color)

    # Optionally add annotations
    if annotate_positions is not None:
        for pos, label in zip(annotate_positions, annotate_labels):
            x_pos = pos - plot_start + 0.5
            if 0 <= x_pos <= plot_end - plot_start:
                ax.axvline(x=x_pos, color='red', linestyle='--')
                ax.text(x_pos, 2, label, color='red', ha='center', va='bottom', fontsize=12)

    plt.xlim((0, plot_end - plot_start))
    plt.ylim((0, 2))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)
    for axis in fig.axes:
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    plt.tight_layout()
    if save_figs and fig_name is not None:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")
    plt.show()
    plt.clf()

def parse_fasta_header(header):
    match = re.match(r'>(chr[\w]+):(\d+)-(\d+)\|(\w+)', header)
    return match.groups() if match else (None, None, None, None)

def process_fasta(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    chromosome, start, end, species = parse_fasta_header(line.strip())
                    if chromosome:
                        data.append([chromosome, int(start), int(end), species])
        return pd.DataFrame(data, columns=['Chromosome', 'Start', 'End', 'species'])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame(columns=['Chromosome', 'Start', 'End', 'species'])

def check_window_is_reversed(window, chrom, start, end, fasta_open):
    gene_seq = fasta_open.fetch(chrom, start, end).upper()
    nucleotides = ['A', 'C', 'G', 'T']
    window_gene_seq = ''.join(nucleotides[np.argmax(row)] for row in window)
    return gene_seq != window_gene_seq

def reverse_complement(x_true_gene, x_pred_gene):
    x_true_rev = np.flip(x_true_gene, axis=0)[:, [3, 2, 1, 0]]
    x_pred_rev = np.flip(x_pred_gene, axis=0)[:, [3, 2, 1, 0]]
    return x_true_rev, x_pred_rev

def norm_pwm(pwm):
    pwm += 0.0001
    p_mean = np.mean(pwm, axis=1, keepdims=True)
    pwm = pwm * np.log(pwm / p_mean)
    return pwm

def main():
    # Parameters
    dataset = 'saccharomycetales'
    model_archs = [
        "unet_small_bert_drop",
        "unet_small_bert_drop_retry_1",
        "unet_small_bert_drop_retry_2"
    ]
    # Dictionaries to accumulate overall averages across model architectures
    overall_avg_x_true = {}
    overall_avg_x_pred = {}

    for model_arch in model_archs:
        print(f"Processing model architecture: {model_arch}")
        pad_bp = 512
        coding_viz_len = 500

        # File paths (update as needed)
        fasta_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask'
        gtf_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/gtf/GCA_000146045_2.59.gtf'
        output_dir = f'{dataset}_viz_seq_new_norm/{model_arch}/'
        os.makedirs(output_dir, exist_ok=True)
        fasta_open = pysam.Fastafile(fasta_file)

        # Initialize lists for data
        seqs_df_list = []
        cleaned_seqs_df_list = []
        x_true_list = []
        x_pred_list = []
        label_list = []

        for target_type in ['train', 'test', 'valid']:
            sequences_bed_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/sequences_{target_type}.cleaned.bed'
            cleaned_sequences_fasta_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/extracted_fasta/sequences_{target_type}.cleaned.fasta'
            if target_type == 'test':
                predictions_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_{dataset}_gtf/lm_{dataset}_gtf_{model_arch}/test_testset/preds_{target_type}.npz'
            elif target_type == 'train':
                predictions_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_{dataset}_gtf/lm_{dataset}_gtf_{model_arch}/test_trainset/preds_{target_type}.npz'
            elif target_type == 'valid':
                predictions_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_{dataset}_gtf/lm_{dataset}_gtf_{model_arch}/test_validset/preds_{target_type}.npz'

            print(f"\tProcessing {target_type} dataset")
            print(f"\tPredictions file: {predictions_file}")

            seqs_df = pd.read_csv(sequences_bed_file, sep='\t', names=['Chromosome', 'Start', 'End', 'label', 'species'])
            seqs_df = seqs_df[seqs_df['species'] == "GCA_000146045_2"].reset_index(drop=True)
            seqs_df = seqs_df[seqs_df['label'] == target_type].reset_index(drop=True)
            seqs_df['row_index'] = seqs_df.index
            seqs_df['Strand'] = "."
            seqs_df = seqs_df[['Chromosome', 'Start', 'End', 'species', 'row_index', 'Strand']]
            seqs_df_list.append(seqs_df)
            print(f"\tLoaded {len(seqs_df)} sequences for {target_type}")
        
            cleaned_seqs_df = process_fasta(cleaned_sequences_fasta_file)
            cleaned_seqs_df_list.append(cleaned_seqs_df)

            cache_bundle = np.load(predictions_file)
            x_true_list.append(cache_bundle['x_true'])
            x_pred_list.append(cache_bundle['x_pred'])
            label_list.append(cache_bundle['label'])

        seqs_df = pd.concat(seqs_df_list, ignore_index=True)
        cleaned_seqs_df = pd.concat(cleaned_seqs_df_list, ignore_index=True)
        x_true = np.concatenate(x_true_list, axis=0)
        x_pred = np.concatenate(x_pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)

        gtf_df = pd.read_csv(gtf_file, sep='\t', skiprows=5, 
                             names=['Chromosome', 'havana_str', 'feature', 'Start', 'End', 'feat1', 'Strand', 'feat2', 'id_str'])
        gtf_df = gtf_df[gtf_df['feature'] == 'gene'].reset_index(drop=True)
        gtf_df['gene_id'] = gtf_df['id_str'].str.extract(r'gene_id "([^"]+)"')[0].str.split('.').str[0]
        gtf_df['Chromosome'] = 'chr' + gtf_df['Chromosome']
        gtf_df['Start'] -= 1
        gtf_df = gtf_df[['Chromosome', 'Start', 'End', 'gene_id', 'feat1', 'Strand']].drop_duplicates('gene_id')

        final_seqs_df = pd.merge(seqs_df, cleaned_seqs_df, on=['Chromosome', 'Start', 'End', 'species'])
        final_seqs_df['row_index'] = final_seqs_df.index

        seqs_pr = pr.PyRanges(final_seqs_df)
        gtf_pr = pr.PyRanges(gtf_df)
        seqs_gtf_pr = seqs_pr.join(gtf_pr)
        seqs_gtf_df = seqs_gtf_pr.df
        seqs_gtf_df = seqs_gtf_df[(seqs_gtf_df['Start_b'] >= seqs_gtf_df['Start'] + pad_bp) & 
                                  (seqs_gtf_df['End_b'] < seqs_gtf_df['End'] + pad_bp)]
        seqs_gtf_df['Start_rel'] = seqs_gtf_df['Start_b'] - seqs_gtf_df['Start']
        seqs_gtf_df['End_rel'] = seqs_gtf_df['End_b'] - seqs_gtf_df['Start']
        seqs_gtf_df['gene_len'] = seqs_gtf_df['End_rel'] - seqs_gtf_df['Start_rel']
        
        # Define your gene list
        # gene_list = ["YJR094W-A"]
        # plot_start = 204
        # plot_end = 500
        # plot_end = 312
        
        gene_list = ["YDR510W"]
        plot_start = 204
        plot_end = 500
        
        for gene in gene_list:
            gene_df = seqs_gtf_df[seqs_gtf_df['gene_id'] == gene]
            print("gene:", gene)
            print("gene_df:", gene_df)
            if gene_df.empty:
                continue

            # Prepare to accumulate average for the gene
            x_true_upstream_accum = np.zeros((pad_bp, 4))
            x_true_coding_accum = np.zeros((coding_viz_len, 4))
            x_true_downstream_accum = np.zeros((pad_bp, 4))
            x_pred_upstream_accum = np.zeros((pad_bp, 4))
            x_pred_coding_accum = np.zeros((coding_viz_len, 4))
            x_pred_downstream_accum = np.zeros((pad_bp, 4))
            window_16k_num = 0

            # Create a subdirectory for individual sequence plots
            indiv_dir = os.path.join(output_dir, gene + "_individual")
            os.makedirs(indiv_dir, exist_ok=True)

            # Process each window (i.e. each sequence) for this gene
            for _, row in gene_df.iterrows():
                chrom = row['Chromosome']
                seq_start = row['Start']
                seq_end = row['End']
                start = row['Start_b']
                end = row['End_b']
                gene_seq = fasta_open.fetch(chrom, start, end)
                gene_id = row['gene_id']
                gene_len = row['gene_len']
                start_rel = row['Start_rel']
                end_rel = row['End_rel']
                row_index = row['row_index']
                strand = row['Strand_b']

                seq_out_dir = os.path.join(output_dir, f"{gene_id}_{chrom}_{start}-{end}")
                os.makedirs(seq_out_dir, exist_ok=True)

                x_true_gene = x_true[row_index]
                x_pred_gene = x_pred[row_index]
                window_reversed = check_window_is_reversed(x_true_gene, chrom, seq_start, seq_end, fasta_open)
                if (window_reversed and strand == "+") or (not window_reversed and strand == "-"):
                    x_true_gene, x_pred_gene = reverse_complement(x_true_gene, x_pred_gene)
                    print("** Reversed gene sequence for row", row_index)
                if strand == "-":
                    start_rel, end_rel = 16384 - end_rel, 16384 - start_rel
                if not (0 <= start_rel <= 16384 and 0 <= end_rel <= 16384):
                    continue

                # Extract the upstream segment for this sequence
                start_idx_upstream = max(start_rel - pad_bp, 0)
                upstream_seq = x_true_gene[start_idx_upstream:start_rel]
                if upstream_seq.shape[0] < pad_bp:
                    pad_len = pad_bp - upstream_seq.shape[0]
                    upstream_seq = np.pad(upstream_seq, ((pad_len, 0), (0, 0)), mode='constant')
                upstream_seq_pred = x_pred_gene[start_idx_upstream:start_rel]
                if upstream_seq_pred.shape[0] < pad_bp:
                    pad_len = pad_bp - upstream_seq_pred.shape[0]
                    upstream_seq_pred = np.pad(upstream_seq_pred, ((pad_len, 0), (0, 0)), mode='constant')

                # Plot individual conservation-based DNA logos for this sequence (region 98-208)
                plot_dna_logo(
                    upstream_seq, sequence_template="$" * upstream_seq.shape[0],
                    figsize=(40, 2), plot_start=plot_start, plot_end=plot_end,
                    save_figs=True, fig_name=os.path.join(indiv_dir, f"row{row_index}_x_true")
                )
                plot_dna_logo(
                    upstream_seq_pred, sequence_template="$" * upstream_seq_pred.shape[0],
                    figsize=(40, 2), plot_start=plot_start, plot_end=plot_end,
                    save_figs=True, fig_name=os.path.join(indiv_dir, f"row{row_index}_x_pred")
                )

                # Accumulate for gene-average
                x_true_upstream_accum += upstream_seq
                x_pred_upstream_accum += upstream_seq_pred
                window_16k_num += 1

            # Compute gene-level averages
            x_true_upstream_avg = x_true_upstream_accum / window_16k_num
            x_pred_upstream_avg = x_pred_upstream_accum / window_16k_num

            # Plot gene-average PWM using modisco visualization
            x_true_upstream_avg_norm = norm_pwm(x_true_upstream_avg)
            x_pred_upstream_avg_norm = norm_pwm(x_pred_upstream_avg)
            plt.figure(figsize=(25, 1))
            viz_sequence.plot_weights(x_true_upstream_avg_norm, subticks_frequency=10)
            plt.title(f"Average PWM (x_true) for {model_arch} - gene {gene}")
            plt.savefig(os.path.join(seq_out_dir, "upstream_avg_pwm_x_true_avg.png"), transparent=False, dpi=300)
            plt.clf()
            plt.figure(figsize=(25, 1))
            viz_sequence.plot_weights(x_pred_upstream_avg_norm, subticks_frequency=10)
            plt.title(f"Average PWM (x_pred) for {model_arch} - gene {gene}")
            plt.savefig(os.path.join(seq_out_dir, "upstream_avg_pwm_x_pred_avg.png"), transparent=False, dpi=300)
            plt.clf()

            # Plot conservation-based DNA logos for the gene-average
            plot_dna_logo(
                x_true_upstream_avg, sequence_template="$" * x_true_upstream_avg.shape[0],
                figsize=(40, 2), plot_start=plot_start, plot_end=plot_end,
                save_figs=True, fig_name=os.path.join(seq_out_dir, "upstream_avg_pwm_x_true")
            )
            plot_dna_logo(
                x_pred_upstream_avg, sequence_template="$" * x_pred_upstream_avg.shape[0],
                figsize=(40, 2), plot_start=plot_start, plot_end=plot_end,
                save_figs=True, fig_name=os.path.join(seq_out_dir, "upstream_avg_pwm_x_pred")
            )

            # Save gene-average PWM for overall across model architectures later
            overall_avg_x_true.setdefault(gene, []).append(x_true_upstream_avg)
            overall_avg_x_pred.setdefault(gene, []).append(x_pred_upstream_avg)
        # End of gene loop for this model_arch
    # End of model_arch loop

    # Now compute overall average across all model architectures for each gene and plot
    for gene in overall_avg_x_true.keys():
        overall_true = np.mean(np.array(overall_avg_x_true[gene]), axis=0)
        overall_pred = np.mean(np.array(overall_avg_x_pred[gene]), axis=0)
        gene_out_dir = os.path.join(f'{dataset}_viz_seq_new_norm', "overall", gene)
        os.makedirs(gene_out_dir, exist_ok=True)

        print(f"overall_true shape: {overall_true.shape}")
        print(f"overall_pred shape: {overall_pred.shape}")

        overall_true_norm = norm_pwm(overall_true)
        overall_pred_norm = norm_pwm(overall_pred)

        plt.figure(figsize=(25, 1))
        viz_sequence.plot_weights(overall_true_norm, subticks_frequency=10)
        plt.title(f"Overall Average PWM (x_true) for gene {gene}")
        plt.savefig(os.path.join(gene_out_dir, "overall_upstream_avg_pwm_x_true_avg.png"), transparent=False, dpi=300)
        plt.clf()

        plt.figure(figsize=(25, 1))
        viz_sequence.plot_weights(overall_pred_norm, subticks_frequency=10)
        plt.title(f"Overall Average PWM (x_pred) for gene {gene}")
        plt.savefig(os.path.join(gene_out_dir, "overall_upstream_avg_pwm_x_pred_avg.png"), transparent=False, dpi=300)
        plt.clf()

        plot_dna_logo(
            overall_true, sequence_template="$" * overall_true.shape[0],
            figsize=(40, 2), plot_start=plot_start, plot_end=plot_end,
            save_figs=True, fig_name=os.path.join(gene_out_dir, "overall_upstream_avg_pwm_x_true")
        )
        plot_dna_logo(
            overall_pred, sequence_template="$" * overall_pred.shape[0],
            figsize=(40, 2), plot_start=plot_start, plot_end=plot_end,
            save_figs=True, fig_name=os.path.join(gene_out_dir, "overall_upstream_avg_pwm_x_pred")
        )

if __name__ == "__main__":
    main()
