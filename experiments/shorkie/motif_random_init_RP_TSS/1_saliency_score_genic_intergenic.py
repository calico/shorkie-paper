#!/usr/bin/env python3
"""
Script to process multiple HDF5 score files and corresponding BED window files,
compute reference-channel normalized, average channel, and conservation scores per base
across all continuous genic/intergenic segments, then output a combined per-base CSV
and distribution plots comparing genic vs intergenic.
"""
from optparse import OptionParser
import os
import h5py
import numpy as np
import pandas as pd
import pyranges as pr
import pysam
from yeast_helpers import make_seq_1hot
import matplotlib.pyplot as plt


def parse_args():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option(
        "--target_exp", dest="target_exp", default="RP_TSS",
        help="Target experiment [Default: %default]"
    )    
    # parser.add_option(
    #     "--scores_h5", dest="scores_h5", help="Comma-separated HDF5 score files"
    # )
    # parser.add_option(
    #     "--bed_files", dest="bed_files", help="Comma-separated BED files matching HDF5 order"
    # )
    parser.add_option(
        "--gtf_file", dest="gtf_file", help="GTF annotation file"
    )
    parser.add_option(
        "--fasta_file", dest="fasta_file", help="FASTA file for one-hot encoding"
    )
    parser.add_option(
        "--score_name", dest="score_name", default="logSED",
        help="Dataset name in HDF5 file [Default: %default]"
    )
    parser.add_option(
        "--output_base_csv", dest="output_base_csv", default="per_base_scores.csv",
        help="Output CSV for per-base scores [Default: %default]"
    )
    parser.add_option(
        "--output_dir", dest="output_dir", default="viz_saliency_scores",
        help="Directory to save distribution plots [Default: %default]"
    )
    opts, _ = parser.parse_args()
    return opts


def load_annotation(gtf_file):
    pr_gtf = pr.read_gtf(gtf_file)
    feature_col = 'Feature' if 'Feature' in pr_gtf.df.columns else 'feature'
    return pr_gtf[pr_gtf.df[feature_col] == 'gene']


def split_window_regions(bed_df, genes_pr):
    df = bed_df.copy()
    df['window_id'] = df.index
    win_pr = pr.PyRanges(pd.DataFrame({
        'Chromosome': df['Chromosome'],
        'Start': df['Start'],
        'End': df['End'],
        'window_id': df['window_id']
    }))
    genic = win_pr.intersect(genes_pr).df
    genic['is_genic'] = True
    inter = win_pr.subtract(genes_pr).df
    inter['is_genic'] = False
    segments = pd.concat([
        genic[['Chromosome','Start','End','window_id','is_genic']],
        inter[['Chromosome','Start','End','window_id','is_genic']]
    ], ignore_index=True)
    meta = df[['window_id','Start','End']].rename(
        columns={'Start':'win_start','End':'win_end'}
    )
    segments = segments.merge(meta, on='window_id')
    segments = segments.rename(columns={'Start':'seg_start','End':'seg_end'})
    return segments


def map_sequences_to_indices(h5f):
    chr_arr = [c.decode('utf-8') for c in h5f['chr'][:]]
    starts = h5f['start'][:]
    ends = h5f['end'][:]
    return pd.DataFrame({
        'Chromosome': chr_arr,
        'win_start': starts,
        'win_end': ends,
        'seq_idx': np.arange(len(chr_arr))
    })


def normalize_pwm(seg_data, normalize=True, abs=True):
    # seg_data: (L,4,N_tracks) -> pwm: (L,4)
    # take absolute value of log-odds
    if abs:
        seg_data = np.abs(seg_data)

    # normalize by mean of each base
    pwm = seg_data.mean(axis=2)
    if normalize:
        mean_pwm = pwm.mean(axis=1, keepdims=True)
        pwm_norm = pwm - mean_pwm
        pwm = pwm_norm
    return pwm


def reference_scores(pwm_norm, seq_1hot):
    return (pwm_norm * seq_1hot).sum(axis=1)


def average_scores(pwm_norm):
    return pwm_norm.mean(axis=1)


def conservation_scores(pwm_norm):
    return pwm_norm.std(axis=1)


def main():
    opts = parse_args()
    os.makedirs(opts.output_dir, exist_ok=True)

    target_exp = opts.target_exp
    bed_file_dir = os.path.join(f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/{opts.target_exp}_chunk/")
    # list all bed files in the directory
    bed_files = [os.path.join(bed_file_dir, f) for f in os.listdir(bed_file_dir) if f.endswith('.bed')]

    score_file_dir = os.path.join(f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/motif_shorkie_RP_TSS/gene_exp_motif_test_{opts.target_exp}/f0c0/")
    # list all h5 files in the directory
    score_files =[]
    for num in range(len(bed_files)):
        score_file = score_file_dir + f"part{num}/scores.h5"
        score_files.append(score_file)   

    print("score_files: ", len(score_files))
    print("bed_files: ", len(bed_files))

    score_files = score_files[:10]
    bed_files = bed_files[:10]

    genes_pr = load_annotation(opts.gtf_file)
    fasta = pysam.Fastafile(opts.fasta_file)

    base_records = []
    # iterate over matching pairs
    for score_f, bed_f in zip(score_files, bed_files):
        bed_cols = ['chr','start','end','name','dot','strand']
        bed = pd.read_csv(bed_f, sep='\t', header=None, names=bed_cols)
        bed['Chromosome'] = bed['chr'].str.replace('^chr','', regex=True)
        bed = bed.rename(columns={'start':'Start','end':'End'})
        segments = split_window_regions(bed, genes_pr)

        with h5py.File(score_f, 'r') as h5f:
            idx_df = map_sequences_to_indices(h5f)
            segs = segments.merge(idx_df, on=['Chromosome','win_start','win_end'], how='left')
            for _, r in segs.dropna(subset=['window_id']).iterrows():
                data = h5f[opts.score_name][int(r.window_id)]  # full window
                # slice to segment
                start_off = int(r.seg_start) - int(r.win_start)
                end_off   = int(r.seg_end)   - int(r.win_start)
                seg_data  = data[start_off:end_off]            # (L,4,N_tracks)
                pwm_norm  = normalize_pwm(seg_data, normalize=False)            # (L,4)
                L = end_off - start_off
                seq_1hot  = make_seq_1hot(fasta, "chr"+r.Chromosome, r.seg_start, r.seg_end, L).astype(np.float32)
                ref_vec   = reference_scores(pwm_norm, seq_1hot)
                avg_vec   = average_scores(pwm_norm)
                cons_vec  = conservation_scores(pwm_norm)
                positions = np.arange(r.seg_start, r.seg_end)
                dfb = pd.DataFrame({
                    'position':   positions,
                    'ref_score':  ref_vec,
                    'avg_score':  avg_vec,
                    'cons_score': cons_vec,
                    'is_genic':   r.is_genic
                })
                base_records.append(dfb)

    # compile all bases
    base_df = pd.concat(base_records, ignore_index=True)
    base_df.to_csv(opts.output_base_csv, index=False)
    print(f"Wrote {len(base_df)} per-base records to {opts.output_base_csv}")

    # aggregate & print stats
    for metric in ['ref_score','avg_score','cons_score']:
        print(f"\n{metric} stats:")
        for label in [True, False]:
            grp = base_df[base_df.is_genic == label][metric]
            print(f" {'genic' if label else 'intergenic'}: mean={grp.mean():.4f}, std={grp.std():.4f}")

    # plot distributions
    for metric in ['ref_score','avg_score','cons_score']:
        plt.figure()
        for label, style in zip([True, False], ['-', '--']):
            data = base_df[base_df.is_genic == label][metric]
            name = 'genic' if label else 'intergenic'
            plt.hist(data, bins=50, density=True, alpha=0.6,
                     histtype='step', linestyle=style, label=name)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(opts.output_dir, f"{metric}_distribution.png"), dpi=300)
    print("Saved distribution plots to ", opts.output_dir)

if __name__ == '__main__':
    main()
