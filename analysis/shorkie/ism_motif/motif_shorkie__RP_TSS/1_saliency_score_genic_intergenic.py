#!/usr/bin/env python3
"""
compare_genic_intergenic.py

Extends your saliency‐scores pipeline to:
  1) Compute per‐segment summary stats
  2) Box/violin/KDE/CDF plots of genic vs intergenic
  3) Mann–Whitney U, KS tests, Cohen's d
  4) Logistic‐regression ROC AUC (5‐fold CV)
  5) Average binned profile plots
"""
import os
from optparse import OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pyranges as pr
import pysam
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from yeast_helpers import make_seq_1hot

def parse_args():
    usage = "usage: %prog [options]"
    p = OptionParser(usage)
    p.add_option("--target_exp",    dest="target_exp",    default="RP_TSS",
                 help="Target experiment [Default: %default]")
    p.add_option("--gtf_file",      dest="gtf_file",
                 help="GTF annotation file")
    p.add_option("--fasta_file",    dest="fasta_file",
                 help="FASTA file for one-hot encoding")
    p.add_option("--score_name",    dest="score_name",    default="logSED",
                 help="Dataset name in HDF5 file [Default: %default]")
    p.add_option("--output_base_csv", dest="output_base_csv",
                 default="per_base_scores.csv",
                 help="Intermediate per-base CSV [Default: %default]")
    p.add_option("--output_dir",    dest="output_dir",
                 default="viz_saliency_scores",
                 help="Directory to save outputs [Default: %default]")
    opts, _ = p.parse_args()
    return opts

def load_annotation(gtf_file):
    pr_gtf = pr.read_gtf(gtf_file)
    feature_col = 'Feature' if 'Feature' in pr_gtf.df else 'feature'
    return pr_gtf[pr_gtf.df[feature_col] == 'gene']

def split_window_regions(bed_df, genes_pr):
    df = bed_df.copy()
    df['window_id'] = df.index
    win_pr = pr.PyRanges(pd.DataFrame({
        'Chromosome': df['Chromosome'],
        'Start':      df['Start'],
        'End':        df['End'],
        'window_id':  df['window_id'],
    }))
    genic = win_pr.intersect(genes_pr).df
    genic['is_genic'] = True
    inter = win_pr.subtract(genes_pr).df
    inter['is_genic'] = False

    segments = pd.concat([genic, inter], ignore_index=True)
    segments = segments.rename(columns={'Start':'seg_start','End':'seg_end'})
    # preserve original window bounds
    meta = df[['window_id','Start','End']].rename(
        columns={'Start':'win_start','End':'win_end'}
    )
    segments = segments.merge(meta, on='window_id')
    # unique segment identifier
    segments['segment_id'] = (
        segments['window_id'].astype(str) + "_"
        + segments['seg_start'].astype(str) + "_"
        + segments['seg_end'].astype(str)
    )
    return segments[['Chromosome','win_start','win_end',
                     'seg_start','seg_end','is_genic','window_id',
                     'segment_id']]

def map_sequences_to_indices(h5f):
    chr_arr = [c.decode('utf-8') for c in h5f['chr'][:]]
    starts = h5f['start'][:]
    ends   = h5f['end'][:]
    return pd.DataFrame({
        'Chromosome': chr_arr,
        'win_start':  starts,
        'win_end':    ends,
        'seq_idx':    np.arange(len(chr_arr))
    })

def normalize_pwm(seg_data, normalize=True, abs_val=True):
    if abs_val:
        seg_data = np.abs(seg_data)
    pwm = seg_data.mean(axis=2)
    if normalize:
        pwm = pwm - pwm.mean(axis=1, keepdims=True)
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

    # locate files

    bed_dir = os.path.join(f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/gene_exp_ism_window/{opts.target_exp}_chunk/")
    # list all bed files in the directory
    score_dir = os.path.join(f"/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/motif_shorkie_RP_TSS/gene_exp_motif_test_{opts.target_exp}/f0c0/")


    bed_files = sorted([os.path.join(bed_dir,f)
                        for f in os.listdir(bed_dir) if f.endswith('.bed')])
    score_files = [
        os.path.join(score_dir,f"part{i}/scores.h5")
        for i in range(len(bed_files))
    ]
    
    # load annotation + genome
    genes_pr = load_annotation(opts.gtf_file)
    fasta    = pysam.Fastafile(opts.fasta_file)

    # collect per-base records, including segment_id
    base_records = []
    for score_f, bed_f in zip(score_files, bed_files):
        bed_cols = ['chr','start','end','name','dot','strand']
        bed = pd.read_csv(bed_f, sep='\t', header=None,
                          names=bed_cols)
        bed['Chromosome'] = bed['chr'].str.replace('^chr','',regex=True)
        bed = bed.rename(columns={'start':'Start','end':'End'})
        segments = split_window_regions(bed, genes_pr)
        print(f"Processing {len(segments)} segments from {bed_f}")
        # print("segments: ", segments.head())
        

        with h5py.File(score_f,'r') as h5f:
            idx_df = map_sequences_to_indices(h5f)
            segs = segments.merge(
                idx_df,
                on=['Chromosome','win_start','win_end'],
                how='left'
            )
            for _,r in segs.dropna(subset=['window_id']).iterrows():
                data     = h5f[opts.score_name][int(r.window_id)]
                so,eo    = int(r.seg_start-r.win_start), int(r.seg_end-r.win_start)
                seg_data = data[so:eo]               # (L,4,N_tracks)
                pwm_norm = normalize_pwm(seg_data, normalize=False)
                L        = eo - so
                seq_1hot = make_seq_1hot(
                    fasta, "chr"+r.Chromosome,
                    r.seg_start, r.seg_end, L
                ).astype(np.float32)

                ref_vec  = reference_scores(pwm_norm,seq_1hot)
                avg_vec  = average_scores(pwm_norm)
                cons_vec = conservation_scores(pwm_norm)
                positions = np.arange(r.seg_start, r.seg_end)

                dfb = pd.DataFrame({
                    'segment_id': r.segment_id,
                    'position':   positions,
                    'ref_score':  ref_vec,
                    'avg_score':  avg_vec,
                    'cons_score': cons_vec,
                    'is_genic':   r.is_genic
                })
                base_records.append(dfb)

    base_df = pd.concat(base_records, ignore_index=True)
    base_df.to_csv(opts.output_base_csv, index=False)
    print(f"Wrote {len(base_df)} rows to {opts.output_base_csv}")

    # 1) Compute per-segment summary stats
    seg_stats = (
        base_df
        .groupby(['segment_id','is_genic'])
        .agg(
            mean_ref=('ref_score','mean'),
            med_ref =('ref_score','median'),
            std_ref =('ref_score','std'),
            max_ref =('ref_score','max'),
            mean_avg=('avg_score','mean'),
            std_avg =('avg_score','std'),
            mean_cons=('cons_score','mean'),
            std_cons =('cons_score','std'),
            length=('position','size')
        )
        .reset_index()
    )
    seg_stats.to_csv(
        os.path.join(opts.output_dir,'segment_summary_stats.csv'),
        index=False
    )

    # 2) Box/violin plots + KDE + CDF plots
    import seaborn as sns  # for violin & KDE convenience
    for metric in ['mean_ref','mean_avg','mean_cons']:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='is_genic',y=metric,data=seg_stats)
        plt.xticks([0,1],['intergenic','genic'])
        plt.title(f"Boxplot of {metric}")
        plt.savefig(os.path.join(opts.output_dir,f"{metric}_box.png"))
        plt.close()

        plt.figure(figsize=(6,4))
        sns.kdeplot(data=seg_stats, x=metric, hue='is_genic',
                    common_norm=False)
        plt.title(f"KDE of {metric}")
        plt.savefig(os.path.join(opts.output_dir,f"{metric}_kde.png"))
        plt.close()

        # CDF
        plt.figure(figsize=(6,4))
        for val,lab in zip([False,True],['intergenic','genic']):
            sorted_v = np.sort(seg_stats[seg_stats.is_genic==val][metric])
            cdf      = np.linspace(0,1,len(sorted_v))
            plt.plot(sorted_v,cdf,label=lab)
        plt.legend(); plt.xlabel(metric)
        plt.ylabel('CDF'); plt.title(f"CDF of {metric}")
        plt.savefig(os.path.join(opts.output_dir,f"{metric}_cdf.png"))
        plt.close()

    # 3) Statistical tests & Cohen's d
    def cohens_d(a,b):
        na,nb = len(a),len(b)
        da = a.mean(); db = b.mean()
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        pooled = np.sqrt(((na-1)*sa*sa + (nb-1)*sb*sb) / (na+nb-2))
        return (da - db) / pooled

    print("\nStatistical tests:")
    for metric in ['mean_ref','mean_avg','mean_cons']:
        gen = seg_stats[seg_stats.is_genic][metric]
        intg= seg_stats[~seg_stats.is_genic][metric]
        u,p_u = mannwhitneyu(gen,intg,alternative='two-sided')
        ks_res= ks_2samp(gen,intg)
        d     = cohens_d(gen,intg)
        print(f"{metric}: MWU p={p_u:.2e}, KS p={ks_res.pvalue:.2e}, Cohen's d={d:.2f}")

    # 4) Logistic‐regression ROC AUC
    X = seg_stats[['mean_ref','mean_avg','mean_cons']]
    y = seg_stats['is_genic'].astype(int)
    clf = LogisticRegression(solver='liblinear')
    auc = cross_val_score(clf, X, y, cv=5,
                          scoring='roc_auc').mean()
    print(f"\nLogReg 5-fold ROC AUC = {auc:.3f}")

    # 5) Average binned profile (ref_score only) across 100 bins
    n_bins = 100
    profs = {False: [], True: []}
    for sid, grp in base_df.groupby('segment_id'):
        vals = grp['ref_score'].values
        L    = len(vals)
        x_old= np.linspace(0,1,L)
        x_new= np.linspace(0,1,n_bins)
        interp = np.interp(x_new, x_old, vals)
        isg = grp['is_genic'].iat[0]
        profs[isg].append(interp)

    avg_prof = {k: np.mean(v,axis=0) for k,v in profs.items()}

    plt.figure(figsize=(6,4))
    for isg,lab,style in zip([True,False], ['genic','intergenic'], ['-','--']):
        plt.plot(
            np.linspace(0,1,n_bins),
            avg_prof[isg],
            linestyle=style,
            label=lab
        )
    plt.legend()
    plt.xlabel('Normalized position')
    plt.ylabel('Mean ref_score')
    plt.title('Average saliency profile (100 bins)')
    plt.tight_layout()
    plt.savefig(os.path.join(opts.output_dir,'binned_profile_ref_score.png'))
    plt.close()

    print(f"\nAll plots and stats saved under {opts.output_dir}")

if __name__ == "__main__":
    main()
