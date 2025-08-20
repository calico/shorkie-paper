#!/usr/bin/env python3
"""
evaluate_gene_predictions_full.py

1) Read & merge gene-level tables from three data directories
2) Compute per-timepoint, per-gene Pearson & R² between true vs. pred
3) Plug in your bin-level normalization arrays and compute the six accuracy stats.
4) Parse "description" into numerical timepoint
5) Draw boxplots of each metric by timepoint
6) Filter to only those tracks whose description contains the --tf substring
7) Optionally restrict raw‐value bar chart & boxplot to a single gene via --gene
8) Plot raw (unlogged) target vs. pred medians with standard error by timepoint
9) Plot raw (unlogged) target values as boxplots: all genes vs. selected gene

Usage:
    python evaluate_gene_predictions_full.py \
      --data_dirs    dir1 dir2 dir3 \
      --tf           MSN2 \
      --targets_file targets_with_descriptions.tsv \
      --out          gene_metrics_by_timepoint.tsv \
      --out_dir      results/ \
      [--gene        YAL001C]
"""
import argparse, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, sem
from sklearn.metrics import explained_variance_score

from baskerville import dataset
from qnorm import quantile_normalize


def parse_group(desc: str) -> str:
    dl = desc.lower()
    if "pos_logfe" in dl or "chip-exo" in dl:
        return "ChIP-exo"
    elif "chip-mnase" in dl:
        return "ChIP-MNase"
    elif "1000 strains rnaseq" in dl:
        return "1000-RNA-seq"
    elif "rnaseq" in dl or "rna_seq" in dl:
        return "RNA-Seq"
    else:
        return "Other"


def load_df(path):
    return pd.read_csv(path, sep='\t', index_col=0)


# Add this function to calculate log fold change
def log_fold_change(target, pred, timepoint, baseline_timepoint=0):
    """Calculates log2 fold change relative to baseline_timepoint (e.g., T0)"""
    target_fold_change = np.log2(target.loc[timepoint] / target.loc[baseline_timepoint] + 1)
    pred_fold_change = np.log2(pred.loc[timepoint] / pred.loc[baseline_timepoint] + 1)
    return target_fold_change, pred_fold_change


# Add this to the main function to calculate the log fold changes and plot scatter plot
def plot_log_fold_change_scatter(targets_df, gene_targets, gene_preds, out_dir, timepoints=[5, 10, 15, 30]):
    # Prepare a DataFrame to store fold change comparisons
    fold_change_data = {'timepoint': [], 'target_fold_change': [], 'pred_fold_change': []}

    for timepoint in timepoints:
        # Calculate log fold change between target and prediction for each timepoint
        for i, col in enumerate(gene_targets.columns):
            if timepoint in gene_targets.index and timepoint in gene_preds.index:
                target_fc, pred_fc = log_fold_change(gene_targets[col], gene_preds[col], timepoint)
                fold_change_data['timepoint'].append(timepoint)
                fold_change_data['target_fold_change'].append(target_fc)
                fold_change_data['pred_fold_change'].append(pred_fc)

    # Convert fold change data to DataFrame
    fold_change_df = pd.DataFrame(fold_change_data)

    # Scatter plot: Predicted vs Measured log fold change at each timepoint
    plt.figure()
    plt.scatter(fold_change_df['target_fold_change'], fold_change_df['pred_fold_change'], alpha=0.6)
    plt.plot([-3, 3], [-3, 3], color='red', linestyle='--')  # Identity line for comparison
    plt.xlabel('Measured log Fold Change (Tn - T0)')
    plt.ylabel('Predicted log Fold Change (Tn - T0)')
    plt.title(f'Scatter Plot of Predicted vs Measured Log Fold Change')

    # Save the scatter plot
    scatter_fp = os.path.join(out_dir, 'log_fold_change_scatter.png')
    plt.tight_layout()
    plt.savefig(scatter_fp, dpi=300)
    plt.close()
    print(f"→ saved log fold change scatter plot: {scatter_fp}")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data_dirs', nargs=1, required=True,
                   help='three directories each containing gene_targets_norm.tsv, gene_preds_norm.tsv, gene_var.tsv, gene_within.tsv')
    p.add_argument('--tf',           required=True,
                   help='substring to filter descriptions (e.g. "T30_")')
    p.add_argument('--targets_file', required=True,
                   help='TSV of tracks with description column')
    p.add_argument('--out',          required=True,
                   help='output TSV of all metrics by track')
    p.add_argument('--out_dir',      required=True,
                   help='where to save boxplots, bar charts, and metrics')
    p.add_argument('--gene',         default=None,
                   help='(optional) gene ID to restrict raw-value charts/plots')
    args = p.parse_args()
    
    gene_name = ""
    if args.gene == "YCL040W":
        gene_name = "GLK1"
    elif args.gene == "YBR139W":
        gene_name = "ATG42"
    elif args.gene == "YML100W":
        gene_name = "TSL1"
    elif args.gene == "YIL124W":
        gene_name = "AYR1"

    # set up output directory (subdir per gene if specified)
    out_dir = os.path.join(args.out_dir, f"{args.gene}_{gene_name}") if args.gene else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load & merge across the three data_dirs
    tgt_list, pred_list, var_list, within_list = [], [], [], []
    for d in args.data_dirs:
        tgt_list.append(load_df(os.path.join(d, 'gene_targets_norm.tsv')))
        pred_list.append(load_df(os.path.join(d, 'gene_preds_norm.tsv')))
        var_list.append(load_df(os.path.join(d, 'gene_var.tsv')))
        within_list.append(load_df(os.path.join(d, 'gene_within.tsv')))

    gene_targets = pd.concat(tgt_list, axis=1)
    gene_preds   = pd.concat(pred_list, axis=1)
    gene_var     = pd.concat(var_list, axis=1)
    gene_within  = pd.concat(within_list, axis=1)

    print("gene_targets: ", gene_targets.shape)
    print("gene_preds:   ", gene_preds.shape)
    print("gene_var:     ", gene_var.shape)
    print("gene_within:  ", gene_within.shape)

    # 2) Track metadata
    targets_df = pd.read_csv(args.targets_file, sep='\t', index_col=0)
    targets_df['group'] = targets_df['description'].apply(parse_group)
    targets_strand_df = dataset.targets_prep_strand(targets_df)

    # 3) FILTER: only tracks matching --tf
    mask = targets_strand_df['description'].str.contains(args.tf)
    targets_strand_df = targets_strand_df.loc[mask]
    keep_cols = targets_strand_df['identifier'].tolist()

    # safety-check & subset
    for df in (gene_targets, gene_preds, gene_var, gene_within):
        missing = set(keep_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {df}: {missing}")

    gene_targets = gene_targets[keep_cols]
    gene_preds   = gene_preds[keep_cols]
    gene_var     = gene_var[keep_cols]
    gene_within  = gene_within[keep_cols]
    print(f"→ {len(keep_cols)} tracks match '{args.tf}'")

    # 4) Capture raw (unlogged) values
    raw_targets_all = gene_targets.copy()
    raw_preds_all   = gene_preds.copy()

    raw_targets_sel = None
    raw_preds_sel   = None
    if args.gene:
        if args.gene not in raw_targets_all.index:
            raise ValueError(f"Gene '{args.gene}' not found (rows start: {list(raw_targets_all.index)[:5]}...)")
        raw_targets_sel = raw_targets_all.loc[[args.gene]]
        raw_preds_sel   = raw_preds_all.loc[[args.gene]]
        print(f"→ Restricted raw-value charts to gene '{args.gene}'")

    # 5) log2+1 transform for metrics
    gene_targets = np.log2(gene_targets + 1)
    gene_preds   = np.log2(gene_preds   + 1)

    # 6) Quantile-normalize + mean-center
    qt = quantile_normalize(gene_targets.values, ncpus=2)
    qp = quantile_normalize(gene_preds.values,   ncpus=2)
    gene_targets_norm = (
        pd.DataFrame(qt, index=gene_targets.index, columns=gene_targets.columns)
          .subtract(pd.Series(qt.mean(axis=1), index=gene_targets.index), axis=0)
    )
    gene_preds_norm = (
        pd.DataFrame(qp, index=gene_preds.index, columns=gene_preds.columns)
          .subtract(pd.Series(qp.mean(axis=1),   index=gene_preds.index),   axis=0)
    )

    # 7) Compute per-track Pearson & R² (raw vs. norm’d)
    pearsonr_, r2_, pearsonr_n, r2_n = [], [], [], []
    # for i in range(len(keep_cols)):
    #     yt, yp = gene_targets.iloc[:, i], gene_preds.iloc[:, i]
    #     pearsonr_.append(pearsonr(yt, yp)[0])
    #     r2_.append(explained_variance_score(yt, yp))
    #     ytn, ypn = gene_targets_norm.iloc[:, i], gene_preds_norm.iloc[:, i]
    #     pearsonr_n.append(pearsonr(ytn, ypn)[0])
    #     r2_n.append(explained_variance_score(ytn, ypn))


    for i, col in enumerate(keep_cols):
        yt = gene_targets.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        yp = gene_preds.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        # keep only positions where *both* measurements exist
        valid = yt.notna() & yp.notna()
        if valid.sum() >= 2:
            y1, y2 = yt[valid], yp[valid]
            pearsonr_.append(pearsonr(y1, y2)[0])
            r2_.append(explained_variance_score(y1, y2))
        else:
            # too few points to compute a meaningful correlation
            pearsonr_.append(np.nan)
            r2_.append(np.nan)
        
        # ytn, ypn = gene_targets_norm.iloc[:, i], gene_preds_norm.iloc[:, i]
        # pearsonr_n.append(pearsonr(ytn, ypn)[0])
        # r2_n.append(explained_variance_score(ytn, ypn))

        ytn = gene_targets_norm.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        ypn = gene_preds_norm.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        # keep only positions where *both* measurements exist
        valid = ytn.notna() & ypn.notna()
        if valid.sum() >= 2:
            y1, y2 = ytn[valid], ypn[valid]
            pearsonr_n.append(pearsonr(y1, y2)[0])
            r2_n.append(explained_variance_score(y1, y2))
        else:
            # too few points to compute a meaningful correlation
            pearsonr_n.append(np.nan)
            r2_n.append(np.nan)


        # ytn, ypn = gene_targets_norm.iloc[:, i].replace([np.inf, -np.inf], np.nan), \
        #           gene_preds_norm.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        # # keep only positions where *both* measurements exist
        # valid = ytn.notna() & ypn.notna()
        # if valid.sum() >= 2:
        #     y1, y2 = ytn[valid], ypn[valid]
        #     pearsonr_n.append(pearsonr(y1, y2)[0])
        #     r2_n.append(explained_variance_score(y1, y2))
        # else:
        #     # too few points to compute a meaningful correlation
        #     pearsonr_n.append(np.nan)
        #     r2_n.append(np.nan)


    # 8) Write metrics table
    acc_df = pd.DataFrame({
        'identifier':    targets_strand_df.identifier,
        'description':   targets_strand_df.description,
        'group':         targets_strand_df.group,
        'pearsonr':      pearsonr_,
        'r2':            r2_,
        'pearsonr_norm': pearsonr_n,
        'r2_norm':       r2_n
    })

    metric_name_mapping = {
        'pearsonr':      'Pearson\'s R',
        'r2':            'R²',
        'pearsonr_norm': 'Normalized Pearson\'s R',
        'r2_norm':       'Normalized R²'
    }
    acc_df.to_csv(args.out, sep='\t', index=False)
    print(f"→ wrote metrics to {args.out}")

    # 9) Parse integer timepoint
    acc_df['timepoint'] = (
        acc_df['description']
          .str.extract(r'-T(\d+)', expand=False)
          .pipe(pd.to_numeric, errors='coerce')
          .astype('Int64')
    )

    # 10) Boxplots per metric by timepoint
    for metric in ['pearsonr', 'r2', 'pearsonr_norm', 'r2_norm']:
        data, labels = [], []
        for t in sorted(acc_df['timepoint'].dropna().unique()):
            vals = acc_df.loc[acc_df['timepoint']==t, metric].dropna().values
            if len(vals):
                data.append(vals); labels.append(t)
        plt.figure(); ax = plt.gca()
        plt.boxplot(data, labels=labels)
        ymin, ymax = ax.get_ylim()
        for i, vals in enumerate(data):
            ax.text(i+1, ymin + 0.03*(ymax-ymin), f'n={len(vals)}',
                    ha='center', va='top', fontsize=8)
        plt.xlabel('Timepoint')
        plt.ylabel(metric_name_mapping[metric])
        plt.title(f'Timepoint-Resolved {metric_name_mapping[metric]} for \nMeasured Genes in {args.tf} Induction Tracks')
        plt.tight_layout()
        fp = os.path.join(out_dir, f'{metric}_by_timepoint_boxplot.png')
        plt.savefig(fp, dpi=300); plt.close()
        print(f"→ saved boxplot: {fp}")

    # 11) Bar chart of raw (unlogged) scores by timepoint
    raw_t = raw_targets_sel if raw_targets_sel is not None else raw_targets_all
    raw_p = raw_preds_sel   if raw_preds_sel   is not None else raw_preds_all

    tps = sorted(acc_df['timepoint'].dropna().unique())
    med_t, se_t, med_p, se_p = [], [], [], []
    for t in tps:
        ids     = acc_df.loc[acc_df['timepoint']==t, 'identifier']
        vals_t  = raw_t[ids].values.flatten()
        vals_p  = raw_p[ids].values.flatten()
        med_t.append(np.median(vals_t)); se_t.append(sem(vals_t))
        med_p.append(np.median(vals_p)); se_p.append(sem(vals_p))

    x = np.arange(len(tps)); w = 0.35
    plt.figure(); ax = plt.gca()
    plt.bar(x - w/2, med_t, w, yerr=se_t, capsize=5, label='Experiment measurements')
    plt.bar(x + w/2, med_p, w, yerr=se_p, capsize=5, label='Shorkie predictions')
    plt.xticks(x, tps)
    plt.xlabel('Timepoint')
    plt.ylabel('Gene Expression (Summed RPM)')
    subtitle = f'gene {gene_name}' 
    plt.title(f'Gene Expression by timepoint for {subtitle}')
    plt.legend()

    # set y-limits to min/max ±5%
    lowers = [m - s for m, s in zip(med_t, se_t)] + [m - s for m, s in zip(med_p, se_p)]
    uppers = [m + s for m, s in zip(med_t, se_t)] + [m + s for m, s in zip(med_p, se_p)]
    ymin, ymax = min(lowers), max(uppers)
    offset = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - offset, ymax + offset)

    plt.tight_layout()
    fp_bar = os.path.join(out_dir, 'raw_scores_by_timepoint_bar.png')
    plt.savefig(fp_bar, dpi=300)
    plt.close()
    print(f"→ saved raw scores bar chart: {fp_bar}")

    # After all calculations and metrics, call the new scatter plot function
    plot_log_fold_change_scatter(targets_df, gene_targets, gene_preds, out_dir)

if __name__ == '__main__':
    main()
