#!/usr/bin/env python3
"""
evaluate_gene_predictions_full.py

Usage:
    python evaluate_gene_predictions_full.py \
      --data_dirs    dir1 dir2 dir3 \
      --tf           MSN2 \
      --targets_file targets_with_descriptions.tsv \
      --out          gene_metrics_by_timepoint.tsv \
      --out_dir      results/ \
      [--gene        YAL001C]
"""
import argparse
import os

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
    print("targets_strand_df: ", targets_strand_df.shape)
    print("targets_strand_df: ", targets_strand_df.head())
    print("keep_cols: ", keep_cols)

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

    print("gene_targets_norm: ", gene_targets_norm.shape)
    print("gene_preds_norm:   ", gene_preds_norm.shape)

    # 7) Compute per-track Pearson & R² (raw vs. norm’d)
    pearsonr_, r2_, pearsonr_n, r2_n = [], [], [], []
    for i, col in enumerate(keep_cols):
        # raw
        yt = gene_targets.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        yp = gene_preds.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        valid = yt.notna() & yp.notna()
        if valid.sum() >= 2:
            y1, y2 = yt[valid], yp[valid]
            pearsonr_.append(pearsonr(y1, y2)[0])
            r2_.append(explained_variance_score(y1, y2))
        else:
            pearsonr_.append(np.nan)
            r2_.append(np.nan)
        # norm’d
        ytn = gene_targets_norm.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        ypn = gene_preds_norm.iloc[:, i].replace([np.inf, -np.inf], np.nan)
        valid = ytn.notna() & ypn.notna()
        if valid.sum() >= 2:
            y1, y2 = ytn[valid], ypn[valid]
            pearsonr_n.append(pearsonr(y1, y2)[0])
            r2_n.append(explained_variance_score(y1, y2))
        else:
            pearsonr_n.append(np.nan)
            r2_n.append(np.nan)

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
    metric_name_mapping = {
        'pearsonr':      "Pearson's R",
        'r2':            "R²",
        'pearsonr_norm': "Normalized Pearson's R",
        'r2_norm':       "Normalized R²"
    }
    for metric in ['pearsonr', 'r2', 'pearsonr_norm', 'r2_norm']:
        data, labels = [], []
        for t in sorted(acc_df['timepoint'].dropna().unique()):
            vals = acc_df.loc[acc_df['timepoint'] == t, metric].dropna().values
            if len(vals):
                data.append(vals)
                labels.append(t)
        plt.figure(); ax = plt.gca()
        plt.boxplot(data, labels=labels)
        ymin, ymax = ax.get_ylim()
        for i, vals in enumerate(data):
            ax.text(i+1, ymin + 0.03*(ymax-ymin), f'n={len(vals)}',
                    ha='center', va='top', fontsize=8)
        plt.xlabel('Timepoint')
        plt.ylabel(metric_name_mapping[metric])
        plt.title(f'Timepoint-Resolved {metric_name_mapping[metric]} for\nMeasured Genes in {args.tf} Induction Tracks')
        plt.tight_layout()
        fp = os.path.join(out_dir, f'{metric}_by_timepoint_boxplot.png')
        plt.savefig(fp, dpi=300)
        plt.close()
        print(f"→ saved boxplot: {fp}")

    # 11) Bar chart of raw (unlogged) scores by timepoint
    raw_t = raw_targets_sel if raw_targets_sel is not None else raw_targets_all
    raw_p = raw_preds_sel   if raw_preds_sel   is not None else raw_preds_all

    print("raw_t: ", raw_t.shape)   
    print("raw_p: ", raw_p.shape)

    tps = sorted(acc_df['timepoint'].dropna().unique())
    med_t, se_t, med_p, se_p = [], [], [], []
    for t in tps:
        ids     = acc_df.loc[acc_df['timepoint'] == t, 'identifier']
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
    subtitle = f'gene {gene_name}' if gene_name else args.tf
    plt.title(f'Gene Expression by timepoint for {subtitle}')
    plt.legend()
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

    # ----------------------------------------
    # 12) Fold‐change relative to T0 baseline
    # ----------------------------------------

    # 12.1 Identify T0 tracks and all timepoints
    t0_ids = acc_df.loc[acc_df['timepoint'] == 0, 'identifier'].tolist()
    tps   = sorted(acc_df['timepoint'].dropna().unique())

    # 12.2 Per‐gene fold-change bar chart (if --gene)
    if args.gene:
        gene = args.gene
        meas = raw_targets_sel.loc[gene]
        pred = raw_preds_sel.loc[gene]
        baseline_meas = meas[t0_ids].mean()
        baseline_pred = pred[t0_ids].mean()

        # compute FC
        fc_meas = meas / baseline_meas
        fc_pred = pred / baseline_pred

        # aggregate by timepoint
        means_m, sems_m = [], []
        means_p, sems_p = [], []
        for t in tps:
            ids_t = acc_df.loc[acc_df['timepoint'] == t, 'identifier']
            vals_m = fc_meas[ids_t].values
            vals_p = fc_pred[ids_t].values
            means_m.append(np.nanmean(vals_m))
            sems_m .append(sem  (vals_m, nan_policy='omit'))
            means_p.append(np.nanmean(vals_p))
            sems_p .append(sem  (vals_p, nan_policy='omit'))

        # plot
        x  = np.arange(len(tps)); w = 0.35
        plt.figure(); ax = plt.gca()
        ax.bar(x - w/2, means_m, w, yerr=sems_m, capsize=5, label='Measurement FC')
        ax.bar(x + w/2, means_p, w, yerr=sems_p, capsize=5, label='Prediction FC')
        ax.set_xticks(x); ax.set_xticklabels(tps)
        ax.set_xlabel('Timepoint'); ax.set_ylabel('Fold-change vs T0')
        ax.set_title(f'Fold-change by timepoint for gene {gene_name or gene}')
        ax.legend()

        # ---- new: adjust y-limits based on your data so bars are clearly visible ----
        all_lowers = [m - s for m, s in zip(means_m, sems_m)] + [m - s for m, s in zip(means_p, sems_p)]
        all_uppers = [m + s for m, s in zip(means_m, sems_m)] + [m + s for m, s in zip(means_p, sems_p)]
        ymin, ymax = min(all_lowers), max(all_uppers)
        offset = 0.1 * (ymax - ymin)  # use 10% padding
        ax.set_ylim(ymin - offset, ymax + offset)

        plt.tight_layout()
        fp_fc_bar = os.path.join(out_dir, 'fold_change_by_timepoint_bar.png')
        plt.savefig(fp_fc_bar, dpi=300)
        plt.close()
        print(f"→ saved fold-change bar chart: {fp_fc_bar}")

    # 12.3 Across-genes fold-change scatter & extra plots
    baseline_all_meas = raw_targets_all[t0_ids].mean(axis=1)
    baseline_all_pred = raw_preds_all  [t0_ids].mean(axis=1)

    for t in tps:
        if t == 0:
            continue

        ids_t   = acc_df.loc[acc_df['timepoint'] == t, 'identifier'].tolist()
        meas_t  = raw_targets_all[ids_t].mean(axis=1)
        pred_t  = raw_preds_all  [ids_t].mean(axis=1)
        fc_all_m = meas_t / baseline_all_meas
        fc_all_p = pred_t / baseline_all_pred

        # determine axis limits from Prediction FC
        pmin, pmax = fc_all_p.min(), fc_all_p.max()

        # filter to within [pmin, pmax] on both axes to remove outliers
        mask = (fc_all_m >= pmin) & (fc_all_m <= pmax) & (fc_all_p >= pmin) & (fc_all_p <= pmax)
        fc_all_m_filt = fc_all_m[mask]
        fc_all_p_filt = fc_all_p[mask]

        # scatter
        # plt.figure()
        # plt.scatter(fc_all_m_filt, fc_all_p_filt, s=5, alpha=0.4)
        # # shared x and y limits
        # plt.xlim(pmin, pmax)
        # plt.ylim(pmin, pmax)
        # # identity line
        # plt.plot([pmin, pmax], [pmin, pmax], 'k--')
        # plt.xlabel('Measurement FC'); plt.ylabel('Prediction FC')
        # plt.title(f'Fold-change scatter at timepoint T{t}')
        # plt.tight_layout()
        # fn = os.path.join(out_dir, f'fold_change_scatter_T{t}.png')
        # plt.savefig(fn, dpi=300)
        # plt.close()
        # print(f"→ saved scatter: {fn}")
        # --- updated: square scatter with Pearson's r & trend line, no diagonal ---
        # calculate Pearson's r
        r_val, p_val = pearsonr(fc_all_m_filt, fc_all_p_filt)

        # fit a simple linear trend line (y = slope * x + intercept)
        slope, intercept = np.polyfit(fc_all_m_filt, fc_all_p_filt, 1)
        x_line = np.array([pmin, pmax])
        y_line = slope * x_line + intercept

        # create figure & axes
        fig, ax = plt.subplots()
        # scatter points
        ax.scatter(fc_all_m_filt, fc_all_p_filt, s=5, alpha=0.4,
                   label=f'ρ = {r_val:.2f}')
        # plot trend line
        ax.plot(x_line, y_line, linestyle='-', 
                label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
        
        # set square aspect ratio & shared limits
        ax.set_xlim(pmin, pmax)
        ax.set_ylim(pmin, pmax)
        ax.set_aspect('equal', 'box')

        # labels & title
        ax.set_xlabel('Measurement FC')
        ax.set_ylabel('Prediction FC')
        ax.set_title(f'Fold-change scatter at timepoint T{t}')
        
        # legend shows ρ and fit equation
        ax.legend(fontsize=8)

        plt.tight_layout()
        fn = os.path.join(out_dir, f'fold_change_scatter_T{t}.png')
        plt.savefig(fn, dpi=300)
        plt.close()
        print(f"→ saved scatter with trend line: {fn}")


        # --- 2.2 Hexbin density plot ---
        plt.figure()
        hb = plt.hexbin(fc_all_m, fc_all_p, gridsize=50, mincnt=1)
        plt.colorbar(hb, label='count')
        plt.plot([0, fc_all_m.max()], [0, fc_all_m.max()], 'k--')
        plt.xlabel('Measurement FC'); plt.ylabel('Prediction FC')
        plt.title(f'Fold‐change density (hexbin) T{t}')
        plt.tight_layout()
        fn = os.path.join(out_dir, f'fold_change_hexbin_T{t}.png')
        plt.savefig(fn, dpi=300); plt.close()
        print(f"→ saved hexbin: {fn}")

        # --- 2.3 Bland–Altman (difference vs mean) ---
        diff    = fc_all_p - fc_all_m
        mean_fc = 0.5 * (fc_all_p + fc_all_m)
        md      = np.nanmean(diff)
        sd      = np.nanstd (diff)
        plt.figure()
        plt.scatter(mean_fc, diff, s=5, alpha=0.4)
        plt.axhline(md,      linestyle='--', label='Mean diff')
        plt.axhline(md+1.96*sd, linestyle=':', label='+1.96 SD')
        plt.axhline(md-1.96*sd, linestyle=':', label='-1.96 SD')
        plt.xlabel('Mean FC'); plt.ylabel('Difference (Pred − Meas)')
        plt.title(f'Bland–Altman plot T{t}')
        plt.legend(fontsize=8)
        plt.tight_layout()
        fn = os.path.join(out_dir, f'fold_change_blandaltman_T{t}.png')
        plt.savefig(fn, dpi=300); plt.close()
        print(f"→ saved Bland–Altman: {fn}")



if __name__ == '__main__':
    main()