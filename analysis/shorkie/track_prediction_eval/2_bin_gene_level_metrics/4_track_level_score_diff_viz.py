#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optparse import OptionParser

###############################################################################
#                          DATA LOADING + PROCESSING                          #
###############################################################################

def load_and_merge_data(options, model_arch, data_types, exp_model_type, n_folds=8):
    """
    Loads, merges, and computes the mean per identifier across folds for each data_type.
    Returns a dictionary: {data_type: DataFrame}.
    The DataFrame has columns like <metric>_self, <metric>_sup for each metric.
    """
    result = {}
    for data_type in data_types:
        all_folds = []
        for idx in range(n_folds):
            sup_path = (f"{options.root_dir}/seq_experiment/"
                        f"{exp_model_type}/16bp/supervised_{model_arch}_variants/learning_rate_0.0005/"
                        f"gene_level_eval_rc/f{idx}c0/{data_type}/acc.txt")

            self_path = (f"{options.root_dir}/seq_experiment/"
                         f"{exp_model_type}/16bp/self_supervised_{model_arch}/"
                         f"gene_level_eval_rc/f{idx}c0/{data_type}/acc.txt")

            df_sup = pd.read_csv(sup_path, sep="\t")
            df_self = pd.read_csv(self_path, sep="\t")

            merged = pd.merge(
                df_self, df_sup,
                on=["identifier", "description", "group"],
                suffixes=("_self", "_sup")
            )
            all_folds.append(merged)

        df_combined = pd.concat(all_folds)
        mean_df = df_combined.groupby(["identifier", "description", "group"])\
                               .mean().reset_index()
        result[data_type] = mean_df
    return result


def add_diff_and_mean_cols(df, metrics):
    """
    For each metric in [pearsonr, r2, etc.], creates:
        <metric>_diff = <metric>_self - <metric>_sup
        <metric>_mean = 0.5*(<metric>_self + <metric>_sup)
    """
    for m in metrics:
        df[f"{m}_diff"] = df[f"{m}_self"] - df[f"{m}_sup"]
        df[f"{m}_mean"] = 0.5 * (df[f"{m}_self"] + df[f"{m}_sup"])
    return df


def format_metric_name(metric):
    """
    Convert e.g. 'pearsonr_norm' -> 'Pearsonr Norm', 'r2' -> 'R2'.
    """
    if metric.lower() == "r2":
        return "R2"
    return metric.replace("_", " ").title()


def format_data_type(dt):
    """
    Optionally format data_type string.
    """
    return dt

###############################################################################
#                          PLOTTING FUNCTIONS (1x4)                          #
###############################################################################

def plot_paired_beeswarm_1x4(data_dict, data_types, metric, outdir):
    """
    One row of 4 plots for 'Paired Beeswarm' across the data types.
    X=0 for Shorkie_Random_Init, X=1 for Shorkie.
    """
    fig, axes = plt.subplots(1, len(data_types), figsize=(16, 4))
    fig.suptitle(f"Paired Beeswarm - {format_metric_name(metric)}", fontsize=16)

    for i, dt in enumerate(data_types):
        ax = axes[i]
        df = data_dict[dt].dropna(subset=[f"{metric}_sup", f"{metric}_self"])
        sup_vals = df[f"{metric}_sup"].values
        self_vals = df[f"{metric}_self"].values
        n = len(df)

        jitter_sup = np.random.uniform(-0.05, 0.05, size=n)
        jitter_self = np.random.uniform(-0.05, 0.05, size=n)
        x_sup = np.zeros(n) + jitter_sup
        x_self = np.ones(n) + jitter_self

        for j in range(n):
            ax.plot([x_sup[j], x_self[j]],
                    [sup_vals[j], self_vals[j]],
                    linewidth=0.5)

        ax.plot(x_sup, sup_vals, 'o', markersize=1, alpha=0.7, color="#ff7f00") # Orange
        ax.plot(x_self, self_vals, 'o', markersize=1, alpha=0.7, color="#377eb8") # Blue

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Random Init", "Shorkie"])
        ax.set_title(format_data_type(dt))
        ax.set_ylabel(format_metric_name(metric))

        if i == 0:
            ax.legend(["Random Init", "Shorkie"], loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(outdir, "paired_beeswarm.png"), dpi=300)
    plt.close()


def plot_bland_altman_1x4(data_dict, data_types, metric, outdir):
    """
    One row of 4 Bland–Altman plots (Mean vs. Difference).
    """
    fig, axes = plt.subplots(1, len(data_types), figsize=(16, 4))
    fig.suptitle(f"Bland–Altman - {format_metric_name(metric)}", fontsize=16)

    for i, dt in enumerate(data_types):
        ax = axes[i]
        df = data_dict[dt].dropna(subset=[f"{metric}_mean", f"{metric}_diff"])
        x = df[f"{metric}_mean"].values
        y = df[f"{metric}_diff"].values

        ax.scatter(x, y, s=3, color="#377eb8")
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_title(format_data_type(dt))
        ax.set_xlabel("Mean (Shorkie & Random Init)")
        ax.set_ylabel("Difference (Shorkie - Random Init)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(outdir, "bland_altman.png"), dpi=300)
    plt.close()


def plot_ranking_1x4(data_dict, data_types, metric, outdir):
    """
    One row of 4 Ranking plots sorted by Random Init.
    """
    fig, axes = plt.subplots(1, len(data_types), figsize=(16, 4))
    fig.suptitle(f"Ranking (Sorted by Random Init) - {format_metric_name(metric)}", fontsize=16)

    for i, dt in enumerate(data_types):
        ax = axes[i]
        df = data_dict[dt].dropna(subset=[f"{metric}_sup", f"{metric}_self"])
        df_sorted = df.sort_values(by=f"{metric}_sup")
        scratch = df_sorted[f"{metric}_sup"].values
        fine = df_sorted[f"{metric}_self"].values
        ranks = np.arange(len(df_sorted))

        ax.plot(ranks, scratch, marker='o', markersize=1, linewidth=1, color="#ff7f00") # Orange
        ax.plot(ranks, fine, marker='o', markersize=1, linewidth=1, color="#377eb8") # Blue

        ax.set_title(format_data_type(dt))
        ax.set_xlabel("Gene Rank")
        ax.set_ylabel(format_metric_name(metric))

        if i == 0:
            ax.legend(["Random Init", "Shorkie"], loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(outdir, "ranking.png"), dpi=300)
    plt.close()

###############################################################################
#                                    MAIN                                     #
###############################################################################

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("--root_dir", dest="root_dir", default="../../..", type="str",
                      help="Root directory pointing to Yeast_ML")
    parser.add_option("--out_dir", dest="out_dir", default="results", type="str",
                      help="Output directory prefix")
    (options, args) = parser.parse_args()

    model_arch = "unet_small_bert_drop"

    data_types = ["ChIP-exo", "ChIP-MNase", "RNA-Seq", "1000-RNA-seq"]
    metrics = ["pearsonr", "r2", "pearsonr_norm", "r2_norm", "pearsonr_gene"]

    exp_model_type = "exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"

    base_outdir = f"{options.out_dir}/{model_arch}/improved_visuals"
    os.makedirs(base_outdir, exist_ok=True)

    data_dict = load_and_merge_data(options, model_arch, data_types, exp_model_type, n_folds=8)

    for dt in data_dict:
        data_dict[dt] = add_diff_and_mean_cols(data_dict[dt], metrics)

    for m in metrics:
        metric_outdir = os.path.join(base_outdir, m)
        os.makedirs(metric_outdir, exist_ok=True)

        # Paired Beeswarm
        plot_paired_beeswarm_1x4(data_dict, data_types, m, metric_outdir)
        # Bland–Altman
        plot_bland_altman_1x4(data_dict, data_types, m, metric_outdir)
        # Ranking
        plot_ranking_1x4(data_dict, data_types, m, metric_outdir)

if __name__ == "__main__":
    main()
