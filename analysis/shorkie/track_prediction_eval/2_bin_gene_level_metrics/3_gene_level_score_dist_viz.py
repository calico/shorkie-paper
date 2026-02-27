#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # In case you're on a system without GUI
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from scipy.stats import gaussian_kde

training_label_map = {
    "supervised": "Random init",
    "self_supervised": "Shorkie"
}
base_color_map = {
    "ChIP-exo": "#1f77b4",
    "ChIP-MNase": "#ff7f0e",
    "1000-RNA-seq": "#2ca02c",
    "RNA-Seq": "#d62728",
}
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc, colorsys
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = mc.to_rgb(c)
    return [(1 - amount) * comp + amount for comp in c]

METRICS = ["pearsonr", "r2", "pearsonr_norm", "r2_norm", "pearsonr_gene", "pearsonr_gene_norm"]

METRICS_TITLES = {
    "pearsonr": "Pearson's R",
    "r2": "R²",
    "pearsonr_norm": "Pearson's R Norm",
    "r2_norm": "R² Norm",
    "pearsonr_gene": "Pearson's R within-gene",
    "pearsonr_gene_norm": "Pearson's R within-gene Norm",    
}

def load_evaluation_data(options, exp_model_type, model_arch, data_type, fold_idx, level="track"):
    """
    Load a single fold of evaluation data (supervised & self-supervised), merge them, and return as DataFrame.
    level="track" uses 'acc.txt'
    level="gene"  uses 'gene_acc.txt'
    """
    eval_subdir = options.experiment
    if level == "track":
        eval_filename = "acc.txt"
    else:
        eval_filename = "gene_acc.txt"

    self_supervised_acc_fn = (
        f"{options.root_dir}/seq_experiment/"
        f"{exp_model_type}/16bp/self_supervised_{model_arch}/"
        f"{eval_subdir}/f{fold_idx}c0/{data_type}/{eval_filename}"
    )
    supervised_acc_fn = (
        f"{options.root_dir}/seq_experiment/"
        f"{exp_model_type}/16bp/supervised_{model_arch}_variants/learning_rate_0.0005/"
        f"{eval_subdir}/f{fold_idx}c0/{data_type}/{eval_filename}"
    )

    print(f"Loading data for {data_type} at {level}-level from fold {fold_idx} ...")
    print(f"  Self-supervised: {self_supervised_acc_fn}")
    print(f"  Supervised: {supervised_acc_fn}")

    self_supervised_acc_df = pd.read_csv(self_supervised_acc_fn, sep="\t")
    supervised_acc_df = pd.read_csv(supervised_acc_fn, sep="\t")

    # Decide on merge columns
    if level == "track":
        on_columns = ["identifier", "description", "group"]
    else:
        if "gene_id" in self_supervised_acc_df.columns:
            on_columns = ["gene_id"]
        else:
            on_columns = ["identifier"]

    merged_df = pd.merge(
        self_supervised_acc_df,
        supervised_acc_df,
        on=on_columns,
        suffixes=("_self", "_sup"),
    )
    return merged_df


def load_and_average_across_folds(options, exp_model_type, model_arch, data_type, level="track", num_folds=8):
    """
    Load data for multiple folds, average them by grouping over the key columns.
    Returns a DataFrame with the mean across folds for each item (identifier/gene).
    If any fold is missing its files, that fold is skipped.
    """
    all_folds = []
    for idx in range(num_folds):
        try:
            merged_df = load_evaluation_data(options, exp_model_type, model_arch, data_type, idx, level=level)
            print("len(merged_df): ", len(merged_df))
            all_folds.append(merged_df)
        except FileNotFoundError as e:
            print(f"[WARNING] Skipping fold {idx} for {data_type} at {level}-level: {e}")
            continue

    if not all_folds:
        print(f"[WARNING] No valid folds for {data_type} at {level}-level.")
        return pd.DataFrame()

    all_folds_df = pd.concat(all_folds, ignore_index=True)
    print("len(all_folds_df): ", len(all_folds_df))

    if level == "track":
        group_cols = ["identifier", "description", "group"]
    else:
        if "gene_id" in all_folds_df.columns:
            group_cols = ["gene_id"]
        else:
            group_cols = ["identifier"]

    mean_df = (
        all_folds_df
        .groupby(group_cols, as_index=False)
        .mean(numeric_only=True)
    )
    print("len(mean_df): ", len(mean_df))
    print("mean_df.head(): ", mean_df.head())
    return mean_df


def plot_scatter_comparison(df, group_label, output_dir, level):
    # Subdirectory for scatter plots
    scatter_dir = os.path.join(output_dir, f"{level}_scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)

    for metric in METRICS:
        col_sup = f"{metric}_sup"
        col_self = f"{metric}_self"
        if col_sup not in df.columns or col_self not in df.columns:
            print(f"[WARNING] Metric {metric} not in columns. Skipping.")
            continue

        x_vals = df[col_sup].values
        y_vals = df[col_self].values

        # Remove NaNs
        mask = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        if len(x_vals) == 0:
            print(f"[WARNING] No valid data for {metric}. Skipping.")
            continue

        # --- 1) Remove any scores < 0 ---
        nonneg_mask = (x_vals >= 0) & (y_vals >= 0)
        x_vals = x_vals[nonneg_mask]
        y_vals = y_vals[nonneg_mask]

        # --- 2) Compute % of points above diagonal (y > x) ---
        total = len(x_vals)
        if total > 0:
            above = np.sum(y_vals > x_vals)
            pct_above = above / total * 100
        else:
            pct_above = 0.0
        print(f"[INFO] {metric}: {above}/{total} points above diagonal → {pct_above:.1f}%")


        # Increase figure size for a larger main panel
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(x_vals, y_vals, s=15, color="#1f77b4", alpha=0.8, edgecolor="none")

        # Mean values (only for points > 0)
        valid_sup = x_vals[x_vals > 0]
        valid_self = y_vals[y_vals > 0]
        x_mean = valid_sup.mean() if len(valid_sup) > 0 else 0
        y_mean = valid_self.mean() if len(valid_self) > 0 else 0
        mean_point = ax.scatter(x_mean, y_mean, color="red", s=100, edgecolor="black", label=f"Mean: ({x_mean:.2f}, {y_mean:.2f})")

        # Diagonal
        ax.plot([-0.05, 1.05], [-0.05, 1.05], "k--", alpha=0.6)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")

        # Labels with updated mapping:
        # x-axis (supervised) → "Shorkie_Random_Init"
        # y-axis (self_supervised) → "Shorkie"
        avg_grp = None
        if level == "gene":
            avg_grp = "tracks"
        elif level == "track":
            avg_grp = "genes"
        title_str = f"{group_label} - {METRICS_TITLES.get(metric, metric)} ({level} level; average across {avg_grp})"
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("Shorkie_Random_Init")
        ax.set_ylabel("Shorkie")
        ax.legend()

        plt.tight_layout()
        out_name = os.path.join(scatter_dir, f"scatter_{group_label}_{metric}_{level}.png")
        plt.savefig(out_name, dpi=300)
        plt.close()


def plot_all_groups_scatter(groups, output_dir, level):
    """
    Plots all data types in the same scatter figure for each metric (1 figure per metric).
    Each group is plotted with its own distinct color.
    """
    scatter_dir = os.path.join(output_dir, f"{level}_scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)

    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628", "#f781bf"]
    mean_colors = ["#2b6291", "#cc6600", "#3d8c3b", "#773d80", "#854420", "#c46698"]

    # Plot each metric in a single figure
    for metric in METRICS:
        plt.figure(figsize=(4.8, 4.8))

        # Initialize lists to track the min and max values for x and y axes
        all_x_vals = []
        all_y_vals = []

        for gp_idx, (group_df, group_label) in enumerate(groups):
            col_sup = f"{metric}_sup"
            col_self = f"{metric}_self"
            if (col_sup not in group_df.columns) or (col_self not in group_df.columns):
                continue

            subset = group_df[[col_sup, col_self]].dropna()
            x_vals = subset[col_sup].values
            y_vals = subset[col_self].values

            all_x_vals.extend(x_vals)
            all_y_vals.extend(y_vals)

            plt.scatter(
                x_vals,
                y_vals,
                s=15,
                color=colors[gp_idx % len(colors)],
                alpha=0.45,
                label=group_label
            )

        # Add mean points for each group and update the legend with mean values
        for gp_idx, (group_df, group_label) in enumerate(groups):
            col_sup = f"{metric}_sup"
            col_self = f"{metric}_self"
            if (col_sup not in group_df.columns) or (col_self not in group_df.columns):
                continue
            valid_sup = group_df[col_sup].dropna()
            valid_self = group_df[col_self].dropna()
            if (len(valid_sup[valid_sup > 0]) == 0) or (len(valid_self[valid_self > 0]) == 0):
                x_mean, y_mean = 0, 0
            else:
                x_mean = valid_sup[valid_sup > 0].mean()
                y_mean = valid_self[valid_self > 0].mean()

            # Label both Shorkie_Random_Init mean (x-axis) and Shorkie mean (y-axis)
            mean_label = f"Shorkie_Random_Init Mean (x): {x_mean:.2f}, \nShorkie Mean (y): {y_mean:.2f}"
            
            plt.scatter(
                x_mean,
                y_mean,
                s=100,
                edgecolor="black",
                color=mean_colors[gp_idx % len(mean_colors)],
                label=mean_label  # Add both mean labels to the legend
            )

    

        # Set dynamic limits based on the data's min and max values
        x_min, x_max = min(all_x_vals), max(all_x_vals)
        y_min, y_max = min(all_y_vals), max(all_y_vals)
        axis_min = min(x_min, y_min)
        axis_max = max(x_max, y_max)

        if axis_min < 0:
            axis_min = 0.05
        # Add some padding to the min/max values to make the plot more visually appealing
        padding = 0.05  # Adjust this value for more or less padding
        plt.xlim(axis_min - padding, axis_max + padding)
        plt.ylim(axis_min - padding, axis_max + padding)

        plt.plot([axis_min - padding, axis_max + padding], [axis_min - padding, axis_max + padding], "k--", alpha=0.6)
        plt.gca().set_aspect("equal", adjustable="box")

        if level == "gene":
            level_name = "Track"
        elif level == "track":
            level_name = "Gene"
        

        title_str = f"{METRICS_TITLES.get(metric, metric)} ({level_name} Level)"
        plt.title(title_str, fontsize=11)

        plt.xlabel("Shorkie_Random_Init (x)")
        plt.ylabel("Shorkie (y)")
        plt.legend(fontsize=9, loc="lower right")
        plt.tight_layout()

        out_fname = os.path.join(scatter_dir, f"scatter_all_{metric}_{level}.png")
        plt.savefig(out_fname, dpi=300)
        plt.close()

def plot_score_distributions(df, data_label, output_dir, level="track"):
    """
    Creates separate figures (one per metric) for distribution of self-supervised vs. supervised scores.
    """
    dist_dir = os.path.join(output_dir, f"{level}_score_distributions", data_label)
    os.makedirs(dist_dir, exist_ok=True)

    for metric in METRICS:
        self_col = f"{metric}_self"
        sup_col = f"{metric}_sup"
        if self_col not in df.columns or sup_col not in df.columns:
            continue

        plt.figure(figsize=(6, 4.5))
        # Updated legend labels based on name mapping
        sns.kdeplot(
            data=df,
            x=self_col,
            fill=True,
            alpha=0.4,
            label="Shorkie",
            color="#377eb8"
        )
        sns.kdeplot(
            data=df,
            x=sup_col,
            fill=True,
            alpha=0.4,
            label="Shorkie_Random_Init",
            color="#ff7f00"
        )

        min_val = min(df[self_col].min(), df[sup_col].min())
        plt.xlim(min_val - 0.2, 1.0)
        plt.title(f"{data_label} - {metric} Distribution ({level} level)", fontsize=14)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(loc="upper left")

        plot_path = os.path.join(dist_dir, f"{metric}_distribution_{level}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()


def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("--model_arch", dest="model_arch", default=None, type="str",
                      help="Model architecture name to be used for reading files.")
    parser.add_option("--experiment", dest="experiment", default=None, type="str",
                      help="Experiment type (e.g. exp_histone__chip_exo__rna_seq_no_norm_5215_tracks).")
    parser.add_option("--root_dir", dest="root_dir", default="../../..", type="str",
                      help="Root directory pointing to Yeast_ML")
    parser.add_option("--out_dir", dest="out_dir", default="results", type="str",
                      help="Output directory prefix")
    (options, args) = parser.parse_args()

    if options.model_arch is None:
        parser.error("Please specify --model_arch")

    exp_model_type = "exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"
    # data_types = ["ChIP-exo", "ChIP-MNase", "RNA-Seq", "1000-RNA-seq"]

    data_types = ["RNA-Seq", "1000-RNA-seq"]
    num_folds = 8

    # We process both gene-level and track-level METRICS.
    for level in ["gene", "track"]:
    # for level in ["gene"]:
        output_dir = f"{options.out_dir}_{exp_model_type}/{level}_level/{options.experiment}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n========== Processing {level}-level METRICS ==========\n")

        dataframes = []
        groups = []  # will hold (df, label) for plotting

        for dt in data_types:
            print(f"Loading data_type={dt} ...")
            try:
                df_avg = load_and_average_across_folds(
                    options, exp_model_type, options.model_arch, dt,
                    level=level, num_folds=num_folds
                )
                print(f"  [INFO] Loaded data for {dt} at {level}-level.")
                print(df_avg.head())

                if len(df_avg) == 0:
                    print(f"  [WARNING] No data for {dt} at {level}-level. Skipping.")
                    continue

                # NEW FILTERING STEP: remove bottom 10% by coverage_norm_self, if present.
                if "coverage_norm_self" in df_avg.columns:
                    threshold = np.percentile(df_avg["coverage_norm_self"].dropna(), 10)
                    before_count = len(df_avg)
                    df_avg = df_avg[df_avg["coverage_norm_self"] > threshold]
                    after_count = len(df_avg)
                    print(f"  [INFO] Filtered out bottom 10% coverage_norm_self: {before_count} -> {after_count}")

                dataframes.append((dt, df_avg))
                groups.append((df_avg, dt))

            except Exception as e:
                print(f"  [ERROR] Problem reading data for {dt} at {level}-level: {e}")

        # For gene-level "all" scatter plot, only keep RNA types
        if level == "gene":
            groups = [(group_df, group_label) for group_df, group_label in groups 
                      if group_label.lower() in ["rna-seq", "1000-rna-seq"]]

        # 1) Plot separate scatter plots and distributions for each data_type
        for df_label, df_avg in dataframes:
            print(f"Plotting {df_label} ...")
            plot_scatter_comparison(df_avg, df_label, output_dir, level=level)
            # plot_score_distributions(df_avg, df_label, output_dir, level=level)

        # 2) Plot all groups together on the same scatter (one figure per metric)
        if len(groups) > 1:
            plot_all_groups_scatter(groups, output_dir, level=level)

        # ----- Combined Distribution Plots -----
        master_df_list = []
        for dt, df_avg in dataframes:
            print(f"Processing {dt} ...")
            print(f"  len(df_avg): {len(df_avg)}")
            print(f"  df_avg.head(): {df_avg.head()}")
            # pick the right id column
            id_col = 'identifier' if 'identifier' in df_avg.columns else 'gene_id'

            # SUPERVISED
            sup = (
                df_avg[[id_col] + [f"{m}_sup" for m in METRICS]]
                # rename id_col → identifier, and metric_sup → metric
                .rename(columns={id_col: 'identifier', **{f"{m}_sup": m for m in METRICS}})
                .melt(id_vars=['identifier'], var_name='metric', value_name='value')
            )
            sup['training_type'] = 'supervised'
            sup['track_type']    = dt

            # SELF‑SUPERVISED
            self_ = (
                df_avg[[id_col] + [f"{m}_self" for m in METRICS]]
                .rename(columns={id_col: 'identifier', **{f"{m}_self": m for m in METRICS}})
                .melt(id_vars=['identifier'], var_name='metric', value_name='value')
            )
            self_['training_type'] = 'self_supervised'
            self_['track_type']    = dt

            master_df_list.extend([sup, self_])

        combined_df = pd.concat(master_df_list, ignore_index=True)
        print(f"len(combined_df): {len(combined_df)}")
        print(combined_df.head())

        # ----- Combined Distribution Plots (updated) -----
        # melt as before into combined_df with columns:
        # ['identifier','metric','value','training_type','track_type']
        metrics_of_interest         = ["pearsonr", "r2"]
        metrics_of_interest_display = ["Pearson's R",    "R²"]

        for disp_name, metric_name in zip(metrics_of_interest_display, metrics_of_interest):
            # plt.figure(figsize=(9.5, 5))
            plt.figure(figsize=(6.5, 3.5))

            # loop over each track_type and training_type
            for track_name in combined_df['track_type'].unique():
                for training_type in ["supervised", "self_supervised"]:
                    # select only this metric + group
                    subset = combined_df[
                        (combined_df['metric'] == metric_name) &
                        (combined_df['track_type'] == track_name) &
                        (combined_df['training_type'] == training_type)
                    ]['value'].dropna()

                    if subset.empty:
                        continue

                    # compute KDE
                    kde = gaussian_kde(subset.values)
                    xg = np.linspace(0, 1, 200)
                    yg = kde(xg)

                    # pick color
                    base = base_color_map.get(track_name, "#7f7f7f")
                    color = (
                        lighten_color(base, amount=0.6)
                        if training_type == 'supervised'
                        else base
                    )

                    label = f"{track_name} ({training_label_map[training_type]})"
                    plt.plot(xg, yg, label=label, color=color)
                    plt.fill_between(xg, 0, yg, color=color, alpha=0.2)

            level_cap = level.capitalize()
            plt.title(f"{disp_name} Density of {level_cap}-Level Predictions", fontsize=16)
            plt.xlabel(disp_name, fontsize=14)
            plt.ylabel("Density", fontsize=14)
            # plt.xlim(0, 1)
            plt.grid(axis='y', alpha=0.5)
            # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.legend(loc='upper left', frameon=False)
            out_comb = os.path.join(output_dir, f"combined_{metric_name}_density_{level}.png")
            plt.tight_layout()
            plt.savefig(out_comb, dpi=300)
            plt.close()
        # ----- end combined block -----




if __name__ == "__main__":
    main()
