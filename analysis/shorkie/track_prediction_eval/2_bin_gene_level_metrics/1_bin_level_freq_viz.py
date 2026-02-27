import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import argparse

# Define parameters
parser = argparse.ArgumentParser(description="Evaluate bin-level frequency visualizations.")
parser.add_argument("--root_dir", required=True, help="Root directory for seq_experiment/")
parser.add_argument("--out_dir", default="./results", help="Output directory prefix")
args = parser.parse_args()

root_dir = args.root_dir
out_dir = args.out_dir

training_types = ["supervised", "self_supervised"]
folds = list(range(8))
eval_types = ["eval"]  # You can also include "eval_train" if needed
model_archs = ["unet_small_bert_drop"]

eval_exp = "exp_histone__chip_exo__rna_seq_no_norm_5215_tracks"

# ---- 1. Helper function to categorize based on description text ----
def categorize_description(desc):
    desc_lower = desc.lower()
    if "pos_logfe" in desc_lower or "chip-exo" in desc_lower:
        return "ChIP-exo"
    elif "chip-mnase" in desc_lower or "mnase" in desc_lower:
        return "ChIP-MNase"
    elif "1000 strains rnaseq" in desc_lower:
        return "1000 strains RNA-Seq"
    elif "rnaseq" in desc_lower or "rna_seq" in desc_lower:
        return "RNA-Seq"
    else:
        return "Other"

# ---- 2. Load data into master_df_list ----
master_df_list = []
for training_type in training_types:
    for eval_type in eval_types:
        for model_arch in model_archs:
            print(f">> Processing {training_type}; {eval_type}; {model_arch}")
            output_dir = f"{out_dir}_{eval_exp}/{model_arch}/{training_type}/{eval_type}"
            os.makedirs(output_dir, exist_ok=True)
            
            condition_dfs = []
            for fold in folds:
                if training_type == "supervised":
                     eval_out_log = (
                        f"{root_dir}/"
                        f"seq_experiment/{eval_exp}/16bp/"
                        f"{training_type}_{model_arch}_variants/learning_rate_0.0005/train/f{fold}c0/{eval_type}/acc.txt"
                    )
                else:
                    eval_out_log = (
                        f"{root_dir}/"
                        f"seq_experiment/{eval_exp}/16bp/"
                        f"{training_type}_{model_arch}/train/f{fold}c0/{eval_type}/acc.txt"
                    )
                try:
                    df_fold = pd.read_csv(eval_out_log, sep="\t")
                    df_fold["track_type"]     = df_fold["description"].apply(categorize_description)
                    df_fold["training_type"]  = training_type
                    df_fold["eval_type"]      = eval_type
                    df_fold["model_arch"]     = model_arch
                    df_fold["fold"]           = fold
                    condition_dfs.append(df_fold)
                except FileNotFoundError:
                    print(f"File not found: {eval_out_log}")
                except Exception as e:
                    print(f"Error processing {eval_out_log}: {e}")
            
            if condition_dfs:
                master_df_list.append(pd.concat(condition_dfs, ignore_index=True))
            else:
                print(f"No data for {training_type}, {eval_type}, {model_arch}")

# ---- Combined Density Plots: median + dotted median line + track count ----
training_label_map = {
    "supervised": "Shorkie_Random_Init",
    "self_supervised": "Shorkie"
}
base_color_map = {
    "ChIP-exo": "#1f77b4",
    "ChIP-MNase": "#ff7f0e",
    "1000 strains RNA-Seq": "#2ca02c",
    "RNA-Seq": "#d62728",
}
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    rgb = mcolors.to_rgb(c)
    return [(1-amount)*comp + amount for comp in rgb]

if master_df_list:
    combined_df = pd.concat(master_df_list, ignore_index=True)
    metrics = ["pearsonr", "r2"]
    metrics_lbls = ["Pearson's R", r"$R^2$"]

    for idx, metric in enumerate(metrics):
        data = combined_df.dropna(subset=[metric])
        grouped = data.groupby(["track_type", "training_type"])
        # plt.figure(figsize=(9.4, 6.2))
        plt.figure(figsize=(9.2, 6.2))

        for (track, ttype), grp in grouped:
            if track == "Other":
                continue

            # median per identifier
            meds = grp.groupby("identifier")[metric].median().values
            if meds.size == 0:
                continue
            med_val = np.median(meds)
            n_tracks = len(meds)

            base_color = base_color_map.get(track, "#7f7f7f")
            color = lighten_color(base_color, 0.6) if ttype=="supervised" else base_color

            # KDE or histogram
            if ttype == "self_supervised":
                plt.hist(meds, bins=80, density=True, alpha=0.3, color=color, range=(0,1))
                kde = gaussian_kde(meds)
                x = np.linspace(0,1,200)
                y = kde(x)
                plt.fill_between(x,0,y,color=color,alpha=0.2)
                plt.plot(x,y, color=color)
            else:
                kde = gaussian_kde(meds)
                x = np.linspace(0,1,200)
                y = kde(x)
                plt.plot(x,y, color=color)

            # dotted median line + legend
            label = f"{track} ({training_label_map[ttype]}): median={med_val:.3f}, n={n_tracks}"
            plt.axvline(med_val, linestyle='--', color=color, linewidth=2, label=label)

        plt.title("Density of Bin Level Predictions", fontsize=20)
        plt.xlim(0, 0.9)
        plt.xlabel(metrics_lbls[idx], fontsize=16)
        plt.ylabel("Density", fontsize=16)
        plt.grid(axis="y", alpha=0.75)
        plt.legend(fontsize=9.2, loc="best")
        out_file = f"{out_dir}_{eval_exp}/combined_{metric}_median_density.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.show()


# -------------------------------------------------------------------------
#
# ---- 3. Comparison Plots Section ----
# Global mapping for training types
TRAINING_TYPE_LABELS = {
    "supervised": "Shorkie_Random_Init",
    "self_supervised": "Shorkie",
}

EVAL_TYPE_LABELS = {
    "eval": "Test set",
    "eval_train": "Train set",
}

COLOR_MAP = {
    ("supervised", "eval"): "#1f77b4",         # darker blue
    ("supervised", "eval_train"): "#aec7e8",     # lighter blue
    ("self_supervised", "eval"): "#ff7f0e",       # darker orange
    ("self_supervised", "eval_train"): "#FFDBBB",  # lighter orange
}

LINESTYLES = {
    "eval": "-",
    "eval_train": "--",
}

def comparison_plots(master_df):
    """
    Create bar plots and distribution (KDE) comparison plots overlaying
    'eval' and 'eval_train' in the same figure, also displaying average
    Pearson's R and number of tracks (identifiers) in the legend.
    """
    comparison_output_dir = f"{out_dir}_{eval_exp}/comparison_supervised_self_supervised"
    os.makedirs(comparison_output_dir, exist_ok=True)
    
    dist_output_dir = os.path.join(comparison_output_dir, "distributions")
    os.makedirs(dist_output_dir, exist_ok=True)

    metrics_of_interest = ["pearsonr", "r2"]
    metrics_of_interest_display = ["Pearson's R", r"$R^2$"]

    base_colors = {
        "self_supervised": "#377eb8",  # Blue
        "supervised": "#ff7f00",       # Orange
    }
    eval_shades = {
        "eval_train": 0.4,
        "eval": 1,
    }

    for model_arch in master_df["model_arch"].unique():
        os.makedirs(f"{comparison_output_dir}/{model_arch}/", exist_ok=True)
        os.makedirs(f"{dist_output_dir}/{model_arch}/", exist_ok=True)

        arch_df = master_df[master_df["model_arch"] == model_arch].copy()

        agg_df = (
            arch_df
            .groupby(["track_type", "training_type", "eval_type", "identifier"], as_index=False)
            .agg({
                "pearsonr": "mean",
                "r2": "mean",
                "description": "first"
            })
        )

        # --- 3A. BAR PLOTS ---
        for metric_name, metric_label in zip(metrics_of_interest, metrics_of_interest_display):
            grouped = (
                agg_df
                .groupby(["track_type", "training_type", "eval_type"])[metric_name]
                .mean()
                .reset_index()
            )
            pivot_df = grouped.pivot(
                index="track_type",
                columns=["training_type", "eval_type"],
                values=metric_name
            )

            if "Other" in pivot_df.index:
                pivot_df.drop(index="Other", inplace=True, errors="ignore")

            if pivot_df.empty:
                continue

            new_cols = []
            bar_colors = []
            for (t_type, e_type) in pivot_df.columns:
                t_label = TRAINING_TYPE_LABELS.get(t_type, t_type)
                e_label = EVAL_TYPE_LABELS.get(e_type, e_type)
                new_cols.append(f"{t_label} ({e_label})")
                base_color = mcolors.to_rgba(base_colors[t_type])
                alpha = eval_shades[e_type]
                bar_colors.append(mcolors.to_rgba(base_color, alpha))
            pivot_df.columns = new_cols

            ax = pivot_df.plot(
                kind="bar",
                figsize=(8, 3.1),
                width=0.8,
                color=bar_colors,
            )
            ax.set_title(f"Comparison of Average {metric_label} by Track Type\n")
            ax.set_xlabel("Track Type")
            ax.set_ylabel(metric_label)
            plt.xticks(rotation=0)
            plt.grid(axis="y", alpha=0.75)
            plt.legend(loc="best", ncol=1)
            plt.tight_layout()

            bar_out_path = os.path.join(
                comparison_output_dir,
                model_arch,
                f"{metric_name}_comparison_bar.png"
            )
            plt.savefig(bar_out_path, dpi=300)
            plt.show()

        # --- 3B. DISTRIBUTION (KDE) COMPARISONS ---
        for metric_name, metric_label in zip(metrics_of_interest, metrics_of_interest_display):
            agg_valid = agg_df.dropna(subset=[metric_name])
            track_types_in_data = sorted(agg_valid["track_type"].unique())
            track_types_in_data = [t for t in track_types_in_data if t != "Other"]

            if len(track_types_in_data) == 0:
                continue

            for track_name in track_types_in_data:
                data_for_track = agg_valid[agg_valid["track_type"] == track_name]
                if data_for_track.empty:
                    continue

                plt.figure(figsize=(6.2, 4.13))
                combos_plotted = 0

                for t_type in ["supervised", "self_supervised"]:
                    for e_type in ["eval", "eval_train"]:
                        subset_te = data_for_track[
                            (data_for_track["training_type"] == t_type) &
                            (data_for_track["eval_type"] == e_type)
                        ]
                        if subset_te.empty:
                            continue

                        metric_values = subset_te[metric_name].values
                        kde = gaussian_kde(metric_values)
                        x_min, x_max = 0, 1
                        x_grid = np.linspace(x_min, x_max, 200)
                        y_grid = kde(x_grid)

                        train_label = "Shorkie_Random_Init" if t_type == "supervised" else "Shorkie"
                        eval_label = EVAL_TYPE_LABELS.get(e_type, e_type)

                        if metric_name == "pearsonr":
                            avg_r = metric_values.mean()
                            n_ids = subset_te["identifier"].nunique()
                            combo_label = f"{train_label} ({eval_label}) \n[avg R={avg_r:.3f}, n={n_ids}]"
                        else:
                            combo_label = f"{train_label} ({eval_label})"

                        color = COLOR_MAP.get((t_type, e_type), "gray")

                        plt.plot(
                            x_grid,
                            y_grid,
                            color=color,
                            linestyle=LINESTYLES.get(e_type, "-"),
                            label=combo_label
                        )
                        plt.fill_between(
                            x_grid, 0, y_grid,
                            color=color,
                            alpha=0.2
                        )
                        combos_plotted += 1

                if combos_plotted == 0:
                    plt.close()
                    continue

                plt.title(f"Distribution of {metric_label}\nTrack: {track_name}, Model: {model_arch}")
                plt.xlabel(metric_label)
                plt.ylabel("Density")
                plt.grid(axis="y", alpha=0.75)
                plt.legend()

                dist_out_path = os.path.join(
                    dist_output_dir,
                    model_arch,
                    f"{track_name}_{metric_name}_kde_eval_evaltrain.png"
                )
                plt.tight_layout()
                plt.savefig(dist_out_path, dpi=300)
                plt.show()

# ---- 4. New Scatter Plots: Comparing Shorkie vs Shorkie_Random_Init ----
def scatter_comparison_plots(master_df):
    """
    For each model architecture, evaluation type, track type, and metric,
    create scatter plots comparing the identifier-level averages for
    'Shorkie_Random_Init' (supervised) versus 'Shorkie' (self_supervised).
    The x-axis represents Shorkie_Random_Init and the y-axis represents Shorkie.
    """
    metrics = ["pearsonr", "r2"]
    metrics_display = ["Pearson's R", "R²"]
    
    for model_arch in master_df["model_arch"].unique():
        df_model = master_df[master_df["model_arch"] == model_arch]
        for eval_type in df_model["eval_type"].unique():
            df_eval = df_model[df_model["eval_type"] == eval_type]
            for track in df_eval["track_type"].unique():
                if track == "Other":
                    continue
                df_track = df_eval[df_eval["track_type"] == track]
                for metric, metric_disp in zip(metrics, metrics_display):
                    # Group by identifier and training type and compute mean metric
                    df_group = df_track.groupby(["identifier", "training_type"])[metric].mean().reset_index()
                    # Pivot so that we have one row per identifier with columns for each training type
                    df_pivot = df_group.pivot(index="identifier", columns="training_type", values=metric)
                    # Ensure both training types are available
                    if "supervised" not in df_pivot.columns or "self_supervised" not in df_pivot.columns:
                        continue
                    # Drop rows with missing values in either column
                    df_pivot = df_pivot.dropna(subset=["supervised", "self_supervised"])
                    if df_pivot.empty:
                        continue
                    x = df_pivot["supervised"]  # Shorkie_Random_Init
                    y = df_pivot["self_supervised"]  # Shorkie
                    # Calculate Pearson correlation between x and y
                    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
                    
                    plt.figure(figsize=(6, 6))
                    plt.scatter(x, y, alpha=0.7)
                    # Plot diagonal line for reference
                    min_val = min(x.min(), y.min())
                    max_val = max(x.max(), y.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
                    
                    plt.xlabel(f"Shorkie_Random_Init ({metric_disp})")
                    plt.ylabel(f"Shorkie ({metric_disp})")
                    plt.title(f"{track} - {eval_type}\nScatter: r = {corr:.2f}")
                    plt.grid(True, alpha=0.5)
                    
                    out_dir_scatter = os.path.join(f"{out_dir}_{eval_exp}/{model_arch}/scatter")
                    os.makedirs(out_dir_scatter, exist_ok=True)
                    out_path = os.path.join(out_dir_scatter, f"{track}_{metric}_scatter_{eval_type}.png")
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=300)
                    plt.show()

# ---- 5. New Scatter Plot Combining All Groups ----
def scatter_all_groups_scatter(master_df):
    """
    Creates one scatter plot per metric (at the identifier level) for each model architecture,
    combining all track types (except 'Other') in a single figure. Each track type (group)
    is plotted with a distinct color, and group mean points are added.
    """
    METRICS = ["pearsonr", "r2"]
    METRICS_TITLES = {"pearsonr": "Pearson's R", "r2": "R²"}

    # Define output directory for combined scatter plots
    output_dir_scatter_all = f"{out_dir}_{eval_exp}/scatter_all_groups"
    os.makedirs(output_dir_scatter_all, exist_ok=True)
    scatter_dir = os.path.join(output_dir_scatter_all, "identifier_scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)

    # colors = ["#6699FF", "#66FF66", "#EE4B2B", "#FFCC33", "#A08000", "#800020"]
    # mean_colors = ["#0000A0", "#005000", "#800020", "#A08000", "#A05000", "#501080"]
    # Color-blind friendly palette (Blue, Orange, Green, Purple, Brown, Pink)
    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628", "#f781bf"]
    # Darker shades for means
    mean_colors = ["#2b6291", "#cc6600", "#3d8c3b", "#773d80", "#854420", "#c46698"]

    for model_arch in master_df["model_arch"].unique():
        df_model = master_df[master_df["model_arch"] == model_arch]
        df_eval = df_model[df_model["eval_type"] == "eval"]

        groups = []
        for track in df_eval["track_type"].unique():
            if track == "Other" or track == "ChIP-exo" or track == "ChIP-MNase":
                continue
            df_track = df_eval[df_eval["track_type"] == track]
            # Compute identifier-level averages for each training type
            df_group = (
                df_track
                .groupby(["identifier", "training_type"])
                .agg({"pearsonr": "mean", "r2": "mean"})
                .reset_index()
            )
            # Pivot so supervised vs self-supervised columns
            df_pivot = df_group.pivot(
                index="identifier",
                columns="training_type",
                values=["pearsonr", "r2"]
            )
            df_pivot.columns = [
                f"{metric}_{'sup' if tt=='supervised' else 'self'}"
                for metric, tt in df_pivot.columns
            ]
            df_pivot.reset_index(inplace=True)
            groups.append((df_pivot, track))

        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            # plot each group
            for idx, (gdf, label) in enumerate(groups):
                xcol = f"{metric}_sup"
                ycol = f"{metric}_self"
                if xcol in gdf and ycol in gdf:
                    pts = gdf.dropna(subset=[xcol, ycol])
                    ax.scatter(
                        pts[xcol],
                        pts[ycol],
                        s=15,
                        color=colors[idx % len(colors)],
                        alpha=0.45,
                        label=label
                    )

            # plot group means
            for idx, (gdf, label) in enumerate(groups):
                xcol = f"{metric}_sup"
                ycol = f"{metric}_self"
                if xcol in gdf and ycol in gdf:
                    sup_vals = gdf[xcol].dropna()
                    self_vals = gdf[ycol].dropna()
                    if len(sup_vals) > 0 and len(self_vals) > 0:
                        x_mean = sup_vals.mean()
                        y_mean = self_vals.mean()
                    else:
                        x_mean, y_mean = None, None
                    if x_mean is not None and y_mean is not None:
                        mean_label = f"Shorkie_Random_Init Mean (x): {x_mean:.2f}, \nShorkie Mean (y): {y_mean:.2f}"
                        ax.scatter(
                            x_mean,
                            y_mean,
                            s=100,
                            edgecolor="black",
                            color=mean_colors[idx % len(mean_colors)],
                            label=mean_label  # Add both mean labels to the legend
                        )

            # unity line across current axes limits
            ax.autoscale()  # let matplotlib set limits
            lims = ax.get_xlim(), ax.get_ylim()
            min_lim = min(lims[0][0], lims[1][0])
            max_lim = max(lims[0][1], lims[1][1])
            ax.plot([min_lim, max_lim], [min_lim, max_lim], "k--", alpha=0.6)

            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{METRICS_TITLES.get(metric, metric)} (Bin level)", fontsize=10)
            ax.set_xlabel("Shorkie_Random_Init")
            ax.set_ylabel("Shorkie")
            ax.legend(fontsize=9, loc="best")
            plt.tight_layout()

            out_fname = os.path.join(
                scatter_dir,
                f"scatter_all_{metric}_identifier_{model_arch}.png"
            )
            fig.savefig(out_fname, dpi=300)
            plt.close(fig)

# ---- 6. Box and Violin Plots ----
def plot_box_violin(master_df):
    """
    Generates box and violin plots for metrics (Pearson R, R2),
    aggregating by identifier (median across folds).
    """
    print("Generating Box and Violin plots...")
    output_dir_box = f"{out_dir}_{eval_exp}/box_violin_plots"
    os.makedirs(output_dir_box, exist_ok=True)
    
    metrics = ["pearsonr", "r2"]
    metrics_lbls = ["Pearson's R", "R²"]
    
    # Filter for 'eval' set only
    df_eval = master_df[master_df["eval_type"] == "eval"]
    
    for metric, label in zip(metrics, metrics_lbls):
        # Aggregate: Median per identifier across folds
        df_agg = (df_eval
                  .groupby(["track_type", "training_type", "identifier"])[metric]
                  .median()
                  .reset_index())
        
        # Filter out 'Other'
        df_agg = df_agg[df_agg["track_type"] != "Other"]

        # Rename training types for Legend
        df_agg["training_type"] = df_agg["training_type"].map(TRAINING_TYPE_LABELS)
        
        # --- Create detailed labels with counts and medians ---
        # Count unique identifiers per track_type
        counts = df_agg.groupby("track_type")["identifier"].nunique()
        
        # Calculate medians per track_type and training_type
        medians = df_agg.groupby(["track_type", "training_type"])[metric].median()
        
        # Create mapping: "ChIP-exo" -> "ChIP-exo\n(n=150)\nShorkie med=...\nRandom med=..."
        def get_detailed_label(t_type):
            c = counts.get(t_type, 0)
            m_shorkie = medians.get((t_type, "Shorkie"), np.nan)
            m_random = medians.get((t_type, "Shorkie_Random_Init"), np.nan)
            return f"{t_type}\n(n={c})\nShorkie med={m_shorkie:.3f}\nRandom med={m_random:.3f}"
            
        df_agg["track_type_detailed"] = df_agg["track_type"].apply(get_detailed_label)
        
        # Define preferred order: RNA-Seq first, then others alphabetically
        unique_tracks = sorted(df_agg["track_type"].unique())
        if "RNA-Seq" in unique_tracks:
            unique_tracks.remove("RNA-Seq")
            unique_tracks = ["RNA-Seq"] + unique_tracks
            
        # Map preferred order to detailed labels
        order_detailed = [get_detailed_label(t) for t in unique_tracks]
        
        # --- Violin Plot ---
        plt.figure(figsize=(8, 7))  # Increased size for detailed labels
        sns.violinplot(
            data=df_agg, 
            x="track_type_detailed", 
            y=metric, 
            hue="training_type", 
            split=True, 
            inner="quart",
            palette={"Shorkie": "#377eb8", "Shorkie_Random_Init": "#ff7f00"},
            order=order_detailed
        )
        plt.title(f"{label} Distribution by Track Type (Violin)", fontsize=16)
        plt.xlabel("Track Type", fontsize=14)
        plt.ylabel(label, fontsize=14)
        plt.legend(title="Model", loc='upper right', fontsize=14, title_fontsize=16)
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir_box}/violin_{metric}.png", dpi=300)
        plt.close()
        
        # --- Box Plot ---
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            data=df_agg, 
            x="track_type_detailed", 
            y=metric, 
            hue="training_type", 
            palette={"Shorkie": "#377eb8", "Shorkie_Random_Init": "#ff7f00"},
            showfliers=False,
            order=order_detailed
        )
        plt.title(f"{label} Distribution by Track Type (Box)", fontsize=16)
        plt.xlabel("Track Type", fontsize=14)
        plt.ylabel(label, fontsize=14)
        plt.legend(title="Model", loc='upper right', fontsize=14, title_fontsize=16)
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir_box}/boxplot_{metric}.png", dpi=300)
        plt.close()
        
    print(f"Box and Violin plots saved to {output_dir_box}")

def darken_color(color, amount=0.2):
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    rgb = mcolors.to_rgb(c)
    return [comp * (1 - amount) for comp in rgb]

# ---- 7. Split Density Plots (Up/Down) ----
def plot_split_density(master_df):
    """
    Plots density distributions in two subplots:
    Upper: Shorkie (self_supervised)
    Lower: Shorkie_Random_Init (supervised)
    """
    print("Generating Split Density plots...")
    metrics = ["pearsonr", "r2"]
    metrics_lbls = ["Pearson's R", r"$R^2$"]

    df_eval = master_df[master_df["eval_type"] == "eval"]

    for idx, metric in enumerate(metrics):
        data = df_eval.dropna(subset=[metric])
        # Only proceed if we have data for both types
        if data.empty:
            continue
            
        # Reduced height by ~30% per request (originally 10 -> now 7)
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        
        # Define the two panels configuration
        panels = [
            # (ax_index, training_type_key, title)
            (0, "self_supervised", "Shorkie (Self-Supervised)"),
            (1, "supervised",      "Shorkie_Random_Init (Supervised)")
        ]

        # Iterate over both panels
        for ax_idx, ttype, title in panels:
            ax = axes[ax_idx]
            
            # Subset data for this training type
            subset = data[data["training_type"] == ttype]
            if subset.empty:
                continue
            
            # Group by track_type to plot each track density
            grouped = subset.groupby("track_type")
            
            for track, grp in grouped:
                if track == "Other":
                    continue
                
                # median per identifier (to match previous logic)
                meds = grp.groupby("identifier")[metric].median().values
                if meds.size == 0:
                    continue
                
                med_val = np.median(meds)
                n_tracks = len(meds)
                
                base_color = base_color_map.get(track, "#7f7f7f")
                
                if ttype == "self_supervised":
                    # Shorkie: 10% darker
                    color = darken_color(base_color, 0.1)
                else:
                    # Random Init: 20% lighter
                    color = lighten_color(base_color, 0.2)
                
                # Plot Histogram (Original request)
                ax.hist(meds, bins=80, density=True, alpha=0.3, color=color, range=(0,1))

                # Plot KDE
                try:
                    kde = gaussian_kde(meds)
                    x = np.linspace(0, 1, 200)
                    y = kde(x)
                    
                    label = f"{track}: med={med_val:.3f}, n={n_tracks}"
                    ax.plot(x, y, color=color, label=label)
                    # ax.fill_between(x, 0, y, color=color, alpha=0.2) # Optional fill
                    
                    # Median line (only on current axis)
                    ax.axvline(med_val, linestyle='--', color=color, linewidth=2)
                        
                except Exception as e:
                    print(f"Skipping KDE for {track} ({ttype}) due to error: {e}")

            ax.set_title(title, fontsize=14)
            ax.set_ylabel("Density", fontsize=12)
            ax.grid(axis="y", alpha=0.75)
            # ax.legend(fontsize=9, loc="upper left")
            # Move legend outside or corner to avoid clutter if lines cross
            ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
        
        axes[1].set_xlabel(metrics_lbls[idx], fontsize=14)
        axes[0].set_xlim(0, 0.9)
        
        plt.tight_layout()
        out_file = f"{out_dir}_{eval_exp}/combined_{metric}_median_density_split.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved {out_file}")

# ========== Main Execution ==========
if __name__ == "__main__":
    if master_df_list:
        master_df = pd.concat(master_df_list, ignore_index=True)
        # Combined median-density plots already ran above
        comparison_plots(master_df)
        scatter_comparison_plots(master_df)
        scatter_all_groups_scatter(master_df)
        plot_box_violin(master_df)
        plot_split_density(master_df)
    else:
        print("No data collected in master_df_list, so skipping comparison and scatter plots.")