#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
import pickle
import sys
import argparse  # Added for parsing command-line arguments

def get_insertion_position(fname):
    """
    Extract the insertion position from the filename by mapping the extracted index 
    to the actual nt upstream value.
    
    The mapping is:
        insertion_position = 100 + (index * 10)
        
    Expected filename format:
        <target_gene>_context_<index>_<...>.npz  
    For example:
        YKL062W_context_0_nt_upstream.npz  -> returns 100
        YKL062W_context_1_nt_upstream.npz  -> returns 110
    """
    base = os.path.basename(fname)
    m = re.search(r'_context_(\d+)_', base)
    if m:
        index = int(m.group(1))
        return 100 + index * 10
    return 0

def process_gene_plots(target_gene, gene_name, seq_type, input_dir, output_dir, visualize=True):
    """
    Process the NPZ files for a given gene and create per-context plots,
    a combined subplot figure, and an overall comparison plot.
    
    Additionally, return a dictionary with the overall statistics for this gene.
    
    If visualize is set to False, skip all plotting/saving steps.
    """
    # Look for all NPZ files for this gene.
    npz_files = glob.glob(os.path.join(input_dir, f"{target_gene}_context_*.npz"))
    if not npz_files:
        print(f"No NPZ files found for gene {target_gene} in {input_dir}.")
        return None

    # Sort files by the extracted insertion position.
    npz_files = sorted(npz_files, key=get_insertion_position)
    nfiles = len(npz_files)

    # Lists to collect overall statistics per insertion position.
    insertion_positions = []
    overall_means = []  # overall mean expression (averaged over sequences)
    overall_stds = []   # standard deviation across sequences

    # Process each NPZ file (each insertion position).
    for file_path in npz_files:
        data = np.load(file_path)
        si = data["si"]         # Sequence indices
        logSED = data["logSED"] # Shape: (num_entries, num_targets)
        ctx_id = data["ctx_id"]
        if isinstance(ctx_id, np.ndarray):
            ctx_id = ctx_id.item()
        if isinstance(ctx_id, bytes):
            ctx_id = ctx_id.decode("utf-8")
        
        ins_pos = get_insertion_position(file_path)
        
        # Sort entries by sequence index.
        order = np.argsort(si)
        si_sorted = si[order]
        logSED_sorted = logSED[order, :]

        # Compute per-sequence average and standard deviation (across targets).
        seq_means = np.mean(logSED_sorted, axis=1)
        seq_stds  = np.std(logSED_sorted, axis=1)

        # Create per-insertion plot if visualization is enabled.
        if visualize:
            plt.figure(figsize=(30, 6))
            plt.errorbar(si_sorted, seq_means, yerr=seq_stds, fmt='o', capsize=3, markersize=4)
            plt.xlabel("Sequence Index (0–963)")
            plt.ylabel("Average logSED (across targets)")
            plt.title(f"Gene {target_gene} – Insertion Position: {ins_pos} nt upstream")
            plt.grid(True)
            per_context_filename = os.path.join(output_dir, f"{target_gene}_insertion_{ins_pos}_nt_upstream.png")
            plt.savefig(per_context_filename, bbox_inches='tight')
            plt.close()
            print(f"Saved per-insertion plot for {ins_pos} nt upstream for gene {target_gene} to {per_context_filename}")

        # Overall stats for this insertion.
        overall_mean = np.mean(seq_means)
        overall_std = np.std(seq_means)
        insertion_positions.append(ins_pos)
        overall_means.append(overall_mean)
        overall_stds.append(overall_std)

    # Create a combined figure with one subplot per insertion position if visualization is enabled.
    if visualize:
        fig, axs = plt.subplots(nfiles, 1, figsize=(20, nfiles * 2.5), sharex=True)
        if nfiles == 1:
            axs = [axs]
        
        for i, file_path in enumerate(npz_files):
            data = np.load(file_path)
            si = data["si"]
            logSED = data["logSED"]
            order = np.argsort(si)
            si_sorted = si[order]
            logSED_sorted = logSED[order, :]
            
            seq_means = np.mean(logSED_sorted, axis=1)
            seq_stds  = np.std(logSED_sorted, axis=1)
            
            ins_pos = get_insertion_position(file_path)
            ax = axs[i]
            ax.errorbar(si_sorted, seq_means, yerr=seq_stds, fmt='o', capsize=3, markersize=4)
            ax.set_ylabel(f"{ins_pos} nt", fontsize=10)
            ax.grid(True)
            ax.set_title(f"Insertion Position: {ins_pos} nt upstream", fontsize=10, loc='left')
        
        axs[-1].set_xlabel("Sequence Index (0–963)", fontsize=12)
        fig.suptitle(f"Gene {target_gene} – Per-Insertion Expression Profiles", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        combined_plot_filename = os.path.join(output_dir, f"{target_gene}_all_insertions.png")
        plt.savefig(combined_plot_filename, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined plot with {nfiles} subplots for gene {target_gene} to {combined_plot_filename}")

    # Overall comparison plot for this gene if visualization is enabled.
    if visualize:
        insertion_positions = np.array(insertion_positions)
        overall_means = np.array(overall_means)
        overall_stds = np.array(overall_stds)
        sort_order = np.argsort(insertion_positions)
        insertion_positions = insertion_positions[sort_order]
        overall_means = overall_means[sort_order]
        overall_stds = overall_stds[sort_order]

        plt.figure(figsize=(20, 6))
        plt.errorbar(insertion_positions, overall_means, yerr=overall_stds, fmt='o', capsize=5, markersize=8)
        plt.xlabel("Insertion Position (nt upstream)")
        plt.ylabel("Mean Expression (averaged over sequences)")
        plt.title(f"Comparison of Insertion Positions for Gene {target_gene}")
        plt.grid(True)
        for pos, mean_val in zip(insertion_positions, overall_means):
            plt.annotate(f"{pos} nt upstream", (pos, mean_val),
                         textcoords="offset points", xytext=(0,5), ha='center')
        overall_plot_filename = os.path.join(output_dir, f"{target_gene}_overall_insertion_comparison.png")
        plt.savefig(overall_plot_filename, bbox_inches='tight')
        plt.close()
        print(f"Saved overall comparison plot for gene {target_gene} to {overall_plot_filename}")

    # Return overall stats for later aggregation.
    return {"insertion_positions": np.array(insertion_positions),
            "overall_means": np.array(overall_means),
            "overall_stds": np.array(overall_stds),
            "gene_name": gene_name,
            "target_gene": target_gene,
            "seq_type": seq_type,
            "strand": None}  # strand will be added later

def aggregate_overall_stats_by_seq_type(gene_stats_list, group_label, root_dir, seq_types, y_limits=None):
    """
    Group gene stats by seq_type and create aggregated overall comparison plots
    for each seq_type (only the allowed ones) in a single figure with shared y-axis.
    
    For the "Positive_and_Negative" group, all genes from both strands are plotted together
    with each gene in a unique color and the legend is displayed in two columns.
    
    If y_limits is provided as a tuple (ymin, ymax), it will be used to set the y-axis limits for all subplots.
    """
    # Filter gene stats to only include those with seq_type in the allowed list.
    filtered_gene_stats = [s for s in gene_stats_list if s.get("seq_type") in seq_types]
    if not filtered_gene_stats:
        print("No gene stats available for the allowed sequence types. Skipping aggregation.")
        return

    # Group stats by seq_type using the order provided in seq_types.
    stats_by_seq_type = {}
    for s in seq_types:
        stats_for_type = [stat for stat in filtered_gene_stats if stat.get("seq_type") == s]
        if stats_for_type:
            stats_by_seq_type[s] = stats_for_type

    num_seq_types = len(stats_by_seq_type)
    if num_seq_types == 0:
        print("No gene stats available for the allowed sequence types after filtering. Skipping aggregation.")
        return

    # fig, axs = plt.subplots(num_seq_types, 1, figsize=(20, num_seq_types * 4.2), sharey=True)
    fig, axs = plt.subplots(num_seq_types, 1, figsize=(14, num_seq_types * 3.5), sharey=True)
    if num_seq_types == 1:
        axs = [axs]

    for ax, (seq_type, stats_list) in zip(axs, stats_by_seq_type.items()):
        # Mapping for nicer seq_type names.
        seq_type_mapping = {
            "yeast_seqs": "natural yeast sequences",
            "high_exp_seqs": "high expression sequences",
            "low_exp_seqs": "low expression sequences"
        }
        display_seq_type = seq_type_mapping.get(seq_type, seq_type)
        # ax.set_title(f"Aggregate Overall Comparison for {group_label.replace('_', ' ')} Genes - {display_seq_type}", fontsize=18)
        ax.set_title(f"{display_seq_type}", fontsize=18)
        
        if group_label == "Positive_and_Negative":
            # Use a color palette for all genes so that each gene is plotted in a unique color.
            num_genes = len(stats_list)
            colors = plt.cm.tab20(np.linspace(0, 1, num_genes))
            for i, s in enumerate(stats_list):
                positions = s['insertion_positions']
                means = s['overall_means']
                ax.plot(positions, means, marker='o', linestyle='--', color=colors[i],
                        label=f"{s['gene_name']} ({s['target_gene']}): {s['strand']}", alpha=0.7)
            # Compute and plot aggregate curves for positive and negative strands.
            pos_stats = [s for s in stats_list if s.get("strand") == '+']
            neg_stats = [s for s in stats_list if s.get("strand") == '-']
            if pos_stats:
                pos_curves = np.array([s['overall_means'] for s in pos_stats])
                pos_aggregate_mean = np.mean(pos_curves, axis=0)
                pos_aggregate_std = np.std(pos_curves, axis=0)
                positions = pos_stats[0]['insertion_positions']
                ax.errorbar(positions, pos_aggregate_mean, yerr=pos_aggregate_std,
                            fmt='o-', color='black', markersize=10, capsize=5, label='Aggregate +', linewidth=2)
            if neg_stats:
                neg_curves = np.array([s['overall_means'] for s in neg_stats])
                neg_aggregate_mean = np.mean(neg_curves, axis=0)
                neg_aggregate_std = np.std(neg_curves, axis=0)
                positions = neg_stats[0]['insertion_positions']
                ax.errorbar(positions, neg_aggregate_mean, yerr=neg_aggregate_std,
                            fmt='o-', color='grey', markersize=10, capsize=5, label='Aggregate -', linewidth=2)
            # Display the legend in two columns.
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        else:
            # For single-strand groups.
            all_curves = []
            for s in stats_list:
                positions = s['insertion_positions']
                means = s['overall_means']
                strand_label = s.get("strand", "unknown")
                ax.plot(positions, means, marker='o', linestyle='--',
                        label=f"{s['gene_name']} ({s['target_gene']}) - {strand_label}", alpha=0.6)
                all_curves.append(means)
            if all_curves:
                all_curves = np.array(all_curves)
                aggregate_mean = np.mean(all_curves, axis=0)
                aggregate_std = np.std(all_curves, axis=0)
                positions = stats_list[0]['insertion_positions']
                ax.errorbar(positions, aggregate_mean, yerr=aggregate_std,
                            fmt='o-', color='black', markersize=10, capsize=5, label='Aggregate')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_xlabel("Sequence Replacement Position (nt upstream)")
        ax.set_ylabel("Mean Expression")
        ax.grid(True)
        # Set fixed y-axis limits if provided.
        if y_limits is not None:
            ax.set_ylim(y_limits)
    
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    aggregated_plot_filename = f"{root_dir}/aggregated_{group_label.replace(' ', '_').lower()}_by_seq_type.png"
    plt.savefig(aggregated_plot_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved aggregated overall comparison plot for {group_label} genes by seq_type to {aggregated_plot_filename}")

def main():
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Process gene plots for expression vs position.")
    parser.add_argument("--seq_types", type=str, default="yeast_seqs,high_exp_seqs,low_exp_seqs",
                        help="Comma-separated list of sequence types to include. Options: challenging_seqs, all_random_seqs, yeast_seqs, high_exp_seqs, low_exp_seqs. Default: high_exp_seqs,low_exp_seqs")
    parser.add_argument("--skip-visualization", action="store_true", default=False, help="Skip per-gene visualization")
    args = parser.parse_args()

    # Determine which sequence types to process.
    seq_types = [s.strip() for s in args.seq_types.split(",")]
    visualize_genes = not args.skip_visualization

    root_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/SUM_data_process/MPRA/results/single_measurement_stranded"
    output_root_dir = os.path.join(root_dir, "viz", "expression_vs_position")
    os.makedirs(output_root_dir, exist_ok=True)
    
    aggregated_data_file = os.path.join(output_root_dir, "aggregated_data.pkl")
    load_aggregated = True  # Set to True to load existing aggregated data instead of reprocessing
    aggregated_data = {'+': [], '-': []}

    if load_aggregated and os.path.exists(aggregated_data_file):
        print("Loading aggregated data from", aggregated_data_file)
        with open(aggregated_data_file, "rb") as f:
            aggregated_data = pickle.load(f)
        print("Loaded aggregated data from", aggregated_data_file)
    else:
        # Define gene mappings for positive and negative strands.
        pos_gene_mapping = {
            "GPM3": "YOL056W", 
            "SLI1": "YGR212W", 
            "VPS52": "YDR484W", 
            "YMR160W": "YMR160W", 
            "MRPS28": "YDR337W", 
            "YCT1": "YLL055W", 
            "RDL2": "YOR286W", 
            "PHS1": "YJL097W", 
            "RTC3": "YHR087W", 
            "MSN4": "YKL062W"
        }
        neg_gene_mapping = {
            "COA4": "YLR218C", 
            "ERI1": "YPL096C-A", 
            "RSM25": "YIL093C", 
            "ERD1": "YDR414C", 
            "MRM2": "YGL136C", 
            "SNT2": "YGL131C", 
            "CSI2": "YOL007C",
            "RPE1": "YJL121C", 
            "PKC1": "YBL105C", 
            "AIM11": "YER093C-A", 
            "MAE1": "YKL029C", 
            "MRPL1": "YDR116C"
        }

        # Process positive strand genes.
        strand = '+'
        for seq_type in seq_types:
            print(f"\nProcessing plots for positive strand genes with seq_type {seq_type}...")
            for gene_name, target_gene in pos_gene_mapping.items():
                print(f"\nProcessing plots for gene {target_gene} (symbol: {gene_name}) on strand {strand} with seq_type {seq_type}...")
                input_dir = os.path.join(root_dir, 'all_seq_types', seq_type, f"{gene_name}_{target_gene}_pos_outputs")
                output_dir = os.path.join(input_dir, "plots")
                os.makedirs(output_dir, exist_ok=True)
                gene_stats = process_gene_plots(target_gene, gene_name, seq_type, input_dir=input_dir, output_dir=output_dir, visualize=visualize_genes)
                if gene_stats is not None:
                    gene_stats['strand'] = strand
                    aggregated_data[strand].append(gene_stats)

        # Process negative strand genes.
        strand = '-'
        for seq_type in seq_types:
            print(f"\nProcessing plots for positive strand genes with seq_type {seq_type}...")
            for gene_name, target_gene in neg_gene_mapping.items():
                print(f"\nProcessing plots for gene {target_gene} (symbol: {gene_name}) on strand {strand} with seq_type {seq_type}...")
                input_dir = os.path.join(root_dir, 'all_seq_types', seq_type, f"{gene_name}_{target_gene}_neg_outputs")
                output_dir = os.path.join(input_dir, "plots")
                os.makedirs(output_dir, exist_ok=True)
                gene_stats = process_gene_plots(target_gene, gene_name, seq_type, input_dir=input_dir, output_dir=output_dir, visualize=visualize_genes)
                if gene_stats is not None:
                    gene_stats['strand'] = strand
                    aggregated_data[strand].append(gene_stats)
        
        # Save aggregated data to file for future use.
        with open(aggregated_data_file, "wb") as f:
            pickle.dump(aggregated_data, f)
        print("Saved aggregated data to", aggregated_data_file)
    
    # Compute fixed y-axis limits across all aggregated gene stats.
    all_stats = aggregated_data['+'] + aggregated_data['-']
    if all_stats:
        all_lower = []
        all_upper = []
        for s in all_stats:
            lower = (s['overall_means'] - s['overall_stds']).min()
            upper = (s['overall_means'] + s['overall_stds']).max()
            all_lower.append(lower)
            all_upper.append(upper)
        fixed_y_limits = (min(all_lower), max(all_upper))
    else:
        fixed_y_limits = None
    
    # Create aggregated plots by seq_type for each strand using only allowed sequence types.
    if aggregated_data['+']:
        aggregate_overall_stats_by_seq_type(aggregated_data['+'], "Positive", output_root_dir, seq_types, y_limits=fixed_y_limits)
    if aggregated_data['-']:
        aggregate_overall_stats_by_seq_type(aggregated_data['-'], "Negative", output_root_dir, seq_types, y_limits=fixed_y_limits)
    
    # Combined aggregated plot for both strands by seq_type.
    combined_data = aggregated_data['+'] + aggregated_data['-']
    if combined_data:
        aggregate_overall_stats_by_seq_type(combined_data, "Positive_and_Negative", output_root_dir, seq_types, y_limits=fixed_y_limits)

if __name__ == "__main__":
    main()
