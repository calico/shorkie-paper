from optparse import OptionParser
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------
# 1. Parsing function
# ---------------------------
def parse_all_metrics(log_file, prefix="Epoch "):
    """
    Parses train_loss, train_r, train_r2, valid_loss, valid_r, valid_r2
    from each epoch line in the given log file.
    """
    train_loss_list = []
    train_r_list = []
    train_r2_list = []
    valid_loss_list = []
    valid_r_list = []
    valid_r2_list = []

    with open(log_file, "r") as f:
        for line_raw in f:
            line = line_raw.strip()
            if line.startswith(prefix):
                # Example: "Epoch 1 - train_loss: 0.5 - train_r: 0.45 - ..."
                parts = line.split(" - ")
                try:
                    train_loss_val = float(parts[2].split(": ")[1])
                    train_r_val    = float(parts[3].split(": ")[1])
                    train_r2_val   = float(parts[4].split(": ")[1])
                    valid_loss_val = float(parts[5].split(": ")[1])
                    valid_r_val    = float(parts[6].split(": ")[1])
                    valid_r2_val   = float(parts[7].split(": ")[1])

                    train_loss_list.append(train_loss_val)
                    train_r_list.append(train_r_val)
                    train_r2_list.append(train_r2_val)
                    valid_loss_list.append(valid_loss_val)
                    valid_r_list.append(valid_r_val)
                    valid_r2_list.append(valid_r2_val)
                except (IndexError, ValueError):
                    continue

    return (np.array(train_loss_list),
            np.array(train_r_list),
            np.array(train_r2_list),
            np.array(valid_loss_list),
            np.array(valid_r_list),
            np.array(valid_r2_list))


# ---------------------------
# 2. Plotting helper
# ---------------------------
def plot_bar_comparison(
    metric_results, # { 'Label': {'mean': float, 'std': float} }
    metric_name,
    out_filename,
    y_label=None
):
    fig, ax = plt.subplots(figsize=(6.8, 6))

    # Sort keys: Special ones first, then others sorted naturally (LRs)
    keys = sorted(metric_results.keys())
    priority = ['Shorkie_Random_Init', 'Shorkie']
    
    sorted_keys = []
    # Add priority keys if they exist
    for p in priority:
        if p in keys:
            sorted_keys.append(p)
    # Add rest
    for k in keys:
        if k not in priority:
            sorted_keys.append(k)

    means = [metric_results[k]['mean'] for k in sorted_keys]
    stds  = [metric_results[k]['std'] for k in sorted_keys]
    
    # Colors
    # Special colors for priority, others generic or cmap
    colors = []
    cmap = cm.get_cmap('tab10')
    lr_count = 0
    # Count how many non-priority items to distribute colors if needed, 
    # but simplest is just fixed mapping or dynamic.
    
    for i, k in enumerate(sorted_keys):
        if k == 'Shorkie_Random_Init':
            colors.append('gray')
        elif k == 'Shorkie':
            colors.append('red')
        else:
            # Use a colormap for LRs
            # We don't know exact index in 'keys' easily without re-enumerating, 
            # let's just cycle or use index in the NON-priority list
            colors.append('skyblue') # Simple fallback or could use specific colors
            
    x_pos = np.arange(len(sorted_keys))

    bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, ecolor='black', capsize=5, color=colors)
    
    ax.set_ylabel(y_label if y_label else metric_name, fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_keys, rotation=45, ha='right', fontsize=10)
    ax.set_title(f"Comparison: {metric_name} (Best Epoch)", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        std_val = stds[i]
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std_val),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename, dpi=300)
        print(f"Saved {out_filename}")
    plt.close()


def main():
    parser = OptionParser()
    parser.add_option('--model_arch', dest='model_arch', default='unet_small_bert_drop', type='str',
                      help='Model architecture[Default: %default]')
    parser.add_option('--root_dir', dest='root_dir', default='../../../Yeast_ML', type='str',
                      help='Root directory for Yeast_ML data')
    parser.add_option('--out_dir', dest='out_dir', default='./results', type='str',
                      help='Output directory')
    (options, args) = parser.parse_args()

    # Paths
    save_root = os.path.join(options.out_dir, 'lr_comparison')
    os.makedirs(save_root, exist_ok=True)

    exp_root = f"{options.root_dir}/seq_experiment"
    variants_dir = f'{exp_root}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_{options.model_arch}_variants'
    
    # 1. Collect Shorkie_Random_Init files (LR=0.0005 variant)
    random_init_dir = os.path.join(variants_dir, 'learning_rate_0.0005')
    random_init_files = [
        os.path.join(random_init_dir, 'train', f'f{i}c0', 'train.out')
        for i in range(8)
    ]

    # 1.1 Collect Shorkie Model Files
    shorkie_files = [
        f'{exp_root}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_{options.model_arch}/train/f{i}c0/train.out'
        for i in range(8)
    ]

    # 2. Collect Variant Models (Learning Rates), excluding 0.0005 (now Shorkie_Random_Init)
    lr_dirs = glob.glob(os.path.join(variants_dir, "learning_rate_*"))
    lr_dirs = [d for d in lr_dirs if os.path.basename(d) != 'learning_rate_0.0005']
    
    # Structure to hold all data:
    models_data = {}
    
    models_data['Shorkie_Random_Init'] = {'files': random_init_files}
    models_data['Shorkie'] = {'files': shorkie_files}

    for lr_dir in lr_dirs:
        basename = os.path.basename(lr_dir) # e.g., learning_rate_0.001
        label = basename.replace('learning_rate_', 'LR=')
        
        files = []
        for i in range(8):
            f_path = os.path.join(lr_dir, 'train', f'f{i}c0', 'train.out')
            files.append(f_path)
        
        models_data[label] = {'files': files}

    # 3. Parse and Compute Best Metrics
    
    # We want: { metric_name: { label: { 'mean': ..., 'std': ... } } }
    final_stats = {
        'train_loss': {}, 'train_r': {}, 'train_r2': {},
        'valid_loss': {}, 'valid_r': {}, 'valid_r2': {}
    }
    
    for label, info in models_data.items():
        print(f"Processing {label}...")
        
        # Store best values for each fold
        fold_bests = {k: [] for k in final_stats.keys()}
        
        for fpath in info['files']:
            if not os.path.exists(fpath):
                # We interpret missing file as no data -> skip or error?
                # skipping means fewer folds contribution
                continue
                
            out = parse_all_metrics(fpath)
            # out: (tr_loss, tr_r, tr_r2, vl_loss, vl_r, vl_r2)
            
            # Extract Best
            # Loss -> Min, R/R2 -> Max
            
            if len(out[0]) == 0: continue # Should not happen if file exists and parse works
            
            fold_bests['train_loss'].append(np.min(out[0]))
            fold_bests['train_r'].append(np.max(out[1]))
            fold_bests['train_r2'].append(np.max(out[2]))
            
            fold_bests['valid_loss'].append(np.min(out[3]))
            fold_bests['valid_r'].append(np.max(out[4]))
            fold_bests['valid_r2'].append(np.max(out[5]))
            
        # Compute stats for this label
        for metric in fold_bests:
            vals = np.array(fold_bests[metric])
            if len(vals) > 0:
                final_stats[metric][label] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals)
                }
            else:
                 final_stats[metric][label] = {'mean': 0.0, 'std': 0.0}

    # 4. Plotting
    plot_bar_comparison(final_stats['train_loss'], "Train Loss", 
                        os.path.join(save_root, "train_loss_bar.png"))
    plot_bar_comparison(final_stats['valid_loss'], "Validation Loss", 
                        os.path.join(save_root, "valid_loss_bar.png"))
    
    plot_bar_comparison(final_stats['train_r'], "Train Pearson's R", 
                        os.path.join(save_root, "train_r_bar.png"))
    plot_bar_comparison(final_stats['valid_r'], "Validation Pearson's R", 
                        os.path.join(save_root, "valid_r_bar.png"))
    
    plot_bar_comparison(final_stats['train_r2'], "Train R²", 
                        os.path.join(save_root, "train_r2_bar.png"), y_label="R²")
    plot_bar_comparison(final_stats['valid_r2'], "Validation R²", 
                        os.path.join(save_root, "valid_r2_bar.png"), y_label="R²")

if __name__ == "__main__":
    main()
