from optparse import OptionParser
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

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
# 2. Alignment helper
# ---------------------------
def pad_array_last_value(arr, desired_length):
    current_length = len(arr)
    if current_length < desired_length:
        pad_length = desired_length - current_length
        last_value = arr[-1] if current_length > 0 else 0.0
        return np.concatenate([arr, np.full(pad_length, last_value)])
    return arr

# ---------------------------
# 3. Plotting helper
# ---------------------------
def plot_multi_comparison(
    data_dict,
    aligned_length,
    metric_name,
    is_loss=False,
    out_filename=None
):
    """
    data_dict: { "Label": numpy array of shape (n_folds, n_epochs) }
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # X-axis
    x = np.arange(aligned_length)

    # Sort keys to have a consistent order (Original first, then Shorkie, then sorted LRs)
    keys = sorted(data_dict.keys())
    
    # Priority list
    priority = ['Shorkie_Random_Init', 'Shorkie']
    for p in reversed(priority):
        if p in keys:
            keys.remove(p)
            keys.insert(0, p)

    for label in keys:
        vals = data_dict[label]
        mean_vals = np.mean(vals, axis=0)
        std_vals  = np.std(vals, axis=0)
        
        final_val = mean_vals[-1]
        final_std = std_vals[-1]
        
        lbl_str = f"{label} (final={final_val:.3f}±{final_std:.3f})"
        
        # Style adjustments
        if label == 'Shorkie_Random_Init':
            color = 'black'
            linestyle = '--'
            linewidth = 2.5
            zorder = 10
        elif label == 'Shorkie':
            color = 'red'
            linestyle = '-.'
            linewidth = 2.5
            zorder = 9
        else:
            color = None # auto
            linestyle = '-'
            linewidth = 1.5
            zorder = 5

        line, = ax.plot(x, mean_vals, label=lbl_str, linestyle=linestyle, linewidth=linewidth, color=color, zorder=zorder, alpha=0.8)
        color_used = line.get_color()
        ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color_used, alpha=0.05, zorder=zorder-1)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title(f"Comparison: {metric_name}", fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename, dpi=300)
        print(f"Saved {out_filename}")
    plt.close()

def plot_summary_avg(
    data_dict_loss,
    data_dict_r,
    aligned_length,
    out_filename=None
):
    """
    Creates a summary figure with 1 row x 2 columns:
      - Left: Validation Loss
      - Right: Validation Pearson's R
    """
    fig, (ax_loss, ax_r) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # X-axis
    x = np.arange(aligned_length)

    # Sort keys to have a consistent order
    # (Same sorting logic as plot_multi_comparison)
    keys = sorted(data_dict_loss.keys())
    priority = ['Shorkie_Random_Init', 'Shorkie']
    for p in reversed(priority):
        if p in keys:
            keys.remove(p)
            keys.insert(0, p)

    # Helper inner function to plot on a specific axis
    def plot_on_ax(ax, data_dict, metric_name, is_loss_metric):
        for label in keys:
            vals = data_dict[label]
            mean_vals = np.mean(vals, axis=0)
            std_vals  = np.std(vals, axis=0)
            
            final_val = mean_vals[-1]
            final_std = std_vals[-1]
            
            lbl_str = f"{label} (final={final_val:.3f}±{final_std:.3f})"
            
            # Style adjustments
            if label == 'Shorkie_Random_Init':
                color = 'black'
                linestyle = '--'
                linewidth = 2.5
                zorder = 10
            elif label == 'Shorkie':
                color = 'red'
                linestyle = '-.'
                linewidth = 2.5
                zorder = 9
            else:
                color = None # auto
                linestyle = '-'
                linewidth = 1.5
                zorder = 5

            line, = ax.plot(x, mean_vals, label=lbl_str, linestyle=linestyle, linewidth=linewidth, color=color, zorder=zorder, alpha=0.8)
            color_used = line.get_color()
            ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color_used, alpha=0.05, zorder=zorder-1)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_title(f"Comparison: {metric_name}", fontsize=14)
        ax.grid(True, alpha=0.3)
        # We'll put the legend on the right plot or maybe both? 
        # Usually putting close to plot is better.
        # But if it's too crowded, maybe below. Let's stick to 'best'.
        ax.legend(fontsize=8, loc='best')

    # Plot Loss (Left)
    plot_on_ax(ax_loss, data_dict_loss, "Validation Loss", True)
    
    # Plot R (Right)
    plot_on_ax(ax_r, data_dict_r, "Validation Pearson's R", False)

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
    # models_data[label] = { 'files': [], 'metrics': ... }
    models_data = {}
    
    # Add Shorkie_Random_Init (LR=0.0005)
    models_data['Shorkie_Random_Init'] = {'files': random_init_files}
    # Add Shorkie
    models_data['Shorkie'] = {'files': shorkie_files}

    # Add Variants
    for lr_dir in lr_dirs:
        basename = os.path.basename(lr_dir) # e.g., learning_rate_0.001
        label = basename.replace('learning_rate_', 'LR=')
        
        # Collect 8 folds
        files = []
        for i in range(8):
            f_path = os.path.join(lr_dir, 'train', f'f{i}c0', 'train.out')
            files.append(f_path)
        
        models_data[label] = {'files': files}

    # 3. Parse Data
    # We need to collect lists of arrays for alignment
    # storage = { metric_name: { label: [arr_fold0, arr_fold1, ...] } }
    
    metrics_map = {
        'train_loss': {}, 'train_r': {}, 'train_r2': {},
        'valid_loss': {}, 'valid_r': {}, 'valid_r2': {}
    }
    
    for label, info in models_data.items():
        print(f"Parsing {label}...")
        
        # Temp lists for this model
        m_lists = {k: [] for k in metrics_map.keys()}
        
        for fpath in info['files']:
            if not os.path.exists(fpath):
                print(f"  Warning: File not found: {fpath}")
                # Append empty or partial? Let's skip or handle gracefully.
                # If we skip, alignment might break if we expect 8 folds.
                # For now let's just assume missing means empty arrays.
                # But parse_all_metrics handles reading errors.
                # We'll just create empty arrays so logic doesn't crash?
                # Actually, best to just have 0-length arrays.
                m_lists['train_loss'].append(np.array([]))
                m_lists['train_r'].append(np.array([]))
                m_lists['train_r2'].append(np.array([]))
                m_lists['valid_loss'].append(np.array([]))
                m_lists['valid_r'].append(np.array([]))
                m_lists['valid_r2'].append(np.array([]))
                continue
                
            out = parse_all_metrics(fpath)
            # out is (tr_loss, tr_r, tr_r2, vl_loss, vl_r, vl_r2)
            
            m_lists['train_loss'].append(out[0])
            m_lists['train_r'].append(out[1])
            m_lists['train_r2'].append(out[2])
            m_lists['valid_loss'].append(out[3])
            m_lists['valid_r'].append(out[4])
            m_lists['valid_r2'].append(out[5])
            
        # Store into main map
        for k in metrics_map.keys():
            metrics_map[k][label] = m_lists[k]

    # 4. Alignment
    # Determine max length across ALL models and ALL folds for each metric
    # Actually usually length is consistent across metrics (epochs), but let's be safe.
    # We can just align everything to the global max epoch count found.
    
    max_len = 0
    for k in metrics_map:
        for label in metrics_map[k]:
            for arr in metrics_map[k][label]:
                if len(arr) > max_len:
                    max_len = len(arr)
    
    print(f"Max aligned length: {max_len}")
    
    # Pad everything
    # final_data = { metric_name: { label: np.array(n_folds, max_len) } }
    final_data = {}
    
    for k in metrics_map:
        final_data[k] = {}
        for label in metrics_map[k]:
            padded_list = [pad_array_last_value(a, max_len) for a in metrics_map[k][label]]
            
            # If any are empty, they will be padded with 0 (from empty 0 len) - check pad_array_last_value logic
            # My helper logic for empty is: last_value=0.0.
            
            # Stack
            if len(padded_list) > 0:
                final_data[k][label] = np.vstack(padded_list)
            else:
                final_data[k][label] = np.zeros((1, max_len)) # fallback

    # 5. Plotting
    plot_multi_comparison(final_data['train_loss'], max_len, "Train Loss", is_loss=True, 
                          out_filename=os.path.join(save_root, "train_loss_avg.png"))
    plot_multi_comparison(final_data['valid_loss'], max_len, "Validation Loss", is_loss=True, 
                          out_filename=os.path.join(save_root, "valid_loss_avg.png"))
    
    plot_multi_comparison(final_data['train_r'], max_len, "Train Pearson's R", is_loss=False, 
                          out_filename=os.path.join(save_root, "train_r_avg.png"))
    plot_multi_comparison(final_data['valid_r'], max_len, "Validation Pearson's R", is_loss=False, 
                          out_filename=os.path.join(save_root, "valid_r_avg.png"))
    
    plot_multi_comparison(final_data['train_r2'], max_len, "Train R²", is_loss=False, 
                          out_filename=os.path.join(save_root, "train_r2_avg.png"))
    plot_multi_comparison(final_data['valid_r2'], max_len, "Validation R²", is_loss=False, 
                          out_filename=os.path.join(save_root, "valid_r2_avg.png"))

    # 6. Combined Summary Plot (Valid Loss + Valid R)
    plot_summary_avg(
        final_data['valid_loss'],
        final_data['valid_r'],
        max_len,
        out_filename=os.path.join(save_root, "valid_avg_summary.png")
    )

if __name__ == "__main__":
    main()
