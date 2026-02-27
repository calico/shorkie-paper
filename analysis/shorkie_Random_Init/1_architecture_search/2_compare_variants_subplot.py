import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def _moving_average(x, window_size=1, trim_end=0):
    x_avg = np.zeros_like(x, dtype=float)
    for j in range(x.shape[0]):
        min_j = max(j - window_size // 2, 0)
        max_j = min(j + window_size // 2, x.shape[0] - 1)
        # simplistic moving average
        x_avg[j] = np.mean(x[min_j:max_j + 1])
    if trim_end > 0 and trim_end < x.shape[0]:
        return x_avg[:-trim_end]
    else:
        return x_avg

def get_model_params(log_file):
    """
    Parses 'Total params: X' from the log file.
    Returns the formatted string (e.g. '69.17M') or 'N/A' if not found.
    """
    if not os.path.exists(log_file):
        return "N/A"
    
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("Total params: "):
                # Line format: "Total params: 69172927 (263.87 MB)"
                try:
                    # Extract the number before the first space
                    val_str = line.strip().split(": ")[1].split(" ")[0]
                    val = int(val_str)
                    return f"{val/1e6:.2f}M"
                except:
                    return "N/A"
    return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Compare model variants (subplot)")
    parser.add_argument("--root_dir", default="../../../Yeast_ML", help="Root directory for Yeast_ML data")
    parser.add_argument("--out_dir", default="./results", help="Output directory")
    args = parser.parse_args()

    save_root = os.path.join(args.out_dir, 'model_variants_comparison')
    os.makedirs(save_root, exist_ok=True)
    
    exp_root = f"{args.root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
    variants_root = f"{exp_root}/supervised_unet_small_bert_drop_model_variants"
    
     # Define models
    model_defs = [
        {
            "name": "CovNet Smaller",
            "path": f"{variants_root}/covnet_smaller",
            "color": "#d62728", # red
            "linestyle": "--"
        },
        {
            "name": "U-Net Smaller",
            "path": f"{variants_root}/unet_smaller",
            "color": "#1f77b4", # blue
            "linestyle": "-."
        },
        {
            "name": "Shorkie_Random_Init (U-Net Small)",
            "path": f"{exp_root}/supervised_unet_small_bert_drop",
            "color": "black",
            "linestyle": "-"
        },
        # {
        #     "name": "U-Net Big",
        #     "path": f"{variants_root}/unet_big",
        #     "color": "#2ca02c", # green
        #     "linestyle": ":"
        # }
    ]
    
    # Common settings
    steps_per_epoch = 500
    window_size = 11
    trim_end = 5
    
    # ---------------------------
    # Load data
    # ---------------------------
    # Structure: all_data[metric][model_idx][fold_idx] -> np.array
    metrics_keys = ["train_loss", "train_r", "train_r2", "valid_loss", "valid_r", "valid_r2"]
    all_data = {m: [] for m in metrics_keys}
    
    for m_def in model_defs:
        path_base = m_def["path"]
        
        # Get params from fold 0 log
        log_path_f0 = f"{path_base}/train/f0c0/train.out"
        params_str = get_model_params(log_path_f0)
        m_def["params_str"] = params_str

        print(f"Loading {m_def['name']} (Params: {params_str})...")
        
        m_folds_data = {m: [] for m in metrics_keys}
        
        for fold in range(8):
            log_path = f"{path_base}/train/f{fold}c0/train.out"
            
            local_lists = {k: [] for k in metrics_keys}
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    for line_raw in f:
                        line = line_raw.strip()
                        if line.startswith("Epoch "):
                            try:
                                parts = line.split(" - ")
                                # expected typical parts:
                                # [0]: Epoch X
                                # [1]: train_loss: val
                                # [2]: train_r: val...
                                # Indices in split(" - ") array:
                                # 2: train_loss, 3: train_r, 4: train_r2
                                # 5: valid_loss, 6: valid_r, 7: valid_r2
                                t_loss = float(parts[2].split(": ")[1])
                                t_r    = float(parts[3].split(": ")[1])
                                t_r2   = float(parts[4].split(": ")[1])
                                v_loss = float(parts[5].split(": ")[1])
                                v_r    = float(parts[6].split(": ")[1])
                                v_r2   = float(parts[7].split(": ")[1])
                                
                                local_lists["train_loss"].append(t_loss)
                                local_lists["train_r"].append(t_r)
                                local_lists["train_r2"].append(t_r2)
                                local_lists["valid_loss"].append(v_loss)
                                local_lists["valid_r"].append(v_r)
                                local_lists["valid_r2"].append(v_r2)
                            except:
                                continue
            else:
                 print(f"  Missing: {log_path}")

            # Convert to arrays
            for k in metrics_keys:
                m_folds_data[k].append(np.array(local_lists[k]))
                
        # Add to main structure
        for k in metrics_keys:
            all_data[k].append(m_folds_data[k])

    # ---------------------------
    # Plotting
    # ---------------------------
    # We want 6 large images (one for each metric), each with 8 subplots
    
    metric_display_names = {
        "train_loss": "Train Loss",
        "valid_loss": "Validation Loss",
        "train_r": "Train Pearson's R",
        "valid_r": "Validation Pearson's R",
        "train_r2": "Train R²",
        "valid_r2": "Validation R²"
    }
    
    for m_key in metrics_keys:
        m_name = metric_display_names[m_key]
        print(f"Plotting {m_name}...")
        
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=300, sharey=False) # sharey=False to let each fold scale? or True? usually True is better for comparison but False helps see dynamics if scales differ wildly. Let's try sharey=False first but maybe True per row? No, usually sharey=True is preferred for same metric.
        # Actually existing script uses sharey=True. Let's stick to that if possible, or maybe sharey='all'
        
        # determine global min/max for y-axis to share?
        # or just let matplotlib handle sharey=True
        
        # update: let's do sharey=True to force comparable scales
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        is_loss = "loss" in m_key
        
        for fold_i in range(8):
            ax = axes[fold_i // 4, fold_i % 4]
            ax.set_title(f"Fold {fold_i}", fontsize=12)
            
            for model_idx, m_def in enumerate(model_defs):
                # data for this metric, this model, this fold
                vals = all_data[m_key][model_idx][fold_i]
                if len(vals) == 0:
                    continue
                
                # smoothing
                smoothed = _moving_average(vals, window_size=window_size, trim_end=trim_end)
                steps_arr = (np.arange(len(vals)) + 1) * steps_per_epoch
                # trim steps to match smoothed
                steps_arr = steps_arr[:len(smoothed)]
                
                # Find best
                if is_loss:
                    best_val = np.min(smoothed)
                else:
                    best_val = np.max(smoothed)
                
                # Plot
                label_str = f"{m_def['name']}\n{m_def['params_str']}\n({best_val:.3f})"
                ax.plot(
                    steps_arr, smoothed,
                    color=m_def['color'],
                    linestyle=m_def['linestyle'],
                    linewidth=1.5,
                    label=label_str
                )
                
            if fold_i // 4 == 1:
                ax.set_xlabel("# Batches", fontsize=10)
            if fold_i % 4 == 0:
                ax.set_ylabel(m_name, fontsize=10)
                
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            
        fig.suptitle(f"Comparison: {m_name}", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        out_file = os.path.join(save_root, f"subplot_{m_key}.png")
        plt.savefig(out_file)
        print(f"Saved {out_file}")
        plt.close()

if __name__ == "__main__":
    main()
