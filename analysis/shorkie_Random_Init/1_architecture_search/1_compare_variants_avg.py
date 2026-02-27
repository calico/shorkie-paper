import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---------------------------
# 1. Parsing function for each log file
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

    if not os.path.exists(log_file):
        print(f"Warning: File not found: {log_file}")
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]))

    with open(log_file, "r") as f:
        for line_raw in f:
            line = line_raw.strip()
            if line.startswith(prefix):
                # Example: "Epoch 1 - train_loss: 0.5 - train_r: 0.45 - train_r2: 0.20 - valid_loss: 0.55 - valid_r: 0.43 - valid_r2: 0.18"
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
                except Exception as e:
                    print(f"Error parsing line in {log_file}: {line}\n{e}")
                    continue

    return (np.array(train_loss_list),
            np.array(train_r_list),
            np.array(train_r2_list),
            np.array(valid_loss_list),
            np.array(valid_r_list),
            np.array(valid_r2_list))


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


# ---------------------------
# 2. Alignment function
# ---------------------------
def pad_array_last_value(arr, desired_length):
    current_length = len(arr)
    if current_length == 0:
        return np.zeros(desired_length) # Should not happen ideally
    if current_length < desired_length:
        pad_length = desired_length - current_length
        last_value = arr[-1]
        return np.concatenate([arr, np.full(pad_length, last_value)])
    return arr[:desired_length]

def align_arrays(list_of_lists_of_arrays):
    """
    Find max length across ALL models and ALL folds, then pad everyone to that length.
    list_of_lists_of_arrays: [ [fold1, fold2...], [fold1, fold2...] ... ] for each model
    Returns: list of numpy arrays (n_folds, max_len), one per model
    """
    max_len = 0
    for file_list in list_of_lists_of_arrays:
        for arr in file_list:
            if len(arr) > max_len:
                max_len = len(arr)
    
    out_models = []
    for file_list in list_of_lists_of_arrays:
        padded_list = [pad_array_last_value(arr, max_len) for arr in file_list]
        out_models.append(np.vstack(padded_list))
        
    return out_models, max_len


# ---------------------------
# 3. Plotting function
# ---------------------------
def plot_multi_model_metric(
    model_data_list, model_names,
    aligned_length, metric_name, is_loss=False,
    out_filename=None, colors=None
):
    """
    model_data_list: list of numpy arrays, each shape (n_folds, n_epochs)
    model_names: list of strings
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(aligned_length)
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, data in enumerate(model_data_list):
        mean_val = np.mean(data, axis=0)
        std_val  = np.std(data, axis=0)
        
        final_mean = mean_val[-1]
        final_std = std_val[-1]
        
        lbl = f"{model_names[i]} (final={final_mean:.3f}±{final_std:.3f})"
        
        ax.plot(x, mean_val, linewidth=2, label=lbl, color=colors[i])
        ax.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.05, color=colors[i])

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title(f"Model Comparison: {metric_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if is_loss:
        # Check if we want log scale? Maybe not for now unless requested.
        pass

    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename, dpi=300)
        print(f"Saved: {out_filename}")
    # plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare model variants (average)")
    parser.add_argument("--root_dir", default="../../../Yeast_ML", help="Root directory for Yeast_ML data")
    parser.add_argument("--out_dir", default="./results", help="Output directory")
    args = parser.parse_args()

    save_root = os.path.join(args.out_dir, 'model_variants_comparison')
    os.makedirs(save_root, exist_ok=True)
    
    exp_root = f"{args.root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
    variants_root = f"{exp_root}/supervised_unet_small_bert_drop_model_variants"
    
    # Define models
    models = [
        {
            "name": "CovNet Smaller",
            "path": f"{variants_root}/covnet_smaller",
            "color": "#d62728" # red
        },
        {
            "name": "U-Net Smaller",
            "path": f"{variants_root}/unet_smaller",
            "color": "#1f77b4" # blue
        },
        {
            "name": "Shorkie_Random_Init (U-Net Small)",
            "path": f"{exp_root}/supervised_unet_small_bert_drop",
            "color": "black"
        },
        {
            "name": "Shorkie",
            "path": f"{exp_root}/self_supervised_unet_small_bert_drop",
            "color": "#ff7f0e" # orange
        },
        # {
        #     "name": "U-Net Big",
        #     "path": f"{variants_root}/unet_big",
        #     "color": "#2ca02c" # green
        # }
    ]
    
    # Load data
    metrics = ["train_loss", "train_r", "train_r2", "valid_loss", "valid_r", "valid_r2"]
    # Initialize storage: list (models) of lists (folds) of arrays
    all_data = {m: [] for m in metrics} 
    
    for m_def in models:
        path_base = m_def["path"]
        
        # Get params from fold 0 log
        log_path_f0 = f"{path_base}/train/f0c0/train.out"
        params_str = get_model_params(log_path_f0)
        m_def["params_str"] = params_str
        
        print(f"Loading {m_def['name']} (Params: {params_str}) from {path_base}...")
        
        # Temp lists for this model's folds
        m_folds_data = {m: [] for m in metrics}
        
        for fold in range(8):
            log_path = f"{path_base}/train/f{fold}c0/train.out"
            t_loss, t_r, t_r2, v_loss, v_r, v_r2 = parse_all_metrics(log_path)
            
            # Use empty array if missing, to keep index alignment consistent
            if len(t_loss) == 0:
                print(f"  Empty or missing log for fold {fold}: {log_path}")
                # Append empty arrays
                m_folds_data["train_loss"].append(np.array([]))
                m_folds_data["train_r"].append(np.array([]))
                m_folds_data["train_r2"].append(np.array([]))
                m_folds_data["valid_loss"].append(np.array([]))
                m_folds_data["valid_r"].append(np.array([]))
                m_folds_data["valid_r2"].append(np.array([]))
                continue
                
            m_folds_data["train_loss"].append(t_loss)
            m_folds_data["train_r"].append(t_r)
            m_folds_data["train_r2"].append(t_r2)
            m_folds_data["valid_loss"].append(v_loss)
            m_folds_data["valid_r"].append(v_r)
            m_folds_data["valid_r2"].append(v_r2)
            
        # Append this model's list of arrays to the global store
        for k in metrics:
            all_data[k].append(m_folds_data[k])

    # Align and plot
    # Update names to include params
    model_names = [f"{m['name']}\nParams: {m['params_str']}" for m in models]
    model_colors = [m["color"] for m in models]
    
    metric_display_names = {
        "train_loss": "Train Loss",
        "valid_loss": "Validation Loss",
        "train_r": "Train Pearson's R",
        "valid_r": "Validation Pearson's R",
        "train_r2": "Train R²",
        "valid_r2": "Validation R²"
    }

    for metric_key in metrics:
        print(f"Processing {metric_key}...")
        # all_data[metric_key] is a list of lists of arrays: [ [model1_fold1, ...], [model2_fold1, ...], ... ]
        
        # Check if we have any valid data for this metric across any model
        has_data = any(any(len(arr) > 0 for arr in model_folds) for model_folds in all_data[metric_key])
        if not has_data:
            print(f"  Skipping {metric_key}: No data found.")
            continue
            
        aligned_models_data, max_len = align_arrays(all_data[metric_key])
        
        is_loss = "loss" in metric_key
        
        plot_multi_model_metric(
            aligned_models_data,
            model_names,
            max_len,
            metric_name=metric_display_names[metric_key],
            is_loss=is_loss,
            out_filename=os.path.join(save_root, f"compare_avg_{metric_key}.png"),
            colors=model_colors
        )

if __name__ == "__main__":
    main()
