from optparse import OptionParser
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Paths & model definitions
# ---------------------------
steps = 500
n_epochs = 1030

# ---------------------------
# 2. Parsing function for each log file
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
                # Example line:
                # "Epoch 1 - train_loss: 0.5 - train_r: 0.45 - train_r2: 0.20 - valid_loss: 0.55 - valid_r: 0.43 - valid_r2: 0.18"
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


# ---------------------------
# 3. Align by maximum length
# ---------------------------
def pad_array_last_value(arr, desired_length):
    """
    If an array arr is shorter than desired_length, pad it to desired_length
    using its last value.
    """
    current_length = len(arr)
    if current_length < desired_length:
        pad_length = desired_length - current_length
        last_value = arr[-1]
        return np.concatenate([arr, np.full(pad_length, last_value)])
    return arr

def align_list_of_lists_by_max_len(all_models_data):
    """
    all_models_data: List of List of Arrays.
      - Outer list: N models
      - Inner list: K folds per model (each element is an array of values over epochs)
    
    Returns:
      (aligned_data, max_len)
      aligned_data: List of numpy arrays (N models), each shape (K_folds, max_len)
    """
    # Find global max length across all models and all folds
    max_len = 0
    for model_folds in all_models_data:
        for folder_arr in model_folds:
            if len(folder_arr) > max_len:
                max_len = len(folder_arr)
    
    aligned_data = []
    for model_folds in all_models_data:
        padded_folds = [pad_array_last_value(arr, max_len) for arr in model_folds]
        aligned_data.append(np.vstack(padded_folds))
        
    return aligned_data, max_len


# ---------------------------
# 4. Helper function: plot aggregator
# ---------------------------
def plot_metric_comparison_various_mlm(
    models_data,        # List of arrays (shape: n_folds, max_epochs)
    model_labels,       # List of strings
    model_colors,       # List of colors
    model_styles,       # List of linestyles
    aligned_length,
    metric_name,
    out_filename=None
):
    """
    Creates a single plot with mean ± std curves for multiple models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(aligned_length)

    for i, data in enumerate(models_data):
        mean_val = np.mean(data, axis=0)
        std_val  = np.std(data, axis=0)
        
        final_val = mean_val[-1]
        final_std = std_val[-1]
        
        label_str = f"{model_labels[i]} (final = {final_val:.3f} ± {final_std:.3f})"
        
        ax.plot(x, mean_val, linewidth=2, label=label_str, 
                color=model_colors[i], linestyle=model_styles[i])
        ax.fill_between(x, mean_val - std_val, mean_val + std_val,
                        color=model_colors[i], alpha=0.1)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_title(f"Model Comparison: {metric_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_filename:
        plt.savefig(out_filename, dpi=300)
        print(f"Saved plot to {out_filename}")
    # plt.show()


def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--model_arch', dest='model_arch', default='unet_small_bert_drop', type='str',
                      help='Model architecture[Default: %default]')
    parser.add_option('--root_dir', dest='root_dir', default='../../../Yeast_ML', type='str',
                      help='Root directory for Yeast_ML data')
    parser.add_option('--out_dir', dest='out_dir', default='./results', type='str',
                      help='Output directory')
    (options, args) = parser.parse_args()

    save_root = os.path.join(options.out_dir, f'{options.model_arch}_various_mlm')
    os.makedirs(save_root, exist_ok=True)

    exp_root = f"{options.root_dir}/seq_experiment"
    base_path_16bp = f"{exp_root}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
    
    # Define the 5 models
    # Each entry: (Label, PathTemplate, Color, Style)
    # PathTemplate needs {fold} to be replaced by f0c0, f1c0, etc.
    
    # 1. Supervised
    p1 = f"{base_path_16bp}/supervised_{options.model_arch}/train/{{fold}}/train.out"
    
    # 2. Shorkie (fine-tuned)
    p2 = f"{base_path_16bp}/self_supervised_{options.model_arch}/train/{{fold}}/train.out"
    
    # 3. 80_strains_mlm
    p3 = f"{base_path_16bp}/self_supervised_{options.model_arch}_all_mlm_load/80_strains_mlm/train/{{fold}}/train.out"
    
    # 4. 1341_fungal_mlm
    p4 = f"{base_path_16bp}/self_supervised_{options.model_arch}_all_mlm_load/1341_fungal_mlm/train/{{fold}}/train.out"
    
    # 5. yeast_mlm
    p5 = f"{base_path_16bp}/self_supervised_{options.model_arch}_all_mlm_load/yeast_mlm/train/{{fold}}/train.out"

    model_configs = [
        {
            "label": "Shorkie_Random_Init",
            "path_template": p1,
            "color": "#1f77b4", # Blue
            "style": "-"
        },
        {
            "label": "Shorkie",
            "path_template": p2,
            "color": "#ff7f0e", # Orange
            "style": "--"
        },
        {
            "label": "80 Strain MLM",
            "path_template": p3,
            "color": "#2ca02c", # Green
            "style": "-."
        },
        {
            "label": "1341 Fungal MLM",
            "path_template": p4,
            "color": "#d62728", # Red
            "style": ":"
        },
        {
            "label": "R64 Yeast MLM",
            "path_template": p5,
            "color": "#9467bd", # Purple
            "style": "-"
        }
    ]

    folds = [f"f{i}c0" for i in range(8)]

    # Containers for data: list of lists (Models -> Folds)
    data_train_loss = []
    data_train_r = []
    data_train_r2 = []
    data_valid_loss = []
    data_valid_r = []
    data_valid_r2 = []

    print("Reading data...")
    for config in model_configs:
        print(f"  Processing {config['label']}...")
        model_tr_loss = []
        model_tr_r = []
        model_tr_r2 = []
        model_vl_loss = []
        model_vl_r = []
        model_vl_r2 = []
        
        for fold in folds:
            file_path = config["path_template"].format(fold=fold)
            if not os.path.exists(file_path):
                print(f"    Warning: File not found: {file_path}")
                # We handle missing file by skipping or inserting empty? 
                # Ideally we assume files exist. If not, this might crash later or act empty.
                # Let's create empty arrays to prevent crash, but length 0.
                model_tr_loss.append(np.array([]))
                model_tr_r.append(np.array([]))
                model_tr_r2.append(np.array([]))
                model_vl_loss.append(np.array([]))
                model_vl_r.append(np.array([]))
                model_vl_r2.append(np.array([]))
                continue

            (tr_l, tr_r, tr_r2, vl_l, vl_r, vl_r2) = parse_all_metrics(file_path)
            model_tr_loss.append(tr_l)
            model_tr_r.append(tr_r)
            model_tr_r2.append(tr_r2)
            model_vl_loss.append(vl_l)
            model_vl_r.append(vl_r)
            model_vl_r2.append(vl_r2)
            
        data_train_loss.append(model_tr_loss)
        data_train_r.append(model_tr_r)
        data_train_r2.append(model_tr_r2)
        data_valid_loss.append(model_vl_loss)
        data_valid_r.append(model_vl_r)
        data_valid_r2.append(model_vl_r2)

    # Align Data
    print("Aligning data...")
    aligned_tr_loss, max_tr_l = align_list_of_lists_by_max_len(data_train_loss)
    aligned_tr_r, max_tr_r    = align_list_of_lists_by_max_len(data_train_r)
    aligned_tr_r2, max_tr_r2  = align_list_of_lists_by_max_len(data_train_r2)
    aligned_vl_loss, max_vl_l = align_list_of_lists_by_max_len(data_valid_loss)
    aligned_vl_r, max_vl_r    = align_list_of_lists_by_max_len(data_valid_r)
    aligned_vl_r2, max_vl_r2  = align_list_of_lists_by_max_len(data_valid_r2)

    # Plot
    print("Plotting...")
    labels = [c["label"] for c in model_configs]
    colors = [c["color"] for c in model_configs]
    styles = [c["style"] for c in model_configs]

    # Train Loss
    plot_metric_comparison_various_mlm(
        aligned_tr_loss, labels, colors, styles, max_tr_l,
        "Train Loss", os.path.join(save_root, "comparison_train_loss.png")
    )
    # Valid Loss
    plot_metric_comparison_various_mlm(
        aligned_vl_loss, labels, colors, styles, max_vl_l,
        "Validation Loss", os.path.join(save_root, "comparison_valid_loss.png")
    )
    # Train R
    plot_metric_comparison_various_mlm(
        aligned_tr_r, labels, colors, styles, max_tr_r,
        "Train Pearson's R", os.path.join(save_root, "comparison_train_r.png")
    )
    # Valid R
    plot_metric_comparison_various_mlm(
        aligned_vl_r, labels, colors, styles, max_vl_r,
        "Validation Pearson's R", os.path.join(save_root, "comparison_valid_r.png")
    )
    # Train R2
    plot_metric_comparison_various_mlm(
        aligned_tr_r2, labels, colors, styles, max_tr_r2,
        "Train R²", os.path.join(save_root, "comparison_train_r2.png")
    )
    # Valid R2
    plot_metric_comparison_various_mlm(
        aligned_vl_r2, labels, colors, styles, max_vl_r2,
        "Validation R²", os.path.join(save_root, "comparison_valid_r2.png")
    )

if __name__ == "__main__":
    main()
