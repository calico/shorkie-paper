from optparse import OptionParser
import os
import numpy as np
import matplotlib.pyplot as plt


def _moving_average(x, window_size=1, trim_end=0):
    """
    Compute the moving average of x with the specified window_size,
    then trim the last trim_end points (if trim_end > 0).
    """
    x_avg = np.zeros_like(x, dtype=float)
    for j in range(x.shape[0]):
        min_j = max(j - window_size // 2, 0)
        max_j = min(j + window_size // 2, x.shape[0] - 1)
        actual_window_size = min(j - min_j, max_j - j)
        min_j = j - actual_window_size
        max_j = j + actual_window_size
        x_avg[j] = np.mean(x[min_j:max_j + 1])
    if trim_end > 0 and trim_end < x.shape[0]:
        return x_avg[:-trim_end]
    else:
        return x_avg


def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--root_dir', dest='root_dir', default='../../../Yeast_ML', type='str',
                      help='Root directory for Yeast_ML data')
    parser.add_option('--out_dir', dest='out_dir', default='./results', type='str',
                      help='Output directory')
    parser.add_option('--model_arch', dest='model_arch', default='unet_small_bert_drop', type='str',
                      help='Model architecture[Default: %default]')
    (options, args) = parser.parse_args()

    # ------------------
    # Global Hyper-params
    # ------------------
    steps = 500
    n_epochs = 1030 
    # Adjusted to match the first script which had 1030, though the original 3_...py had 730/900.
    # We will use a safe large number or rely on actual data length.

    trim_end = 5
    window_size = 11
    min_iters = steps
    # We can set max_iters dynamic or fixed.
    max_iters = n_epochs * steps

    save_root = os.path.join(options.out_dir, f'{options.model_arch}_various_mlm_subplots')
    os.makedirs(save_root, exist_ok=True)

    base_path_16bp = f"{options.root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"

    # -------------------------------------------------------------
    # 1) Define the datasets (5 groups x 8 folds)
    # -------------------------------------------------------------
    
    # Template paths
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

    groups = [
        {"name": "Shorkie_Random_Init", "templ": p1},
        {"name": "Shorkie", "templ": p2},
        {"name": "80 Strain MLM", "templ": p3},
        {"name": "1341 Fungal MLM", "templ": p4},
        {"name": "R64 Yeast MLM", "templ": p5},
    ]

    model_files = []
    # Build list of 40 files: Group 1 (f0..f7), then Group 2 (f0..f7), etc.
    for g in groups:
        for i in range(8):
            fold = f"f{i}c0"
            model_files.append(g["templ"].format(fold=fold))

    # -------------------------------------------------------------
    # 2) Configs
    # -------------------------------------------------------------
    
    group_labels = [g["name"] for g in groups]
    # Colors matching the first script
    group_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    group_linestyles = ["-", "--", "-.", ":", "-"]

    # Each file's line prefix and step multiple
    model_prefixes = ["Epoch "] * len(model_files)
    model_steps = [steps] * len(model_files)

    # -------------------------------------------------------------
    # 3) Read all metrics from each model file
    # -------------------------------------------------------------
    train_losses, train_rs, train_r2s = [], [], []
    valid_losses, valid_rs, valid_r2s = [], [], []

    print("Reading data...")
    for idx, model_file in enumerate(model_files):
        prefix = model_prefixes[idx]
        local_train_loss = []
        local_train_r    = []
        local_train_r2   = []
        local_valid_loss = []
        local_valid_r    = []
        local_valid_r2   = []

        if os.path.exists(model_file):
            with open(model_file, 'r') as f:
                for line_raw in f:
                    line = line_raw.strip()
                    if line.startswith(prefix):
                        parts = line.split(" - ")
                        try:
                            t_loss_val = float(parts[2].split(": ")[1])
                            t_r_val    = float(parts[3].split(": ")[1])
                            t_r2_val   = float(parts[4].split(": ")[1])
                            v_loss_val = float(parts[5].split(": ")[1])
                            v_r_val    = float(parts[6].split(": ")[1])
                            v_r2_val   = float(parts[7].split(": ")[1])
                        except (IndexError, ValueError) as e:
                            # print(f"Error parsing line: {line}\n{e}")
                            continue

                        local_train_loss.append(t_loss_val)
                        local_train_r.append(t_r_val)
                        local_train_r2.append(t_r2_val)
                        local_valid_loss.append(v_loss_val)
                        local_valid_r.append(v_r_val)
                        local_valid_r2.append(v_r2_val)
        else:
            print(f"Warning: File not found {model_file}")

        train_losses.append(np.array(local_train_loss))
        train_rs.append(np.array(local_train_r))
        train_r2s.append(np.array(local_train_r2))
        valid_losses.append(np.array(local_valid_loss))
        valid_rs.append(np.array(local_valid_r))
        valid_r2s.append(np.array(local_valid_r2))

    # -------------------------------------------------------------
    # 4) Plotting
    # -------------------------------------------------------------
    metric_list = ["loss", "r", "r2"]
    metric_label = ["Loss", "Pearson's R", "R²"]
    eval_types   = ["min", "max", "max"]

    def get_index_in_arrays(group_id, fold_id):
        return group_id * 8 + fold_id

    for m_idx, m_name in enumerate(metric_list):
        train_data = [train_losses, train_rs, train_r2s][m_idx]
        valid_data = [valid_losses, valid_rs, valid_r2s][m_idx]
        m_lab  = metric_label[m_idx]
        e_type = eval_types[m_idx]

        # ========= (A) Training figure ============
        fig_train, axes_train = plt.subplots(nrows=2, ncols=4, figsize=(20, 8), dpi=300, sharey=True)
        fig_train.suptitle(f"Model Comparison (Training {m_lab})", fontsize=22)

        for fold_i in range(8):
            ax = axes_train[fold_i // 4, fold_i % 4]
            ax.set_title(f"Fold {fold_i}", fontsize=12)
            
            for group_id in range(5):
                idx_data = get_index_in_arrays(group_id, fold_i)
                if idx_data >= len(train_data): continue
                
                tvals = train_data[idx_data]
                if len(tvals) == 0:
                    continue
                
                smoothed = _moving_average(tvals, window_size=window_size, trim_end=trim_end)
                steps_arr = (np.arange(len(tvals)) + 1) * model_steps[idx_data]
                steps_arr_smooth = steps_arr[:len(smoothed)]
                
                if e_type == "min":
                    best_val = np.min(smoothed)
                else:
                    best_val = np.max(smoothed)
                
                # Shorten label for legend space
                g_lbl = group_labels[group_id]
                
                ax.plot(
                    steps_arr_smooth,
                    smoothed,
                    color=group_colors[group_id],
                    linestyle=group_linestyles[group_id],
                    linewidth=1.5,
                    label=f"{g_lbl} ({best_val:.3f})"
                )

            if fold_i // 4 == 1:
                ax.set_xlabel("# Batches", fontsize=10)
            if fold_i % 4 == 0:
                ax.set_ylabel(f"Train {m_lab}", fontsize=10)
            
            # Simple Legend
            if fold_i == 0:
                 ax.legend(fontsize=8, loc='best')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_root, f"train_{m_name}.png"))
        # plt.show()
        plt.close(fig_train)

        # ========= (B) Validation figure ============
        fig_valid, axes_valid = plt.subplots(nrows=2, ncols=4, figsize=(20, 8), dpi=300, sharey=True)
        fig_valid.suptitle(f"Model Comparison (Validation {m_lab})", fontsize=22)

        for fold_i in range(8):
            ax = axes_valid[fold_i // 4, fold_i % 4]
            ax.set_title(f"Fold {fold_i}", fontsize=12)
            
            for group_id in range(5):
                idx_data = get_index_in_arrays(group_id, fold_i)
                if idx_data >= len(valid_data): continue
                
                vvals = valid_data[idx_data]
                if len(vvals) == 0:
                    continue
                
                smoothed = _moving_average(vvals, window_size=window_size, trim_end=trim_end)
                steps_arr = (np.arange(len(vvals)) + 1) * model_steps[idx_data]
                steps_arr_smooth = steps_arr[:len(smoothed)]
                
                if e_type == "min":
                    best_val = np.min(smoothed)
                else:
                    best_val = np.max(smoothed)
                
                g_lbl = group_labels[group_id]

                ax.plot(
                    steps_arr_smooth,
                    smoothed,
                    color=group_colors[group_id],
                    linestyle=group_linestyles[group_id],
                    linewidth=1.5,
                    label=f"{g_lbl} ({best_val:.3f})"
                )

            if fold_i // 4 == 1:
                ax.set_xlabel("# Batches", fontsize=10)
            if fold_i % 4 == 0:
                ax.set_ylabel(f"Valid {m_lab}", fontsize=10)
            
            if fold_i == 0:
                ax.legend(fontsize=8, loc='best')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_root, f"valid_{m_name}.png"))
        # plt.show()
        plt.close(fig_valid)

if __name__ == "__main__":
    main()
