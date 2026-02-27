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
    parser.add_option('--data_dir', dest='data_dir', default='', type='str',
                      help='Data directory [Default: %default]')
    (options, args) = parser.parse_args()

    # ------------------
    # Global Hyper-params
    # ------------------
    steps = 500
    n_epochs = 1050

    trim_end = 5
    window_size = 11
    min_iters = steps
    max_iters = n_epochs * steps

    save_root = './results'
    os.makedirs(save_root, exist_ok=True)

    # -------------------------------------------------------------
    # 1) Define the datasets (each with 8 folds) you want to plot.
    #    We'll have 5 groups:
    #      Group A: Supervised U-Net small (8 folds)
    #      Group B: Fine-tuned U-Net small (8 folds)
    #      Group C: Supervised UNet Params (8 folds)
    #      Group D: Supervised UNet Smaller Params (8 folds)
    #      Group E: Default Adam (finetuned_default_adam, 8 folds)
    # -------------------------------------------------------------
    # Groups A + B: Model A (Supervised) and Model B (Fine-tuned)
    model_names_AB = [
        # Group A: Supervised U-Net small (8 folds)
        'Supervised 16bp U-Net small F0',
        'Supervised 16bp U-Net small F1',
        'Supervised 16bp U-Net small F2',
        'Supervised 16bp U-Net small F3',
        'Supervised 16bp U-Net small F4',
        'Supervised 16bp U-Net small F5',
        'Supervised 16bp U-Net small F6',
        'Supervised 16bp U-Net small F7',
        # Group B: Fine-tuned U-Net small (8 folds)
        'Fine-tuned LM 16bp U-Net small F0',
        'Fine-tuned LM 16bp U-Net small F1',
        'Fine-tuned LM 16bp U-Net small F2',
        'Fine-tuned LM 16bp U-Net small F3',
        'Fine-tuned LM 16bp U-Net small F4',
        'Fine-tuned LM 16bp U-Net small F5',
        'Fine-tuned LM 16bp U-Net small F6',
        'Fine-tuned LM 16bp U-Net small F7',
    ]
    model_files_AB = [
        # Group A: supervised U-Net small
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f0c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f1c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f2c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f3c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f4c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f5c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f6c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f7c0/train.out',
        # Group B: fine-tuned U-Net small
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f0c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f1c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f2c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f3c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f4c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f5c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f6c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f7c0/train.out',
    ]

    # Group C: Supervised UNet Params
    unet_params_names_C = [
        "Supervised UNet Params F0",
        "Supervised UNet Params F1",
        "Supervised UNet Params F2",
        "Supervised UNet Params F3",
        "Supervised UNet Params F4",
        "Supervised UNet Params F5",
        "Supervised UNet Params F6",
        "Supervised UNet Params F7",
    ]
    unet_params_files_C = [
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f0c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f1c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f2c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f3c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f4c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f5c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f6c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_params/train/f7c0/train.out',
    ]

    # Group D: Supervised UNet Smaller Params
    unet_smaller_names_D = [
        "Supervised UNet Smaller Params F0",
        "Supervised UNet Smaller Params F1",
        "Supervised UNet Smaller Params F2",
        "Supervised UNet Smaller Params F3",
        "Supervised UNet Smaller Params F4",
        "Supervised UNet Smaller Params F5",
        "Supervised UNet Smaller Params F6",
        "Supervised UNet Smaller Params F7",
    ]
    unet_smaller_files_D = [
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f0c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f1c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f2c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f3c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f4c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f5c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f6c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_smaller_params/train/f7c0/train.out',
    ]

    # Group E: Default Adam (finetuned_default_adam)
    default_adam_names_E = [
        "Default Adam F0",
        "Default Adam F1",
        "Default Adam F2",
        "Default Adam F3",
        "Default Adam F4",
        "Default Adam F5",
        "Default Adam F6",
        "Default Adam F7",
    ]
    default_adam_files_E = [
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f0c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f1c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f2c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f3c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f4c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f5c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f6c0/train.out',
        f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_params_new/train/f7c0/train.out',
    ]

    # -------------------------------------------------------------
    # 2) Combine them in single arrays for reading/plotting
    #    => total 40 entries
    # -------------------------------------------------------------
    model_names = model_names_AB + unet_params_names_C + unet_smaller_names_D + default_adam_names_E
    model_files = model_files_AB + unet_params_files_C + unet_smaller_files_D + default_adam_files_E

    # We'll define which "group" each set belongs to for color/linestyle.
    # Groups: 0 => A, 1 => B, 2 => C, 3 => D, 4 => E
    group_labels = [
        "Supervised U-Net small",
        "Fine-tuned U-Net small",
        "Supervised UNet Exp1",
        "Supervised UNet Smaller Exp2",
        "Fine-tuned U-Net small default adam",
    ]
    group_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Blue, Orange, Green, Red, Purple
    # group_linestyles = ["-", "--", "-.", ":", "-"]
    group_linestyles = ["-", "--", "-", "-", "--"]

    # Each file's line prefix and step multiple
    model_prefixes = ["Epoch "] * len(model_files)
    model_steps = [steps] * len(model_files)

    # -------------------------------------------------------------
    # 3) Read all metrics from each model file
    # -------------------------------------------------------------
    train_losses, train_rs, train_r2s = [], [], []
    valid_losses, valid_rs, valid_r2s = [], [], []

    for idx, model_file in enumerate(model_files):
        prefix = model_prefixes[idx]
        local_train_loss = []
        local_train_r    = []
        local_train_r2   = []
        local_valid_loss = []
        local_valid_r    = []
        local_valid_r2   = []

        with open(model_file, 'r') as f:
            for line_raw in f:
                line = line_raw.strip()
                if line.startswith(prefix):
                    # Expected format:
                    # "Epoch X - train_loss: # - train_r: # - train_r2: # - valid_loss: # - valid_r: # - valid_r2: #"
                    parts = line.split(" - ")
                    try:
                        # parts[0] = 'Epoch X'
                        # parts[1] = 'train_loss: #'
                        # parts[2] = 'train_r: #'
                        # parts[3] = 'train_r2: #'
                        # parts[4] = 'valid_loss: #'
                        # parts[5] = 'valid_r: #'
                        # parts[6] = 'valid_r2: #'
                        t_loss_val = float(parts[2].split(": ")[1])
                        t_r_val    = float(parts[3].split(": ")[1])
                        t_r2_val   = float(parts[4].split(": ")[1])
                        v_loss_val = float(parts[5].split(": ")[1])
                        v_r_val    = float(parts[6].split(": ")[1])
                        v_r2_val   = float(parts[7].split(": ")[1])
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing line: {line}\n{e}")
                        continue

                    local_train_loss.append(t_loss_val)
                    local_train_r.append(t_r_val)
                    local_train_r2.append(t_r2_val)
                    local_valid_loss.append(v_loss_val)
                    local_valid_r.append(v_r_val)
                    local_valid_r2.append(v_r2_val)

        train_losses.append(np.array(local_train_loss))
        train_rs.append(np.array(local_train_r))
        train_r2s.append(np.array(local_train_r2))
        valid_losses.append(np.array(local_valid_loss))
        valid_rs.append(np.array(local_valid_r))
        valid_r2s.append(np.array(local_valid_r2))

    # We'll define the "evaluation type" for each metric
    # (i.e., 'loss' => min is better; 'r' and 'r2' => max is better)
    metric_list = ["loss", "r", "r2"]
    metric_label = ["Loss", "Pearson's R", "R²"]
    eval_types   = ["min", "max", "max"]

    # A helper to map (fold_i) => index in train_/valid_ arrays.
    # Group A => fold_i (indices 0–7)
    # Group B => fold_i + 8   (indices 8–15)
    # Group C => fold_i + 16  (indices 16–23)
    # Group D => fold_i + 24  (indices 24–31)
    # Group E => fold_i + 32  (indices 32–39)
    def get_index_in_arrays(group_id, fold_id):
        return group_id * 8 + fold_id

    # -------------------------------------------------------------
    # 4) Plot: For each metric, create one figure for train and one for valid.
    #    Each figure uses a 2x4 grid (8 subplots, one per fold).
    # -------------------------------------------------------------
    for m_idx, m_name in enumerate(metric_list):
        train_data = [train_losses, train_rs, train_r2s][m_idx]
        valid_data = [valid_losses, valid_rs, valid_r2s][m_idx]
        m_lab  = metric_label[m_idx]
        e_type = eval_types[m_idx]

        # ========= (A) Training figure with 2x4 subplots ============
        fig_train, axes_train = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=150)
        fig_train.suptitle(f"Training {m_lab}", fontsize=18)

        for fold_i in range(8):
            ax = axes_train[fold_i // 4, fold_i % 4]
            ax.set_title(f"Fold {fold_i}", fontsize=12)
            # Plot the 5 lines (one for each group: A, B, C, D, E)
            for group_id in range(5):
                idx_data = get_index_in_arrays(group_id, fold_i)
                tvals = train_data[idx_data]
                if len(tvals) == 0:
                    continue
                # Smoothing
                smoothed = _moving_average(tvals, window_size=window_size, trim_end=trim_end)
                # Steps: each epoch multiplied by the corresponding step value
                steps_arr = (np.arange(len(tvals)) + 1) * model_steps[idx_data]
                steps_arr_smooth = steps_arr[:len(smoothed)]
                # Best value evaluation
                if e_type == "min":
                    best_val = np.min(smoothed)
                    orig_best_val = np.min(tvals[:-trim_end]) if len(tvals) > trim_end else np.min(tvals)
                else:
                    best_val = np.max(smoothed)
                    orig_best_val = np.max(tvals[:-trim_end]) if len(tvals) > trim_end else np.max(tvals)
                label_str = f"{group_labels[group_id]} | {m_lab}={best_val:.4f} ({orig_best_val:.4f})"
                ax.plot(
                    steps_arr_smooth,
                    smoothed,
                    color=group_colors[group_id],
                    linestyle=group_linestyles[group_id],
                    linewidth=1.8,
                    label=label_str
                )
            if fold_i // 4 == 1:
                ax.set_xlabel("# Batches", fontsize=10)
            if fold_i % 4 == 0:
                ax.set_ylabel(f"Train {m_lab}", fontsize=10)
            if min_iters is not None and max_iters is not None:
                ax.set_xlim(min_iters, max_iters)
            ax.legend(fontsize=7)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(save_root, f"train_{m_name}.png"))
        plt.show()

        # ========= (B) Validation figure with 2x4 subplots ============
        fig_valid, axes_valid = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), dpi=150)
        fig_valid.suptitle(f"Validation {m_lab}", fontsize=18)

        for fold_i in range(8):
            ax = axes_valid[fold_i // 4, fold_i % 4]
            ax.set_title(f"Fold {fold_i}", fontsize=12)
            for group_id in range(5):
                idx_data = get_index_in_arrays(group_id, fold_i)
                vvals = valid_data[idx_data]
                if len(vvals) == 0:
                    continue
                smoothed = _moving_average(vvals, window_size=window_size, trim_end=trim_end)
                steps_arr = (np.arange(len(vvals)) + 1) * model_steps[idx_data]
                steps_arr_smooth = steps_arr[:len(smoothed)]
                if e_type == "min":
                    best_val = np.min(smoothed)
                    orig_best_val = np.min(vvals[:-trim_end]) if len(vvals) > trim_end else np.min(vvals)
                else:
                    best_val = np.max(smoothed)
                    orig_best_val = np.max(vvals[:-trim_end]) if len(vvals) > trim_end else np.max(vvals)
                label_str = f"{group_labels[group_id]} | {m_lab}={best_val:.4f} ({orig_best_val:.4f})"
                ax.plot(
                    steps_arr_smooth,
                    smoothed,
                    color=group_colors[group_id],
                    linestyle=group_linestyles[group_id],
                    linewidth=1.8,
                    label=label_str
                )
            if fold_i // 4 == 1:
                ax.set_xlabel("# Batches", fontsize=10)
            if fold_i % 4 == 0:
                ax.set_ylabel(f"Valid {m_lab}", fontsize=10)
            if min_iters is not None and max_iters is not None:
                ax.set_xlim(min_iters, max_iters)
            ax.legend(fontsize=7)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(save_root, f"valid_{m_name}.png"))
        plt.show()


if __name__ == "__main__":
    main()
