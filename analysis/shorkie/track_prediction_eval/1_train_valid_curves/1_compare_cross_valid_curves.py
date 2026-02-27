from optparse import OptionParser
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Option Parsing
# ---------------------------
usage = 'usage: %prog [options] arg'
parser = OptionParser(usage)
parser.add_option('--data_dir', dest='data_dir', default='', type='str', help='Data directory [Default: %default]')
(options, args) = parser.parse_args()

# ---------------------------
# Global Parameters
# ---------------------------
steps = 500
n_epochs = 1050
trim_end = 5
window_size = 11
plot_train_loss = True

min_iters = steps
max_iters = n_epochs * steps

save_root = f'./results'
os.makedirs(save_root, exist_ok=True) 

# ---------------------------
# Model names, files, colors, linestyles, etc.
# ---------------------------
model_names = [
    'supervised 16bp U-Net small F0',
    'supervised 16bp U-Net small F1',
    'supervised 16bp U-Net small F2',
    'supervised 16bp U-Net small F3',
    'supervised 16bp U-Net small F4',
    'supervised 16bp U-Net small F5',
    'supervised 16bp U-Net small F6',
    'supervised 16bp U-Net small F7',

    'Fine-tuned LM 16bp U-Net small F0',
    'Fine-tuned LM 16bp U-Net small F1',
    'Fine-tuned LM 16bp U-Net small F2',
    'Fine-tuned LM 16bp U-Net small F3',
    'Fine-tuned LM 16bp U-Net small F4',
    'Fine-tuned LM 16bp U-Net small F5',
    'Fine-tuned LM 16bp U-Net small F6',
    'Fine-tuned LM 16bp U-Net small F7',
]

model_files = [
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f0c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f1c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f2c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f3c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f4c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f5c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f6c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_unet_small_bert_drop/train/f7c0/train.out',

    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f0c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f1c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f2c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f3c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f4c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f5c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f6c0/train.out',
    f'{options.data_dir}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_unet_small_bert_drop/train/f7c0/train.out',
]

model_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

linestyles = [
    '-', '-', '-', '-', '-', '-', '-', '-', 
    '--', '--', '--', '--', '--', '--', '--', '--',
]

model_datastrs = [
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
]

model_steps = [
    steps, steps, steps, steps,
    steps, steps, steps, steps,
    steps, steps, steps, steps,
    steps, steps, steps, steps,
]

# ---------------------------
# Load & Parse Training Metrics
# ---------------------------
train_losses = []
train_rs = []
train_r2s = []

valid_losses = []
valid_rs = []
valid_r2s = []

# Loop over model metric files
for model_i, model_file in enumerate(model_files):
    train_loss = []
    train_r = []
    train_r2 = []
    
    valid_loss = []
    valid_r = []
    valid_r2 = []

    with open(model_file, 'rt') as f:
        for line_raw in f.readlines():
            line = line_raw.strip()
            # Assuming each line starts with model_datastrs[model_i]
            if line[:6] == model_datastrs[model_i]:
                # Split the line by " - " and extract metrics
                line_parts = line.split(" - ")
                train_loss.append(float(line_parts[2].split(": ")[1]))
                train_r.append(float(line_parts[3].split(": ")[1]))
                train_r2.append(float(line_parts[4].split(": ")[1]))
                valid_loss.append(float(line_parts[5].split(": ")[1]))
                valid_r.append(float(line_parts[6].split(": ")[1]))
                valid_r2.append(float(line_parts[7].split(": ")[1]))

    train_losses.append(np.array(train_loss))
    train_rs.append(np.array(train_r))
    train_r2s.append(np.array(train_r2))
    
    valid_losses.append(np.array(valid_loss))
    valid_rs.append(np.array(valid_r))
    valid_r2s.append(np.array(valid_r2))

# ---------------------------
# Moving Average Function (existing)
# ---------------------------
def _moving_average(x, window_size=1, trim_end=0):
    x_avg = np.zeros(x.shape)
    for j in range(x.shape[0]):
        min_j = max(j - window_size // 2, 0)
        max_j = min(j + window_size // 2, x.shape[0] - 1)
        
        actual_window_size = min(j - min_j, max_j - j)
        min_j = j - actual_window_size
        max_j = j + actual_window_size

        x_avg[j] = np.mean(x[min_j:max_j+1])
    return x_avg[:-trim_end] if trim_end > 0 else x_avg

# ---------------------------
# (A) Plot Individual Model Curves (as in your original script)
# ---------------------------
Fig_Size = (12, 10)
metrics_names = ["Loss", "Pearson's R", f"R\u00b2"]
for idx, metric in enumerate(["loss", "r", "r2"]):
    metrics_name = metrics_names[idx]
    eval_type = None
    if metric == "loss":
        train_metrics = train_losses
        valid_metrics = valid_losses
        eval_type = "min"
    elif metric == "r":
        train_metrics = train_rs
        valid_metrics = valid_rs
        eval_type = "max"
    elif metric == "r2":
        train_metrics = train_r2s
        valid_metrics = valid_r2s
        eval_type = "max"

    # Compute x-axis steps (common for all models)
    max_length = int(np.max([len(t) for t in train_metrics]))
    m_steps = [(np.arange(max_length) + 1) * model_steps[model_ix] for model_ix in range(len(train_metrics))]
    
    # Plot training curves for each model
    f_train = plt.figure(figsize=Fig_Size, dpi=600)
    for model_i, (model_name, model_color, linestyle, train_loss, steps_arr) in enumerate(zip(model_names, model_colors, linestyles, train_metrics, m_steps)):
        if eval_type == "min":
            val_mavg = _moving_average(train_loss, window_size=window_size, trim_end=trim_end)
            train_loss_val = np.min(val_mavg)
            train_loss_val_orig = np.min(train_loss[:-trim_end])
        else:
            val_mavg = _moving_average(train_loss, window_size=window_size, trim_end=trim_end)
            train_loss_val = np.max(val_mavg)
            train_loss_val_orig = np.max(train_loss[:-trim_end])
        plt.plot(steps_arr[:len(train_loss) - trim_end], val_mavg, color=model_color, linewidth=1.5, linestyle=linestyle, zorder=-1000,
                 label=model_name + f" (train); {metrics_name} = " + '{:.4f}'.format(train_loss_val) + " (" + '{:.4f}'.format(train_loss_val_orig) + ")")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if min_iters is not None and max_iters is not None:
        plt.xlim(min_iters, max_iters)
    plt.xlabel('# Training Batches', fontsize=15)
    plt.ylabel(f'Training {metrics_name}', fontsize=15)
    plt.legend(fontsize=10, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.title(f'Training {metrics_name}', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{save_root}/train_{metric}.png')
    plt.show()

    # Plot validation curves for each model
    f_valid = plt.figure(figsize=Fig_Size, dpi=600)
    for model_i, (model_name, model_color, linestyle, valid_loss, steps_arr) in enumerate(zip(model_names, model_colors, linestyles, valid_metrics, m_steps)):
        if eval_type == "min":
            val_mavg = _moving_average(valid_loss, window_size=window_size, trim_end=trim_end)
            valid_loss_val = np.min(val_mavg)
            valid_loss_val_orig = np.min(valid_loss[:-trim_end])
        else:
            val_mavg = _moving_average(valid_loss, window_size=window_size, trim_end=trim_end)
            valid_loss_val = np.max(val_mavg)
            valid_loss_val_orig = np.max(valid_loss[:-trim_end])
        plt.plot(steps_arr[:len(valid_loss) - trim_end], val_mavg, color=model_color, linewidth=2, linestyle=linestyle, zorder=-1000,
                 label=model_name + f" (valid); {metrics_name} = " + '{:.4f}'.format(valid_loss_val) + " (" + '{:.4f}'.format(valid_loss_val_orig) + ")")
        # Also plot a vertical line at the index of the best value
        if eval_type == "min":
            best_ix = np.argmin(valid_loss[:-trim_end])
        else:
            best_ix = np.argmax(valid_loss[:-trim_end])
        plt.axvline(x=steps_arr[best_ix], color=model_color, linewidth=1, linestyle='--', alpha=0.25, zorder=-1500)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if min_iters is not None and max_iters is not None:
        plt.xlim(min_iters, max_iters)
    plt.xlabel('# Training Batches', fontsize=15)
    plt.ylabel(f'Validation {metrics_name}', fontsize=15)
    plt.legend(fontsize=10, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.title(f'Validation {metrics_name}', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{save_root}/valid_{metric}.png')
    plt.clf()

# ---------------------------
# (B) New: Aggregated Plotting for Two Datasets (Supervised vs. Fine-tuned)
# ---------------------------
def process_moving_average(dataset_list, window_size, trim_end):
    """
    Applies moving average to each 1D array in dataset_list and aligns them to the smallest length.
    Returns a 2D numpy array (each row is one model's processed metric) and the common length.
    """
    processed = []
    for arr in dataset_list:
        processed.append(_moving_average(arr, window_size=window_size, trim_end=trim_end))
    min_len = min(len(p) for p in processed)
    processed_aligned = np.vstack([p[:min_len] for p in processed])
    return processed_aligned, min_len

def plot_aggregated_two_groups(group1, group2, common_len, step, metric_name, group_labels, out_filename=None, ylabel_prefix="Training"):
    """
    Plots two aggregated curves (mean ± std) on the same figure.
    
    Parameters:
      - group1, group2: 2D numpy arrays (each row is one sample, columns are time points)
      - common_len: number of time points (all arrays aligned to this length)
      - step: the multiplication factor for x-axis (e.g. steps)
      - metric_name: string for the metric (e.g. "Loss")
      - group_labels: list of two strings (e.g. ["Supervised", "Fine-tuned"])
      - out_filename: if provided, save the figure
      - ylabel_prefix: e.g. "Training" or "Validation"
    """
    x = (np.arange(common_len) + 1) * step
    mean1 = np.mean(group1, axis=0)
    std1 = np.std(group1, axis=0)
    mean2 = np.mean(group2, axis=0)
    std2 = np.std(group2, axis=0)
    
    plt.figure(figsize=(12, 8), dpi=600)
    plt.plot(x, mean1, linewidth=2, label=f"{group_labels[0]} (final = {mean1[-1]:.4f} ± {std1[-1]:.4f})")
    plt.fill_between(x, mean1 - std1, mean1 + std1, alpha=0.2)
    plt.plot(x, mean2, linewidth=2, linestyle='--', label=f"{group_labels[1]} (final = {mean2[-1]:.4f} ± {std2[-1]:.4f})")
    plt.fill_between(x, mean2 - std2, mean2 + std2, alpha=0.2)
    plt.xlabel('# Training Batches', fontsize=15)
    plt.ylabel(f'{ylabel_prefix} {metric_name}', fontsize=15)
    plt.title(f'{ylabel_prefix} {metric_name} (Aggregated)', fontsize=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename)
    plt.show()

# For each metric type, process and plot aggregated curves
# Group the models: first 8 (Supervised) and next 8 (Fine-tuned)

aggregated_info = {
    "loss": (train_losses, valid_losses, "Loss"),
    "r": (train_rs, valid_rs, "Pearson's R"),
    "r2": (train_r2s, valid_r2s, "R\u00b2")
}

# Aggregated training plots
for metric_key, (train_data, _, metric_str) in aggregated_info.items():
    # Process each group
    group1, min_len1 = process_moving_average(train_data[0:8], window_size, trim_end)
    group2, min_len2 = process_moving_average(train_data[8:16], window_size, trim_end)
    common_len = min(min_len1, min_len2)
    plot_aggregated_two_groups(
        group1[:, :common_len], group2[:, :common_len],
        common_len, steps, metric_str,
        group_labels=["Supervised", "Fine-tuned"],
        out_filename=f'{save_root}/aggregated_train_{metric_key}.png',
        ylabel_prefix="Training"
    )

# Aggregated validation plots
for metric_key, (_, valid_data, metric_str) in aggregated_info.items():
    group1, min_len1 = process_moving_average(valid_data[0:8], window_size, trim_end)
    group2, min_len2 = process_moving_average(valid_data[8:16], window_size, trim_end)
    common_len = min(min_len1, min_len2)
    plot_aggregated_two_groups(
        group1[:, :common_len], group2[:, :common_len],
        common_len, steps, metric_str,
        group_labels=["Supervised", "Fine-tuned"],
        out_filename=f'{save_root}/aggregated_valid_{metric_key}.png',
        ylabel_prefix="Validation"
    )
