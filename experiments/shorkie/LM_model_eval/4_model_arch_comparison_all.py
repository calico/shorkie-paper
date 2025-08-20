#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

FigSize = (5, 2.85)
# —— USER PARAMETERS ——
steps         = 64
trim_end      = 5
window_size   = 21
input_dir     = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI'
out_dir       = 'viz/model_arch_comparison/all_datasets'
os.makedirs(out_dir, exist_ok=True)

# Datasets + their epoch counts
datasets = ['r64', 'strains', 'saccharomycetales', 'fungi_1385']
dataset_epochs = {
    'r64': 23,
    'strains': 41,
    'saccharomycetales': 450,
    'fungi_1385': 350,
}

# Mapping for prettier labels
dataset_labels = {
    'r64':             'R64',
    'strains':         '80_strains',
    'saccharomycetales':'165_Saccharomycetales',
    'fungi_1385':      '1342_fungus',
}

# Model names + plotting styles/colors
model_names    = ['Conv_Small', 'Conv_Big', 'U-Net_Small', 'U-Net_Big']
dataset_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
# brighter colors for model curves in per‑dataset plots:
# model_colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
model_colors   = ['tab:purple', 'tab:olive', 'tab:cyan', 'tab:pink']
model_styles   = ['-', '--']  # [train, valid]

def _moving_average(x, window_size=1, trim_end=0):
    x_avg = np.zeros_like(x, dtype=float)
    for j in range(len(x)):
        lo = max(j - window_size // 2, 0)
        hi = min(j + window_size // 2, len(x)-1)
        x_avg[j] = np.mean(x[lo:hi+1])
    return x_avg[:-trim_end] if trim_end>0 else x_avg

# —— Gather all data into lists —— 
all_train, all_valid, all_steps = [], [], []

for ds in datasets:
    # build file list
    if ds == 'saccharomycetales':
        base = f'{input_dir}/LM_Johannes/lm_{ds}_gtf/lm_{ds}_gtf'
        files = [
            f'{base}_small/train/train.out',
            f'{base}_big/train/train.out',
            f'{base}_unet_small_bert_drop/train/train.out',
            f'{base}_unet_big_bert_drop/train_bk/train.out',
        ]
    elif ds == 'fungi_1385':
        base = f'{input_dir}/LM_Johannes/lm_{ds}_gtf/lm_{ds}_gtf'
        files = [
            f'{base}_small_bert/train/train.out',
            f'{base}_big_bert/train/train.out',
            f'{base}_unet_small_bert_drop/train/train.out',
            f'{base}_unet_big_bert_drop/train/train.out',
        ]
    else:
        base = f'{input_dir}/lm_{ds}_gtf/lm_{ds}_gtf'
        files = [
            f'{base}_small/train/train.out',
            f'{base}_big/train/train.out',
            f'{base}_unet_small/train/train.out',
            f'{base}_unet_big/train/train.out',
        ]

    # parse
    train_losses, valid_losses = [], []
    for fp in files:
        tlist, vlist = [], []
        with open(fp, 'rt') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Epoch '):
                    parts = line.split(': ')
                    tlist.append(float(parts[1].split()[0]))
                    vlist.append(float(parts[3].split()[0]))
        train_losses.append(np.array(tlist))
        valid_losses.append(np.array(vlist))

    # steps
    steps_arr = [(np.arange(len(t))+1) * steps for t in train_losses]

    all_train.extend(train_losses)
    all_valid.extend(valid_losses)
    all_steps.extend(steps_arr)

# —— Per‑model plots with dynamic x‑limits —— 
for m_idx, model_name in enumerate(model_names):
    idxs = [ds_i*4 + m_idx for ds_i in range(len(datasets))]

    # Training
    plt.figure(figsize=FigSize, dpi=300)
    max_step = 0
    for ds_i, idx in enumerate(idxs):
        loss = all_train[idx]
        stp  = all_steps[idx]
        sm   = _moving_average(loss, window_size=window_size, trim_end=trim_end)
        mn   = sm.min()
        arg  = sm.argmin()
        max_step = max(max_step, stp[len(sm)-1])

        plt.plot(
            stp[:len(sm)], sm,
            color=dataset_colors[ds_i],
            linestyle=model_styles[0],
            linewidth=1.5,
            label=f"{datasets[ds_i]}; min={mn:.4f}"
        )
        plt.axvline(x=stp[arg], color=dataset_colors[ds_i],
                    linestyle=model_styles[0], alpha=0.25)

    plt.xlim(steps, max_step+1000)
    plt.xlabel('# Training Batches')
    plt.ylabel('Training Loss')
    plt.title(f'{model_name}: Training Loss Across Datasets')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{model_name}_train_losses.png')
    plt.close()

    # Validation
    plt.figure(figsize=FigSize, dpi=300)
    max_step = 0
    for ds_i, idx in enumerate(idxs):
        loss = all_valid[idx]
        stp  = all_steps[idx]
        sm   = _moving_average(loss, window_size=window_size, trim_end=trim_end)
        mn   = sm.min()
        arg  = sm.argmin()
        max_step = max(max_step, stp[len(sm)-1])

        plt.plot(
            stp[:len(sm)], sm,
            color=dataset_colors[ds_i],
            linestyle=model_styles[1],
            linewidth=1.5,
            label=f"{datasets[ds_i]}; min={mn:.4f}"
        )
        plt.axvline(x=stp[arg], color=dataset_colors[ds_i],
                    linestyle=model_styles[1], alpha=0.25)

    plt.xlim(steps, max_step+1000)
    plt.xlabel('# Training Batches')
    plt.ylabel('Validation Loss')
    plt.title(f'{model_name}: Validation Loss Across Datasets')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{model_name}_valid_losses.png')
    plt.close()

# —— Per‑dataset plots with brighter model colors and mapped labels —— 
for ds_i, ds in enumerate(datasets):
    label = dataset_labels[ds]
    idxs  = [ds_i*4 + m for m in range(4)]

    # Training
    plt.figure(figsize=FigSize, dpi=300)
    max_step = 0
    for m_idx, idx in enumerate(idxs):
        loss = all_train[idx]
        stp  = all_steps[idx]
        sm   = _moving_average(loss, window_size=window_size, trim_end=trim_end)
        mn   = sm.min()
        arg  = sm.argmin()
        max_step = max(max_step, stp[len(sm)-1])

        plt.plot(
            stp[:len(sm)], sm,
            color=model_colors[m_idx],
            linestyle=model_styles[0],
            linewidth=1.5,
            label=f"{model_names[m_idx]}; min={mn:.4f}"
        )
        plt.axvline(
            x=stp[arg],
            color=model_colors[m_idx],
            linestyle=model_styles[0],
            alpha=0.25
        )

    plt.xlim(steps, max_step+1000)
    plt.xlabel('# Training Batches')
    plt.ylabel('Training Loss')
    plt.title(f'{label}: Training Loss Across Models')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{label}_train_losses.png')
    plt.close()

    # Validation
    plt.figure(figsize=FigSize, dpi=300)
    max_step = 0
    for m_idx, idx in enumerate(idxs):
        loss = all_valid[idx]
        stp  = all_steps[idx]
        sm   = _moving_average(loss, window_size=window_size, trim_end=trim_end)
        mn   = sm.min()
        arg  = sm.argmin()
        max_step = max(max_step, stp[len(sm)-1])

        plt.plot(
            stp[:len(sm)], sm,
            color=model_colors[m_idx],
            linestyle=model_styles[1],
            linewidth=1.5,
            label=f"{model_names[m_idx]}; min={mn:.4f}"
        )
        plt.axvline(
            x=stp[arg],
            color=model_colors[m_idx],
            linestyle=model_styles[1],
            alpha=0.25
        )

    plt.xlim(steps, max_step+1000)
    plt.xlabel('# Training Batches')
    plt.ylabel('Validation Loss')
    plt.title(f'{label}: Validation Loss Across Models')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{label}_valid_losses.png')
    plt.close()
