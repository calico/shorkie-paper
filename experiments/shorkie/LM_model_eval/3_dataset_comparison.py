import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('viz/unet_small_comparison/', exist_ok=True)

#Load and parse training metrics
steps = 64
n_epochs = 410
# FigSize=(6, 4)

# n_epochs = 45
# FigSize=(4, 3)
# FigSize=(6, 3)
# FigSize=(13.5, 4)
# FigSize=(8, 6)
FigSize=(6, 4)

# steps = 70
trim_end = 5
window_size = 21

plot_train_loss = True

min_iters = steps
max_iters = n_epochs * steps
root_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/"

model_names = [
    'R64_Yeast',
    '80_Strains',
    '165_Saccharomycetales',
    '1342_Fungus',
]

model_files = [
    f'{root_dir}lm_r64_gtf/lm_r64_gtf_unet_small/train/train.out',
    f'{root_dir}lm_strains_gtf/lm_strains_gtf_unet_small/train/train.out',
    # 'LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_aux_drop/train/train.out',
    f'{root_dir}LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small/train/train.out',
    f'{root_dir}LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small/train/train.out',
]

# model_colors = [
#     "#FF2400",
#     "#603F8B",
#     "#18A558",
#     "#8A2BE2",  # Purple
# ]
model_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

model_styles = [ 
    '--', '--', '-', '-',
    '--', '--', '-', '-',
    '--', '--', '-', '-',
    '--', '--', '-', '-',
    # '-', '-', '--', '--',
    # '-', '-', '--', '--',
    # '-', '-', '--', '--',
]

model_datastrs = [
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
    'Epoch ', 'Epoch ', 'Epoch ', 'Epoch ',
]

# Define the specific step sizes for each model
model_steps = [
    steps, steps, steps, steps,  # 150 steps for lm_r64_gtf models
    steps, steps, steps, steps,  # 150 steps for lm_strains_gtf models
    steps * (500 / 150), steps * (500 / 150), steps * (500 / 150), steps * (500 / 150),  # Rescale steps for lm_saccharomycetales_gtf models
    steps, steps, steps, steps,  # 150 steps for lm_fungi_1385_gtf models
]

train_losses = []
train_rs = []
train_r2s = []
valid_losses = []
valid_rs = []
valid_r2s = []

#Loop over model metric files
for model_i, model_file in enumerate(model_files) :
    train_loss = []
    valid_loss = []
    with open(model_file, 'rt') as f :
        for line_raw in f.readlines() :
            line = line_raw.strip()
            if line[:6] == model_datastrs[model_i] :
                line_parts = line.split(": ")
                train_loss.append( float(line_parts[1].split(" ")[0] ))
                valid_loss.append( float(line_parts[3].split(" ")[0] ))
    train_losses.append(np.array(train_loss))
    valid_losses.append(np.array(valid_loss))

# Calculate moving steps with adjusted steps sizes
m_steps = [
    (np.arange(int(np.max([len(train_loss) for train_loss in train_losses]))) + 1) * model_steps[model_ix] 
    for model_ix in range(len(model_steps))
]

#Plot train / valid loss
def _moving_average(x, window_size=1, trim_end=0) :
    x_avg = np.zeros(x.shape)
    for j in range(x.shape[0]) :
        min_j = max(j - window_size // 2, 0)
        max_j = min(j + window_size // 2, x.shape[0] - 1)
        actual_window_size = min(j - min_j, max_j - j)
        min_j = j - actual_window_size
        max_j = j + actual_window_size
        x_avg[j] = np.mean(x[min_j:max_j+1])
    return x_avg[:-trim_end] if trim_end > 0 else x_avg
m_steps = [
    (np.arange(int(np.max([len(train_loss) for train_loss in train_losses]))) + 1) * model_steps[model_ix] for model_ix in range(len(model_steps))
]

# Set a larger value for max_iters to extend the x-axis
max_iters = n_epochs * steps * 12  # Increase by 50%, or choose another factor

# FigSize=(7, 8)
# Plot train losses
f_train = plt.figure(figsize=FigSize, dpi=600)
for name, color, loss, xs in zip(model_names, model_colors, train_losses, m_steps):
    avg = _moving_average(loss, window_size=window_size, trim_end=trim_end)
    orig = loss[:-trim_end]
    min_avg = np.min(avg)
    min_orig = np.min(orig)
    argmin_orig = np.argmin(orig)
    # plt.plot(xs[:len(avg)], avg,
    #          label=f"{name}; loss = {min_avg:.4f} ({min_orig:.4f})",
    #          linewidth=1.5, linestyle='-',
    #          color=color, zorder=-1000)
    plt.plot(xs[:len(avg)], avg,
             label=f"{name}; loss = {min_orig:.4f}",
             linewidth=1, linestyle='-',
             color=color, zorder=-1000)
    plt.axvline(x=xs[argmin_orig],
                linestyle='-', linewidth=1, alpha=0.25,
                color=color, zorder=-1500)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(min_iters, max_iters)
plt.xlabel('# Training Batches', fontsize=10)
plt.ylabel('Training Loss', fontsize=10)
plt.title('Training Losses')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'viz/unet_small_comparison/train_losses_{n_epochs}.png')
plt.show()


# Plot validation losses
f_valid = plt.figure(figsize=FigSize, dpi=600)
for model_i, [model_name, model_color, model_style, valid_loss, steps] in enumerate(zip(model_names, model_colors, model_styles, valid_losses, m_steps)):
    valid_loss_min = np.min(_moving_average(valid_loss, window_size=window_size, trim_end=trim_end))
    valid_loss_min_orig = np.min(valid_loss[:-trim_end])
    valid_loss_argmin = np.argmin(_moving_average(valid_loss, window_size=window_size, trim_end=trim_end))
    valid_loss_argmin_orig = np.argmin(valid_loss[:-trim_end])
    # plt.plot(steps[:valid_loss.shape[0] - trim_end], _moving_average(valid_loss, window_size=window_size, trim_end=trim_end), color=model_color, linewidth=1.5, linestyle='--', zorder=-1000, label=model_name + "; loss = " + '{:.4f}'.format(valid_loss_min) + " (" + '{:.4f}'.format(valid_loss_min_orig) + ")")
    plt.plot(steps[:valid_loss.shape[0] - trim_end], _moving_average(valid_loss, window_size=window_size, trim_end=trim_end), color=model_color, linewidth=1, linestyle='--', zorder=-1000, label=model_name + "; loss = " + '{:.4f}'.format(valid_loss_min_orig))

    plt.axvline(x=steps[valid_loss_argmin_orig], color=model_color, linewidth=1, linestyle='--', alpha=0.25, zorder=-1500)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
if min_iters is not None and max_iters is not None:
    plt.xlim(min_iters, max_iters)
plt.xlabel('# Training Batches', fontsize=10)
plt.ylabel('Validation Loss', fontsize=10)
plt.legend(fontsize=10)
plt.title('Validation Losses')
plt.tight_layout()
plt.savefig(f'viz/unet_small_comparison/valid_losses_{n_epochs}.png')
plt.show()