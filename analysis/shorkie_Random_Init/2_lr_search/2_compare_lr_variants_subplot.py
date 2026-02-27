from optparse import OptionParser
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def _moving_average(x, window_size=1, trim_end=0):
    if len(x) == 0:
        return np.array([])
    x_avg = np.zeros_like(x, dtype=float)
    for j in range(x.shape[0]):
        min_j = max(j - window_size // 2, 0)
        max_j = min(j + window_size // 2, x.shape[0] - 1)
        # simple mean
        x_avg[j] = np.mean(x[min_j:max_j + 1])
    if trim_end > 0 and trim_end < x.shape[0]:
        return x_avg[:-trim_end]
    return x_avg

def parse_all_metrics(log_file, prefix="Epoch "):
    train_loss, train_r, train_r2 = [], [], []
    valid_loss, valid_r, valid_r2 = [], [], []
    with open(log_file, "r") as f:
        for line_raw in f:
            line = line_raw.strip()
            if line.startswith(prefix):
                parts = line.split(" - ")
                try:
                    train_loss.append(float(parts[2].split(": ")[1]))
                    train_r.append(float(parts[3].split(": ")[1]))
                    train_r2.append(float(parts[4].split(": ")[1]))
                    valid_loss.append(float(parts[5].split(": ")[1]))
                    valid_r.append(float(parts[6].split(": ")[1]))
                    valid_r2.append(float(parts[7].split(": ")[1]))
                except:
                    continue
    return (np.array(train_loss), np.array(train_r), np.array(train_r2),
            np.array(valid_loss), np.array(valid_r), np.array(valid_r2))

def main():
    parser = OptionParser()
    parser.add_option('--root_dir', dest='root_dir', default='../../../Yeast_ML', type='str',
                      help='Root directory for Yeast_ML data')
    parser.add_option('--out_dir', dest='out_dir', default='./results', type='str',
                      help='Output directory')
    parser.add_option('--model_arch', dest='model_arch', default='unet_small_bert_drop', type='str')
    (options, args) = parser.parse_args()

    # Params
    steps_per_epoch = 500
    window_size = 11
    trim_end = 5
    
    save_root = os.path.join(options.out_dir, 'lr_comparison')
    os.makedirs(save_root, exist_ok=True)
    
    # 1. Identify Models
    # We want a list of models: [ {'label': 'Original', 'files': [...]}, {'label': 'LR=...', 'files': [...]}, ... ]
    
    models_list = []
    exp_root = f"{options.root_dir}/seq_experiment"
    variants_dir = f'{exp_root}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/supervised_{options.model_arch}_variants'
    
    # Shorkie_Random_Init (LR=0.0005 variant)
    random_init_dir = os.path.join(variants_dir, 'learning_rate_0.0005')
    random_init_files = [
        os.path.join(random_init_dir, 'train', f'f{i}c0', 'train.out')
        for i in range(8)
    ]
    models_list.append({'label': 'Shorkie_Random_Init', 'files': random_init_files, 'color': 'black', 'style': '--'})
    
    # Shorkie
    shorkie_files = [
        f'{exp_root}/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/self_supervised_{options.model_arch}/train/f{i}c0/train.out'
        for i in range(8)
    ]
    models_list.append({'label': 'Shorkie', 'files': shorkie_files, 'color': 'red', 'style': '-.'})
    
    # Variants (excluding learning_rate_0.0005, now Shorkie_Random_Init)
    lr_dirs = sorted(glob.glob(os.path.join(variants_dir, "learning_rate_*")))
    lr_dirs = [d for d in lr_dirs if os.path.basename(d) != 'learning_rate_0.0005']
    
    # Generate colormap for LRs
    cmap = cm.get_cmap('tab10') # or viral, plasma, etc.
    
    for idx, lr_dir in enumerate(lr_dirs):
        basename = os.path.basename(lr_dir)
        label = basename.replace('learning_rate_', 'LR=')
        
        files = []
        for i in range(8):
            files.append(os.path.join(lr_dir, 'train', f'f{i}c0', 'train.out'))
            
        color = cmap(idx / len(lr_dirs))
        models_list.append({'label': label, 'files': files, 'color': color, 'style': '-'})

    # 2. Load Data
    # structure: data[model_idx][metrics_idx][fold_idx] -> np array
    # metrics order: [t_loss, t_r, t_r2, v_loss, v_r, v_r2]
    
    all_data = [] # list of lists of lists
    for m_obj in models_list:
        m_metrics = [[], [], [], [], [], []] # 6 metrics
        for fpath in m_obj['files']:
            if os.path.exists(fpath):
                out = parse_all_metrics(fpath)
            else:
                out = (np.array([]),)*6
            
            for k in range(6):
                m_metrics[k].append(out[k])
        all_data.append(m_metrics)
        
    # 3. Plotting
    metric_names = ["Train Loss", "Train R", "Train R2", "Valid Loss", "Valid R", "Valid R2"]
    out_names    = ["train_loss", "train_r", "train_r2", "valid_loss", "valid_r", "valid_r2"]
    eval_types   = ["min", "max", "max", "min", "max", "max"] # min is better for loss
    
    for m_idx, m_name in enumerate(metric_names):
        print(f"Plotting {m_name}...")
        
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), dpi=300, sharey=True)
        fig.suptitle(f"Learning Rate Comparison: {m_name}", fontsize=20)
        
        axes_flat = axes.flatten()
        
        for fold_i in range(8):
            ax = axes_flat[fold_i]
            ax.set_title(f"Fold {fold_i}", fontsize=12)
            
            for model_i, m_obj in enumerate(models_list):
                raw_data = all_data[model_i][m_idx][fold_i] # array of values
                if len(raw_data) == 0:
                    continue
                
                # Smoothing
                smoothed = _moving_average(raw_data, window_size=window_size, trim_end=trim_end)
                steps_arr = (np.arange(len(smoothed)) + 1) * steps_per_epoch
                
                # Best value
                if len(smoothed) > 0:
                    if eval_types[m_idx] == "min":
                        best_val = np.min(smoothed)
                    else:
                        best_val = np.max(smoothed)
                else:
                    best_val = 0.0
                    
                label_str = f"{m_obj['label']} ({best_val:.3f})"
                
                ax.plot(steps_arr, smoothed, 
                        color=m_obj['color'], linestyle=m_obj['style'], 
                        linewidth=1.5 if m_obj['label'] not in ['Shorkie_Random_Init', 'Shorkie'] else 2.0, 
                        label=label_str)
            
            if fold_i == 0:
                ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_f = os.path.join(save_root, f"{out_names[m_idx]}_subplot.png")
        plt.savefig(out_f)
        plt.close()
        print(f"Saved {out_f}")

if __name__ == "__main__":
    main()
