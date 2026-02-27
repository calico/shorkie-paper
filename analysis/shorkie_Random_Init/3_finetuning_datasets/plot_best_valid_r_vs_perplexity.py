import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---------------------------
# 1. Configuration & Data
# ---------------------------

# Extracted Perplexity Values
# Mapping: Model Name -> Perplexity
perplexity_data = {
    "Shorkie": 3.58526,        # Saccharomycetales
    "80 Strain MLM": 3.72572,  # Strains
    "1341 Fungal MLM": 3.63795, # Fungi 1385
    "R64 Yeast MLM": 3.74581   # Yeast R64
}

parser = argparse.ArgumentParser(description="Plot best valid r vs perplexity")
parser.add_argument("--root_dir", default="../../../Yeast_ML", help="Root directory for Yeast_ML data")
parser.add_argument("--out_dir", default="./results", help="Output directory")
args = parser.parse_args()

# File Paths for Pearson's R (train.out)
base_path = f"{args.root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"

# Mapping: Model Name -> Path Template
model_paths = {
    "Shorkie": f"{base_path}/self_supervised_unet_small_bert_drop/train/{{fold}}/train.out",
    "80 Strain MLM": f"{base_path}/self_supervised_unet_small_bert_drop_all_mlm_load/80_strains_mlm/train/{{fold}}/train.out",
    "1341 Fungal MLM": f"{base_path}/self_supervised_unet_small_bert_drop_all_mlm_load/1341_fungal_mlm/train/{{fold}}/train.out",
    "R64 Yeast MLM": f"{base_path}/self_supervised_unet_small_bert_drop_all_mlm_load/yeast_mlm/train/{{fold}}/train.out"
}

folds = [f"f{i}c0" for i in range(8)]

# Plotting Configuration
model_colors = {
    "Shorkie": "#ff7f0e",     # Orange
    "80 Strain MLM": "#2ca02c", # Green
    "1341 Fungal MLM": "#d62728", # Red
    "R64 Yeast MLM": "#9467bd"  # Purple
}

# ---------------------------
# 2. Parsing Function
# ---------------------------
def get_best_metrics_from_file(log_file):
    """
    Parses a train.out file to find:
      - Maximum validation Pearson's R (max_valid_r)
      - Minimum validation loss (min_valid_loss)
    """
    max_valid_r = -np.inf
    min_valid_loss = np.inf
    
    if not os.path.exists(log_file):
        print(f"Warning: File not found: {log_file}")
        return None, None

    with open(log_file, "r") as f:
        for line in f:
            if line.strip().startswith("Epoch "):
                # Example line:
                # "Epoch 1 - ... - valid_loss: 0.55 - valid_r: 0.43 - ..."
                try:
                    parts = line.split(" - ")
                    
                    # Parse valid_r
                    valid_r_part = next((p for p in parts if p.startswith("valid_r: ")), None)
                    if valid_r_part:
                        val_r = float(valid_r_part.split(": ")[1])
                        if val_r > max_valid_r:
                            max_valid_r = val_r
                            
                    # Parse valid_loss
                    valid_loss_part = next((p for p in parts if p.startswith("valid_loss: ")), None)
                    if valid_loss_part:
                        val_l = float(valid_loss_part.split(": ")[1])
                        if val_l < min_valid_loss:
                            min_valid_loss = val_l
                            
                except Exception as e:
                    continue
                    
    if max_valid_r == -np.inf:
        max_valid_r = None
    if min_valid_loss == np.inf:
        min_valid_loss = None
        
    return max_valid_r, min_valid_loss

# ---------------------------
# 3. Main Data Aggregation
# ---------------------------
plot_data = [] # List of dicts

print("Extracting Pearson's R and Validation Loss data...")

for model_name in perplexity_data.keys():
    print(f"Processing {model_name}...")
    valid_r_values = []
    valid_loss_values = []
    
    path_template = model_paths[model_name]
    
    for fold in folds:
        file_path = path_template.format(fold=fold)
        best_r, min_loss = get_best_metrics_from_file(file_path)
        
        if best_r is not None:
            valid_r_values.append(best_r)
        if min_loss is not None:
            valid_loss_values.append(min_loss)
            
    if not valid_r_values:
        print(f"  Error: No valid data found for {model_name}")
        continue
        
    # Stats for Pearson's R
    mean_r = np.mean(valid_r_values)
    std_r = np.std(valid_r_values)
    
    # Stats for Validation Loss
    mean_loss = np.mean(valid_loss_values)
    std_loss = np.std(valid_loss_values)
    
    perp = perplexity_data[model_name]
    
    print(f"  Result: Perplexity = {perp}")
    print(f"          Mean R    = {mean_r:.4f} (+/- {std_r:.4f})")
    print(f"          Mean Loss = {mean_loss:.4f} (+/- {std_loss:.4f})")
    
    plot_data.append({
        "model": model_name,
        "perplexity": perp,
        "mean_r": mean_r,
        "std_r": std_r,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "color": model_colors.get(model_name, "blue")
    })

# ---------------------------
# 4. Plotting
# ---------------------------
output_dir = args.out_dir
os.makedirs(output_dir, exist_ok=True)

def create_scatter_plot(data, y_key, yerr_key, y_label, title, filename):
    plt.figure(figsize=(8, 6))
    # sns.set_style("whitegrid")
    plt.grid(True, alpha=0.3)

    for item in data:
        plt.errorbar(
            x=item["perplexity"], 
            y=item[y_key], 
            yerr=item[yerr_key], 
            fmt='o', 
            markersize=10, 
            capsize=5, 
            label=item["model"],
            color=item["color"]
        )

        # Annotate points
        plt.text(
            item["perplexity"], 
            item[y_key] + (0.0005 if "R" in title else 0.05), # Small offset, messy heuristic but fine
            item["model"], 
            fontsize=9, 
            ha='center', 
            va='bottom'
        )

    plt.xlabel("Masked Language Model Perplexity (Lower is better)", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(title="Model", loc='best')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")

# Plot 1: Pearson's R
create_scatter_plot(
    plot_data, 
    "mean_r", 
    "std_r", 
    "Best Validation Pearson's R (Higher is better)", 
    "Model Performance: Pearson's R vs Perplexity", 
    os.path.join(output_dir, "plot_best_valid_r_vs_perplexity.png")
)

# Plot 2: Validation Loss
create_scatter_plot(
    plot_data, 
    "mean_loss", 
    "std_loss", 
    "Best Validation Loss (Lower is better)", 
    "Model Performance: Validation Loss vs Perplexity", 
    os.path.join(output_dir, "plot_best_valid_loss_vs_perplexity.png")
)
