import os
import numpy as np
import pandas as pd
import torch 
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import BertForMaskedLM, AutoTokenizer, DefaultDataCollator

# Load the model
model_name = "johahi/specieslm-fungi-upstream-k1"
model = BertForMaskedLM.from_pretrained(model_name, trust_remote_code=True) 

# Load the corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# If you want to use the flash version, download the model from Zenodo and load with the commented code below:
'''
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertForPreTraining, DefaultDataCollator
from flash_attn.models.bert import BertModel, BertForPreTraining

model_path = 'species_upstream_1000_k1/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)
model = BertForPreTraining.from_pretrained(model_path, config)
'''

device = "cuda"
model.to(device)
model.eval()
print(f'GPU Model: {torch.cuda.get_device_name(0)}')

# Visualization functions
import seaborn as sns

def plot_map_with_seq(matrix, dna_sequence, plot_size=10, vmax=5, tick_label_fontsize=8):
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    sns.heatmap(matrix, cmap='coolwarm', vmax=vmax, ax=ax, 
                xticklabels=False, yticklabels=False)
    ax.set_aspect('equal')
    tick_positions = np.arange(len(dna_sequence)) + 0.5  # Center the ticks
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(list(dna_sequence), fontsize=tick_label_fontsize, rotation=0)
    ax.set_yticklabels(list(dna_sequence), fontsize=tick_label_fontsize)
    plt.show()
    
def plot_map(matrix, vmax=None, display_values=False, annot_size=8, fig_size=10):
    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(matrix, cmap="coolwarm", vmax=vmax, annot=display_values, 
                     fmt=".2f", annot_kws={"size": annot_size})
    ax.set_aspect('equal')
    plt.show()

# Dependency map generation functions
nuc_table = {"A": 0, "C": 1, "G": 2, "T": 3}

def mutate_sequence(seq):
    seq = seq.upper()
    mutated_sequences = {'seq': [], 'mutation_pos': [], 'nuc': [], 'var_nt_idx': []}
    mutated_sequences['seq'].append(seq)
    mutated_sequences['mutation_pos'].append(-1)
    mutated_sequences['nuc'].append('real sequence')
    mutated_sequences['var_nt_idx'].append(-1)
    mutate_until_position = len(seq)
    for i in range(mutate_until_position):
        for nuc in ['A', 'C', 'G', 'T']:
            if nuc != seq[i]:
                mutated_sequences['seq'].append(seq[:i] + nuc + seq[i+1:])
                mutated_sequences['mutation_pos'].append(i)
                mutated_sequences['nuc'].append(nuc)
                mutated_sequences['var_nt_idx'].append(nuc_table[nuc])
    mutations_df = pd.DataFrame(mutated_sequences)
    return mutations_df

def tok_func_species(x, proxy_species):
    res = tokenizer(proxy_species + " " + " ".join(list(x['seq'])))
    return res

def create_dataloader(dataset, proxy_species, batch_size=64, rolling_masking=False):
    ds = Dataset.from_pandas(dataset[['seq']])
    tok_ds = ds.map(lambda x: tok_func_species(x, proxy_species=proxy_species),
                    batched=False, num_proc=20)
    rem_tok_ds = tok_ds.remove_columns('seq')
    data_collator = DefaultDataCollator()
    data_loader = torch.utils.data.DataLoader(rem_tok_ds, batch_size=batch_size,
                                              num_workers=4, shuffle=False,
                                              collate_fn=data_collator)
    return data_loader

acgt_idxs = [tokenizer.get_vocab()[nuc] for nuc in ['A', 'C', 'G', 'T']]

def model_inference(model, data_loader):
    output_arrays = []
    for i, batch in enumerate(data_loader):
        print(f"Batch {i+1}/{len(data_loader)}")
        tokens = batch['input_ids']
        with torch.autocast(device):
            with torch.no_grad():
                outputs = model(tokens.to(device)).logits.cpu().to(torch.float32)
        output_probs = torch.nn.functional.softmax(outputs, dim=-1)[:, :, acgt_idxs]
        output_arrays.append(output_probs)
    snp_reconstruct = torch.concat(output_arrays, axis=0)
    return snp_reconstruct.to(torch.float32).numpy()

def compute_dependency_map(seq, proxy_species, epsilon=1e-10):
    dataset = mutate_sequence(seq)
    data_loader = create_dataloader(dataset, proxy_species=proxy_species)
    snp_reconstruct = model_inference(model, data_loader)
    print("snp_reconstruct: ", snp_reconstruct.shape)
    snp_reconstruct = snp_reconstruct[:, 2:-1, :]  # discard special tokens as needed
    snp_reconstruct = snp_reconstruct + epsilon
    snp_reconstruct = snp_reconstruct / snp_reconstruct.sum(axis=-1)[:, :, np.newaxis]
    seq_len = snp_reconstruct.shape[1]
    snp_effect = np.zeros((seq_len, seq_len, 4, 4))
    reference_probs = snp_reconstruct[dataset[dataset['nuc'] == 'real sequence'].index[0]]
    snp_reconstruct_return = reference_probs.copy()
    snp_effect[dataset.iloc[1:]['mutation_pos'].values, :,
               dataset.iloc[1:]['var_nt_idx'].values, :] = (
        np.log2(snp_reconstruct[1:]) - np.log2(1 - snp_reconstruct[1:]) -
        np.log2(reference_probs) + np.log2(1 - reference_probs)
    )
    dep_map = np.max(np.abs(snp_effect), axis=(2, 3))
    np.fill_diagonal(dep_map, 0)
    return dep_map, snp_reconstruct_return

# --- New Section: Extract Sequence from FASTA File ---
from Bio import SeqIO

def extract_sequence_from_fasta(fasta_path, chrom, start, end):
    """
    Extracts a sequence from a FASTA file given a chromosome (or record) id,
    a start coordinate, and an end coordinate.
    Note: start and end should be 0-indexed, and end is exclusive.
    """
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == chrom:
            return str(record.seq[start:end])
    raise ValueError(f"Chromosome {chrom} not found in FASTA file {fasta_path}")

# Define parameters for sequence extraction
fasta_path = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta"  # Update this with the path to your FASTA file
chrom = "chrX"            # Update this with the correct chromosome or record id
start_coord = 607855                    # Update with your start coordinate (0-indexed)
end_coord = 608355                   # Update with your end coordinate (exclusive)

# Extract the sequence
sequence = extract_sequence_from_fasta(fasta_path, chrom, start_coord, end_coord)
print("Extracted sequence length:", len(sequence))

# Use your proxy species as before
proxy_species = 'kazachstania_africana_cbs_2517_gca_000304475'

# Compute the dependency map for the extracted sequence
dep_map, all_prbs = compute_dependency_map(sequence, proxy_species=proxy_species)
print("Length of sequence:", len(sequence))
print("Dependency map shape:", dep_map.shape)
print("All probabilities shape:", all_prbs.shape)

# Save the probabilities to a file
np.save(f'all_prbs_{chrom}_{start_coord}_{end_coord}.npy', all_prbs)
# plot_map(dep_map, vmax=10)
# # Optionally, visualize a sub-region of the dependency map with sequence labels
# plot_map_with_seq(dep_map[690:800, 690:800], sequence[690:800], vmax=6, tick_label_fontsize=6)

# # Further processing and visualization functions from your original script
# from utils import plot_weights

# def compute_log_ratio(nuc_probs):
#     nucs_mean = nuc_probs[:-3].mean(axis=0)
#     nucs_normed = nuc_probs * np.log2(nuc_probs / nucs_mean)
#     return nucs_normed

# def compute_per_position_ic(ppm, 
#                             background=torch.tensor([0.325, 0.176, 0.175, 0.324]),
#                             pseudocount=0):
#     ppm = ppm.unsqueeze(0)
#     alphabet_len = len(background)
#     bg = torch.log2(background) * background
#     pseudocounted_ppm = (ppm + pseudocount) / (1 + pseudocount * alphabet_len)
#     ic = torch.log2(pseudocounted_ppm) * ppm - bg.unsqueeze(0)
#     return torch.sum(ic, axis=2)

# # Convert numpy probabilities to a torch tensor for further processing
# all_prbs_tensor = torch.from_numpy(all_prbs).float()
# result = all_prbs_tensor * compute_per_position_ic(all_prbs_tensor).swapaxes(0, 1)
# # plot_weights(result.numpy()[690:800], subticks_frequency=1)
