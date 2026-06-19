import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from Bio import SeqIO
import numpy as np
import os
import glob

def find_lowercase_indices(fasta_file):
    lowercase_indices = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = record.seq
        for i, nucleotide in enumerate(seq):
            if nucleotide.islower():
                lowercase_indices.add(i)
    return lowercase_indices

def calculate_metrics(ground_truth, prediction):
    tp = len(ground_truth & prediction)
    fp = len(prediction - ground_truth)
    fn = len(ground_truth - prediction)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / len(ground_truth) if len(ground_truth) > 0 else 0

    return precision, recall, accuracy

def plot_venn_and_metrics(fasta_files, condition1, condition2, output_dir):
    sets = [find_lowercase_indices(fasta_file) for fasta_file in fasta_files]
    print('Number of lowercase regions in each file:', [len(s) for s in sets])

    # Calculate metrics
    precision, recall, accuracy = calculate_metrics(sets[0], sets[1])

    # Plot Venn diagram
    plt.figure(figsize=(8, 6.4))  # Increase the figure size
    venn2(sets, (condition1, condition2))
    plt.title('Venn Diagram of Repeat Regions in FASTA Files (nt)')

    # Add metrics to plot
    plt.text(0.5, -0.15, f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nAccuracy: {accuracy:.2f}',
             ha='center', fontsize=10, transform=plt.gca().transAxes)

    condition1 = condition1.replace('\n', ' ')
    filename = f'venn_diagram_{condition1}_vs_{condition2}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Ensure everything fits in the figure
    plt.close()

    return precision, recall, accuracy

data_type = "fungi_gtf"
FASTA_DIR = f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta"

precision_list = []
recall_list = []
accuracy_list = []
sample_labels = []

for fasta_file in glob.glob(os.path.join(FASTA_DIR, "*.cleaned.fasta")):
    base_name = os.path.basename(fasta_file).replace(".cleaned.fasta", "")
    print(base_name)  # Do something with base_name

    file1 = f'/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta/{base_name}.sm.fasta'
    file1_condition = 'ENSEMBL soft mask release'

    file2_1 = f'/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta/{base_name}.cleaned.fasta.masked'
    file2_2 = f'/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/data_{data_type}/fasta/{base_name}.cleaned.fasta.masked.dust.softmask'

    file2s_condition = ['RepeatModeler \n(RMBlast + Dfam)', 'RepeatModeler \n(RMBlast + Dfam) + Dust']

    file2s = [file2_1, file2_2]

    output_dir = base_name
    os.makedirs(output_dir, exist_ok=True)

    for j, file2 in enumerate(file2s):
        fasta_files = [file1, file2]
        condition1 = file1_condition
        condition2 = file2s_condition[j]
        print(f'Plotting Venn diagram for {condition1} vs {condition2}')
        precision, recall, accuracy = plot_venn_and_metrics(fasta_files, condition1, condition2, output_dir)
        dust_or_not = condition2.split('+')[-1]

        if dust_or_not.strip() == 'Dust':
            sample_labels.append(f'{base_name} (Dust)')
        else:
            continue

        # Store metrics
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)


# Plot scatter plot of precision and recall
plt.figure(figsize=(8, 8))
# Get the colors from tab20 colormap
colors = plt.cm.tab20(np.linspace(0, 1, len(precision_list)))

for i in range(len(precision_list)):
    plt.scatter(precision_list[i], recall_list[i], color=colors[i], label=sample_labels[i], alpha=0.7)

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title(f'Scatter Plot of Precision vs Recall for {data_type} Samples')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')

# Add legend
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Samples")

plt.grid(True)
plt.savefig(f'{data_type}_precision_vs_recall_scatter_plot.png', dpi=300, bbox_inches='tight')
