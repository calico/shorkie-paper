import matplotlib.pyplot as plt
from Bio import SeqIO

def plot_sequence_length_distribution(fasta_files, dbtypes):
    plt.figure(figsize=(10, 6))

    for fasta_file, dbtype in zip(fasta_files, dbtypes):
        # Extract sequence lengths
        sequence_lengths = [len(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
        
        # Plot histogram
        plt.hist(sequence_lengths, bins=30, alpha=0.5, edgecolor='black', label=dbtype)

    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("combined_length_distribution.png")

# Example usage
fasta_files = ["/home/khc/RMRB/Libraries/RMRBSeqs.fasta", "/home/khc/bin/RepeatMasker/Libraries/RepeatMasker.lib"]
dbtypes = ["RepBase", "Dfam"]
plot_sequence_length_distribution(fasta_files, dbtypes)
