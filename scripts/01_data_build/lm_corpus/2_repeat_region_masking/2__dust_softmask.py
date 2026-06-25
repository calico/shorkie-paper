import sys
from Bio import SeqIO

def mask_to_lowercase(masked_fasta, unmasked_fasta, output_fasta):
    with open(masked_fasta, "r") as masked, open(unmasked_fasta, "r") as unmasked, open(output_fasta, "w") as output:
        masked_records = SeqIO.to_dict(SeqIO.parse(masked, "fasta"))
        unmasked_records = SeqIO.to_dict(SeqIO.parse(unmasked, "fasta"))

        for record_id in unmasked_records:
            masked_seq = masked_records[record_id].seq
            unmasked_seq = unmasked_records[record_id].seq
            new_seq = []

            for m, u in zip(masked_seq, unmasked_seq):
                if m == 'N':
                    new_seq.append(u.lower())
                else:
                    new_seq.append(u)

            new_record = unmasked_records[record_id]
            new_record.seq = "".join(new_seq)
            SeqIO.write(new_record, output, "fasta")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mask_to_lowercase.py <masked_fasta> <unmasked_fasta> <output_fasta>")
        sys.exit(1)

    masked_fasta = sys.argv[1]
    unmasked_fasta = sys.argv[2]
    output_fasta = sys.argv[3]

    mask_to_lowercase(masked_fasta, unmasked_fasta, output_fasta)
