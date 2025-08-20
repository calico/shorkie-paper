import os
import sys
import time
import pandas as pd
import numpy as np
import pysam

from basenji import gene as bgene
import pybedtools
import itertools

# helper function to extract running intervals
def extract_intervals(x):
    x_it = sorted(set(x))
    
    for key, group in itertools.groupby(enumerate(x_it), lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]

# function to extract and encode binary exon-overlap mask
def get_exon_mask(chrom, start, end, transcriptome, bedt_span, assembly_level='chromosome', seq_len=16384, chew_bp=2) :
    exon_mask = np.zeros(seq_len, dtype=bool)

    # get sequence bedtool
    seq_bedt = pybedtools.BedTool('%s %d %d' % (chrom[3:] if assembly_level == 'chromosome' else chrom, start, end), from_string=True)
    gene_ids = sorted(list(set([overlap[3] for overlap in bedt_span.intersect(seq_bedt, wo=True)])))

    # loop over gene ids
    for gene_id in gene_ids :

        gene_slice = transcriptome.genes[gene_id].output_slice(start, seq_len, 1, False)

        # mark exon intervals
        exon_intervals = list(extract_intervals(gene_slice.tolist()))

        for exon_interval in exon_intervals :

            chew_bp_left = 0
            if exon_interval[0] > 0 :
                chew_bp_left = chew_bp

            chew_bp_right = 0
            if exon_interval[1] < seq_len - 1 :
                chew_bp_right = chew_bp

            if (exon_interval[1]+1-chew_bp_right) - exon_interval[0]+chew_bp_left > 0 :
                exon_mask[exon_interval[0]+chew_bp_left:exon_interval[1]+1-chew_bp_right] = True

    return exon_mask

def is_file_empty(file_path):
    with open(file_path, 'r') as file:
        return file.read(1) == ''

# function to generate sequence bed files (less strict biotype_constraints = {'' : 0.925})
def generate_beds(species_df, save_suffix, out_dir, train_exclude_dict, valid_include_dict, test_include_dict, seq_length=16384, seq_stride=4096, record_size=32, max_rm_frac=None, contig_padding=512, biotype_constraints={'' : 0.875, '.rRNA' : 0.03125, '.tRNA' : 0.03125}) :
    
    if out_dir != '' and out_dir[-1] != '/' :
        out_dir += '/'

    # train
    species_train = []
    contig_train = []
    start_train = []
    end_train = []
    label_train = []

    # valid
    species_valid = []
    contig_valid = []
    start_valid = []
    end_valid = []
    label_valid = []

    # test
    species_test = []
    contig_test = []
    start_test = []
    end_test = []
    label_test = []
    
    max_rm = None
    if max_rm_frac is not None :
        max_rm = int(np.round(seq_length * max_rm_frac))

    # loop over species rows
    for species_i, [_, row] in enumerate(species_df.iterrows()) :

        accession_id = row['Accession']
        accession_id_cleaned = accession_id.replace(".", "_").replace("-", "_").replace("(", "").replace(")", "")
        
        assembly_level = row['assembly_level']

        print('Generating bed for species ' + str(species_i) + ': ' + str(accession_id_cleaned))
        
        fasta_file = out_dir + 'data' + save_suffix + '/fasta/' + accession_id_cleaned + '.cleaned.fasta'

        # open fasta (pysam)
        fasta_pysam = pysam.Fastafile(fasta_file)

        # extract dictionary of references to lengths
        contig_refs = fasta_pysam.references
        contig_lengths = fasta_pysam.lengths

        length_dict = {contig_ref : int(contig_length) for contig_ref, contig_length in zip(contig_refs, contig_lengths)}
        
        # load gtf annotations
        gtf_dict = {}
        bedt_dict = {}
        
        biotype_violations = {biotype : 0 for biotype in biotype_constraints}
        
        # loop over biotypes
        for biotype in biotype_constraints :


            # load gtf
            gtf_file = out_dir + 'data' + save_suffix + '/gtf/' + accession_id_cleaned + biotype + '.59.gtf'
            if is_file_empty(gtf_file):
                print(f"{gtf_file} is empty.")
                transcriptome = None
                bedt_span = None
            else:
                print(f"{gtf_file} is not empty.")
                transcriptome = bgene.Transcriptome(gtf_file)
                bedt_span = transcriptome.bedtool_span()

            # transcriptome = bgene.Transcriptome()
            # bedt_span = transcriptome.bedtool_span()
            
            gtf_dict[biotype] = transcriptome
            bedt_dict[biotype] = bedt_span

        # loop over contig references
        for contig_i, contig_ref in enumerate(contig_refs) :
            start_i = contig_padding
            end_i = length_dict[contig_ref] - seq_length - contig_padding

            label = 'train'
            if accession_id_cleaned in valid_include_dict and valid_include_dict[accession_id_cleaned] is None :
                label = 'valid'
            elif accession_id_cleaned in valid_include_dict and contig_ref in valid_include_dict[accession_id_cleaned] :
                label = 'valid'
            elif accession_id_cleaned in test_include_dict and test_include_dict[accession_id_cleaned] is None :
                label = 'test'
            elif accession_id_cleaned in test_include_dict and contig_ref in test_include_dict[accession_id_cleaned] :
                label = 'test'
            elif accession_id_cleaned in train_exclude_dict and train_exclude_dict[accession_id_cleaned] is None :
                label = ''
            elif accession_id_cleaned in train_exclude_dict and contig_ref in train_exclude_dict[accession_id_cleaned] :
                label = ''

            if label == '' :
                continue
            
            for j in range(start_i, end_i, seq_stride) :
                start_j = j
                end_j = j + seq_length

                if end_j > end_i :
                    break
                
                if label not in ['valid', 'test'] and max_rm is not None :
                    count_rm = fasta_pysam.fetch(contig_ref, start_j, end_j).count('N')
                    if count_rm > max_rm :
                        continue

                invalid_region = False
                
                # loop over biotypes
                for biotype in biotype_constraints :
                    if gtf_dict[biotype] is not None and bedt_dict[biotype] is not None:
                        
                        
                        # # print("gtf_dict[biotype]: ", gtf_dict[biotype])
                        # # print("bedt_dict[biotype]: ", bedt_dict[biotype])
                        # for row in bedt_dict[biotype]:
                        #     if len(row.fields) != 6:
                        #         continue
                        bedt_dict[biotype] = bedt_dict[biotype].filter(lambda x: len(x.fields) == 6).saveas()
                        # print("type(bedt_dict[biotype]): ", type(bedt_dict[biotype]))
                        # bedt_dict[biotype] = [row for row in bedt_dict[biotype] if (len(row.fields) != 6)]
                        
                        # get exon fraction of sequence window
                        biotype_frac = np.sum(get_exon_mask(contig_ref, start_j, end_j, gtf_dict[biotype], bedt_dict[biotype], assembly_level=assembly_level, seq_len=seq_length, chew_bp=2), dtype='float32') / seq_length
                        # mark as invalid if above max limit
                        if biotype_frac > biotype_constraints[biotype] :
                            invalid_region = True
                            biotype_violations[biotype] += 1
                            break
                
                # skip if invalid region
                if invalid_region :
                    continue
                
                # emit coordinates as row

                # train
                if label == 'train' :
                    species_train.append(accession_id_cleaned)
                    contig_train.append(contig_ref)
                    start_train.append(start_j)
                    end_train.append(end_j)
                    label_train.append(label)

                elif label == 'valid' : # valid
                    species_valid.append(accession_id_cleaned)
                    contig_valid.append(contig_ref)
                    start_valid.append(start_j)
                    end_valid.append(end_j)
                    label_valid.append(label)

                elif label == 'test' : # test
                    species_test.append(accession_id_cleaned)
                    contig_test.append(contig_ref)
                    start_test.append(start_j)
                    end_test.append(end_j)
                    label_test.append(label)
        
        # close fasta
        fasta_pysam.close()
        
        print("biotype_violations = " + str(biotype_violations))
    # store as bed file (merged)
    bed_file_merged = out_dir + 'data' + save_suffix + '/sequences.bed'
    bed_out_merged = open(bed_file_merged, 'w')

    # write train bed file

    # shuffle records
    tfr_shuffle_index = np.arange(len(species_train), dtype='int32')

    rng = np.random.RandomState(42)
    rng.shuffle(tfr_shuffle_index)

    # store as bed file (train)
    bed_file = out_dir + 'data' + save_suffix + '/sequences_train.bed'
    bed_out = open(bed_file, 'w')

    # write rows
    n_rows_train = (len(species_train) // record_size) * record_size

    for ii in range(n_rows_train) :
        i = tfr_shuffle_index[ii]

        line = '%s\t%d\t%d\t%s\t%s' % (contig_train[i], start_train[i], end_train[i], label_train[i], species_train[i])
        print(line, file=bed_out)
        print(line, file=bed_out_merged)

    bed_out.close()

    print("Wrote " + str(n_rows_train) + " train lines.")

    # write valid bed file

    # shuffle records
    tfr_shuffle_index = np.arange(len(species_valid), dtype='int32')

    rng = np.random.RandomState(42)
    rng.shuffle(tfr_shuffle_index)

    # store as bed file (valid)
    bed_file = out_dir + 'data' + save_suffix + '/sequences_valid.bed'
    bed_out = open(bed_file, 'w')

    # write rows
    n_rows_valid = (len(species_valid) // record_size) * record_size

    for ii in range(n_rows_valid) :
        i = tfr_shuffle_index[ii]

        line = '%s\t%d\t%d\t%s\t%s' % (contig_valid[i], start_valid[i], end_valid[i], label_valid[i], species_valid[i])
        print(line, file=bed_out)
        print(line, file=bed_out_merged)

    bed_out.close()

    print("Wrote " + str(n_rows_valid) + " valid lines.")

    # write test bed file

    # shuffle records
    tfr_shuffle_index = np.arange(len(species_test), dtype='int32')

    rng = np.random.RandomState(42)
    rng.shuffle(tfr_shuffle_index)

    # store as bed file (test)
    bed_file = out_dir + 'data' + save_suffix + '/sequences_test.bed'
    bed_out = open(bed_file, 'w')

    # write rows
    n_rows_test = (len(species_test) // record_size) * record_size

    for ii in range(n_rows_test) :
        i = tfr_shuffle_index[ii]

        line = '%s\t%d\t%d\t%s\t%s' % (contig_test[i], start_test[i], end_test[i], label_test[i], species_test[i])
        print(line, file=bed_out)
        print(line, file=bed_out_merged)

    bed_out.close()

    bed_out_merged.close()

    print("Wrote " + str(n_rows_test) + " test lines.")

    # #Write statistics file
    # stats_file = out_dir + 'data' + save_suffix + '/statistics.json'
    # stats_out = open(stats_file, 'w')

    # print("{", file=stats_out)
    # print("\t\"seq_length\": " + str(seq_length) + ",", file=stats_out)
    # print("\t\"seq_1hot\": true,", file=stats_out)
    # print("\t\"test_seqs\": " + str(n_rows_test) + ",", file=stats_out)
    # print("\t\"valid_seqs\": " + str(n_rows_valid) + ",", file=stats_out)
    # print("\t\"train_seqs\": " + str(n_rows_train) + ",", file=stats_out)
    # print("\t\"num_species\": " + str(len(species_df)), file=stats_out)
    # print("}", file=stats_out)

    # stats_out.close()

    # #Write dummy targets file
    # targets_file = out_dir + 'data' + save_suffix + '/targets.txt'
    # targets_out = open(targets_file, 'w')

    # print("\tidentifier\tfile\tclip\tclip_soft\tscale\tsum_stat\tstrand_pair\tdescription", file=targets_out)
    # print("0\tRNA-MISSI.1\t/home/jlinder/tillage/datasets/yeast/rna/RNA-MISSI.1/coverage.w5\t1\t1\t1.0\tsum_sqrt\t0\tRNA:missing", file=targets_out)

    # targets_out.close()
