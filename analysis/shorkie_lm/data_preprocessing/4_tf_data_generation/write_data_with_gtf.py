#!/usr/bin/env python
from optparse import OptionParser

import os

import pandas as pd
import numpy as np

import util
import pysam

from basenji.dna_io import dna_1hot, dna_1hot_index
from basenji import gene as bgene

import pybedtools
import itertools

import tensorflow as tf

'''
write_data_with_gtf.py

Write one-hot coded sequences to TF records, along with gtf annotation masks.
'''

# helper function to extract running intervals
def extract_intervals(x):
    x_it = sorted(set(x))
    for key, group in itertools.groupby(enumerate(x_it), lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


# # function to extract and encode binary exon-overlap mask
# def get_exon_mask(chrom, start, end, transcriptome, bedt_span, assembly_level='chromosome', seq_len=16384, chew_bp=2) :
#     exon_mask = np.zeros(seq_len, dtype=bool)
#     # get sequence bedtool
#     seq_bedt = pybedtools.BedTool('%s %d %d' % (chrom[3:] if assembly_level == 'chromosome' else chrom, start, end), from_string=True)
#     gene_ids = sorted(list(set([overlap[3] for overlap in bedt_span.intersect(seq_bedt, wo=True)])))
#     # loop over gene ids
#     for gene_id in gene_ids :
#         gene_slice = transcriptome.genes[gene_id].output_slice(start, seq_len, 1, False)
#         # mark exon intervals
#         exon_intervals = list(extract_intervals(gene_slice.tolist()))
#         for exon_interval in exon_intervals :
#             chew_bp_left = 0
#             if exon_interval[0] > 0 :
#                 chew_bp_left = chew_bp
#             chew_bp_right = 0
#             if exon_interval[1] < seq_len - 1 :
#                 chew_bp_right = chew_bp
#             if (exon_interval[1]+1-chew_bp_right) - exon_interval[0]+chew_bp_left > 0 :
#                 exon_mask[exon_interval[0]+chew_bp_left:exon_interval[1]+1-chew_bp_right] = True
#     return exon_mask


# function to extract and encode binary exon-overlap mask
def get_exon_mask(chrom, seqname, start, end, transcriptome, bedt_span, fasta_sequences, assembly_level='chromosome', seq_len=16384, chew_bp=2) :
    exon_mask = np.zeros(seq_len, dtype=bool)
    donor_motifs = {}
    acceptor_motifs = {}

    # get sequence bedtool
    seq_bedt = pybedtools.BedTool('%s %d %d' % (chrom[3:] if assembly_level == 'chromosome' else chrom, start, end), from_string=True)
    gene_ids = sorted(list(set([overlap[3] for overlap in bedt_span.intersect(seq_bedt, wo=True)])))

    # loop over gene ids
    for gene_id in gene_ids :
        gene_slice = transcriptome.genes[gene_id].output_slice(start, seq_len, 1, False)
        gene_exons = transcriptome.genes[gene_id].exons

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
            
            # # Check for donor and acceptor motifs
            # for exon in gene_exons:
            #     exon_start = exon[0]
            #     exon_end = exon[1]
            #     if exon_start >= start and exon_end <= end:
            #         # print("fasta_sequences: ", fasta_sequences)
            #         # print("exon_start - exon_end: ", exon_start, exon_end)

            #         sequence = fasta_sequences.fetch(seqname)
            #         # , exon_start, exon_end)
            #         # print("exon_sequence: ", exon_sequence)
            #         donor_site = exon_end - start
            #         acceptor_site = exon_start - start
            #         donor_motifs[donor_site] = sequence[donor_site-4:donor_site+4]  # last 2 bases of exon (GT)
            #         acceptor_motifs[acceptor_site] = sequence[acceptor_site-4:acceptor_site+4]  # first 2 bases of exon (AG)

    return exon_mask, donor_motifs, acceptor_motifs



# function to extract and encode repeat mask
def get_repeat_mask(seq_dna, seq_len=16384):
    repeat_mask = np.zeros(seq_len, dtype=bool)
    for i, base in enumerate(seq_dna):
        if base.islower():
            repeat_mask[i] = True
    return repeat_mask

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--save_suffix', dest='save_suffix', default='', type='str', help='Suffix of species file [Default: %default]')
    parser.add_option('--start', dest='start_i', default=0, type='int', help='Sequence start index [Default: %default]')
    parser.add_option('--end', dest='end_i', default=1, type='int', help='Sequence end index [Default: %default]')
    parser.add_option('--part', dest='part_i', default=0, type='int', help='Part index [Default: %default]')
    parser.add_option('--seq_length', dest='seq_length', default=16384, type='int', help='Sequence length [Default: %default]')
    parser.add_option('--seq_stride', dest='seq_stride', default=4096, type='int', help='Sequence stride [Default: %default]')
    parser.add_option('--record_size', dest='record_size', default=32, type='int', help='TF record size [Default: %default]')
    parser.add_option('--label', dest='label', default='train', type='str', help='Train/valid/test label [Default: %default]')
    parser.add_option('--replace_n', dest='replace_n', default=False, action='store_true', help='Replace masked letters with random nucleotides [Default: %default]')
    parser.add_option('--out_dir', dest='out_dir', default='', type='str', help='Output file path [Default: %default]')
    # parser.add_option('--ref_genome', dest='ref_genome', default='', type='str', help='the reference genome')
    
    (options,args) = parser.parse_args()

    n_seqs = options.end_i - options.start_i
    n_records = n_seqs // options.record_size + (1 if n_seqs % options.record_size > 0 else 0)
    
    print("\toptions.end_i : ", options.end_i)
    print("\toptions.start_i : ", options.start_i)
    print("\tn_seqs : ", n_seqs)
    print("\toptions.record_size : ", options.record_size)
    print("\tn_records : ", n_records)
    if options.out_dir != '' and options.out_dir[-1] != '/' :
        options.out_dir += '/'
    
    os.makedirs(options.out_dir + 'data' + options.save_suffix + '/tfrecords', exist_ok=True)
    
    # read species table
    species_df = pd.read_csv(options.out_dir + "species" + options.save_suffix + ".cleaned.csv", sep=',')
    
    # compile species index vector
    n_species = len(species_df)
    species_dict = {}
    assembly_level_dict = {}
    
    # loop over species file rows
    for species_ix, [_, row] in enumerate(species_df.iterrows()) :
        accession_id = row['Accession']
        species_id = accession_id.replace(".", "_").replace("-", "_").replace("(", "").replace(")", "")
        
        species_dict[species_id] = species_ix
        assembly_level_dict[species_id] = row['assembly_level']
    
    # read sequence fasta
    fasta_path = options.out_dir + 'data' + options.save_suffix + '/extracted_fasta/sequences_' + options.label + '.cleaned.fasta'
    print("\tfasta_path: ", fasta_path)
    fasta_sequences = pysam.Fastafile(fasta_path)

    # define TF options
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
    
    tfr_writer = None
    tfr_n_written = 0
    tfr_i = options.part_i * n_records
    
    gtf_cache = {}
    bedt_cache = {}
    
    # for record in fasta_sequences.fetch():
    counter = 0
    for seqname in fasta_sequences.references[options.start_i:options.end_i]:
        print("seqname: ", seqname)
        counter += 1
        # if counter > 10:
        #     break
        sequence = fasta_sequences.fetch(seqname)
        # Init tf record
        if tfr_writer is None:
            tfr_file = options.out_dir + 'data' + options.save_suffix + '/tfrecords/' + options.label + '-' + str(tfr_i) + '.tfr'
            tfr_writer = tf.io.TFRecordWriter(tfr_file, tf_opts)
        seqname_split = seqname.split('|')
        species = seqname_split[1]
        locus_coords = seqname_split[0]
        header = locus_coords.split(':')
        contig = header[0]
        start = int(header[1].split('-')[0])
        end = int(header[1].split('-')[1])
        # print('Processing:', contig, start, end, species)

        # load gtf
        transcriptome, bedt_span = None, None
        if species in gtf_cache :
            transcriptome = gtf_cache[species]
            bedt_span = bedt_cache[species]
        else :
            transcriptome = bgene.Transcriptome(options.out_dir + 'data' + options.save_suffix + '/gtf/' + species + '.59.gtf')
            bedt_span = transcriptome.bedtool_span()
            gtf_cache[species] = transcriptome
            bedt_cache[species] = bedt_span
        
        # read DNA
        seq_dna = fasta_sequences.fetch(seqname)
        # print("seq_dna: ", seq_dna)

        # one hot code (N's as zero; optionally replace N's with random nucleotides)
        seq_1hot = dna_1hot_index(seq_dna, n_sample=options.replace_n)
        
        # # get exon-overlap mask
        # exon_mask = get_exon_mask(contig, start, end, transcriptome, bedt_span, assembly_level=assembly_level_dict[species], seq_len=options.seq_length, chew_bp=2)

        # get exon-overlap mask
        exon_mask, donor_motifs, acceptor_motifs = get_exon_mask(contig, seqname, start, end, transcriptome, bedt_span, fasta_sequences, assembly_level=assembly_level_dict[species], seq_len=options.seq_length, chew_bp=2)

        # print("donor_motifs   : ", donor_motifs)
        # print("acceptor_motifs: ", acceptor_motifs)

        # get repeat mask
        repeat_mask = get_repeat_mask(seq_dna, seq_len=options.seq_length)
        # rep_m_len = len(repeat_mask[repeat_mask == True])
        # # print("rep_m_len: ", rep_m_len)
        # get species index
        species_ix = species_dict[species]
        species_arr = np.array([species_ix], dtype='int32')

        # hash to bytes
        features_dict = {
            'sequence': feature_bytes(seq_1hot),
            'mask': feature_bytes(exon_mask),
            'repeat_mask': feature_bytes(repeat_mask),
            'species': feature_bytes(species_arr),
        }

        # write example
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        tfr_writer.write(example.SerializeToString())
        
        tfr_n_written += 1
        if tfr_n_written >= options.record_size:
            tfr_writer.close()
            tfr_writer = None
            tfr_n_written = 0
            tfr_i += 1
    
    # write success file
    success_file = options.out_dir + 'data' + options.save_suffix + '/tfrecords/' + 'success-' + options.label + '-part-' + str(options.part_i) + '.txt'
    with open(success_file, 'wt') as f :
        f.write('success\n')
    print('Total sequences processed: ', counter, "\n")
    

def feature_bytes(values) :
    values = values.flatten().tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
