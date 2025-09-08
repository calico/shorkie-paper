#!/usr/bin/env python
from optparse import OptionParser

import os

import pandas as pd
import numpy as np

import util
import pysam

from basenji.dna_io import dna_1hot, dna_1hot_index
import tensorflow as tf

'''
write_data.py

Write one-hot coded sequences to TF records.
'''

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
    
    (options,args) = parser.parse_args()

    n_seqs = options.end_i - options.start_i
    n_records = n_seqs // options.record_size + (1 if n_seqs % options.record_size > 0 else 0)
    
    if options.out_dir != '' and options.out_dir[-1] != '/' :
        options.out_dir += '/'
    
    os.makedirs(options.out_dir + 'data' + options.save_suffix + '/tfrecords', exist_ok=True)
    
    # read species table
    species_df = pd.read_csv(options.out_dir + "species" + options.save_suffix + ".cleaned.csv", sep=',')
    
    # compile species index vector
    n_species = len(species_df)
    species_dict = {}
    
    # loop over species file rows
    for species_ix, [_, row] in enumerate(species_df.iterrows()) :
        accession_id = row['Accession']
        species_id = accession_id.replace(".", "_").replace("-", "_").replace("(", "").replace(")", "")
        
        species_dict[species_id] = species_ix
    
    # read sequence bed
    bed_df = pd.read_csv(options.out_dir + 'data' + options.save_suffix + '/sequences_' + options.label + '.bed', sep='\t', names=['contig', 'start', 'end', 'label', 'species']).iloc[options.start_i:options.end_i].copy().reset_index(drop=True)
    
    # define TF options
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
    
    tfr_writer = None
    tfr_n_written = 0
    tfr_i = options.part_i * n_records
    
    fasta_cache = {}
    
    # loop over bed rows
    for row_i, [_, row] in enumerate(bed_df.iterrows()) :
        
        # init tf record
        if tfr_writer is None :
            tfr_file = options.out_dir + 'data' + options.save_suffix + '/tfrecords/' + options.label + '-' + str(tfr_i) + '.tfr'
            tfr_writer = tf.io.TFRecordWriter(tfr_file, tf_opts)
        
        contig = row['contig']
        start = row['start']
        end = row['end']
        species = row['species']
        
        # open fasta
        fasta_open = None
        if species in fasta_cache :
            fasta_open = fasta_cache[species]
        else :
            fasta_open = pysam.Fastafile(options.out_dir + 'data' + options.save_suffix + '/fasta/' + species + '.cleaned.fasta')
            fasta_cache[species] = fasta_open
        
        # read DNA
        seq_dna = fasta_open.fetch(contig, start, end)
        
        # one hot code (N's as zero; optionally replace N's with random nucleotides)
        seq_1hot = dna_1hot_index(seq_dna, n_sample=options.replace_n)
        
        # get species index
        species_ix = species_dict[species]
        species_arr = np.array([species_ix], dtype='int32')

        # hash to bytes
        features_dict = {
            'sequence': feature_bytes(seq_1hot),
            'species' : feature_bytes(species_arr),
        }

        # write example
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        tfr_writer.write(example.SerializeToString())
        
        tfr_n_written += 1
        if tfr_n_written >= options.record_size or row_i >= len(bed_df) - 1 :
            tfr_writer.close()
            tfr_writer = None
            tfr_n_written = 0
            tfr_i += 1
    
    # close all opened fasta files
    for species in fasta_cache :
        fasta_cache[species].close()
    
    # write success file
    success_file = options.out_dir + 'data' + options.save_suffix + '/tfrecords/' + 'success-' + options.label + '-part-' + str(options.part_i) + '.txt'
    with open(success_file, 'wt') as f :
        f.write('success\n')
    

def feature_bytes(values) :
    values = values.flatten().tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
