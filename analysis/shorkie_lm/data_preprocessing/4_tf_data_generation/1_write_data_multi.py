#!/usr/bin/env python
from optparse import OptionParser

import os, sys

import pandas as pd
import numpy as np

import util
import pysam

from basenji.dna_io import dna_1hot, dna_1hot_index
import tensorflow as tf
import json


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    return None


'''
write_data_multi.py

Write one-hot coded sequences to TF records (parallelized).
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--save_suffix', dest='save_suffix', default='', type='str', help='Suffix of species file [Default: %default]')
    parser.add_option('--seq_length', dest='seq_length', default=16384, type='int', help='Sequence length [Default: %default]')
    parser.add_option('--seq_stride', dest='seq_stride', default=4096, type='int', help='Sequence stride [Default: %default]')
    parser.add_option('--part_size', dest='part_size', default=2048, type='int', help='Sequence end index [Default: %default]')
    parser.add_option('--record_size', dest='record_size', default=32, type='int', help='TF record size [Default: %default]')
    parser.add_option('--run_local', dest='run_local', default=False, action='store_true', help='Run locally in parallel [Default: %default]')
    parser.add_option('--processes', dest='processes', default=4, type='int', help='Number of parallel processes [Default: %default]')
    parser.add_option('--replace_n', dest='replace_n', default=False, action='store_true', help='Replace masked letters with random nucleotides [Default: %default]')
    parser.add_option('--out_dir', dest='out_dir', default='', type='str', help='Output file path [Default: %default]')
    parser.add_option('--use_gtf', dest='use_gtf', default=False, action='store_true', help='Extract exon masks from gtf files [Default: %default]')
    
    (options,args) = parser.parse_args()

    if options.out_dir != '' and options.out_dir[-1] != '/' :
        options.out_dir += '/'
    
    os.makedirs(options.out_dir + 'data' + options.save_suffix + '/tfrecords', exist_ok=True)
    

    stats_file = options.out_dir + 'data' + options.save_suffix + '/statistics.json'
    js_data = read_json_file(stats_file)

    # loop over labels
    # for label in ['train', 'valid', 'test'] :
    for label in ['test'] :
        
        print('Label:', label) 
        # # read sequence bed
        # bed_df = pd.read_csv(options.out_dir + 'data' + options.save_suffix + '/sequences_' + label + '.bed', sep='\t', names=['contig', 'start', 'end', 'label', 'species'])
        # n_parts = len(bed_df) // options.part_size + (1 if len(bed_df) % options.part_size > 0 else 0)

        seq_num = js_data[f'{label}_seqs']
        n_parts = seq_num // options.part_size + (1 if seq_num % options.part_size > 0 else 0)
        print(f"Number of sequences for {label}: {seq_num}")
        print(f"Number of parts for {label}: {n_parts}")


        # loop over parts
        write_jobs = []
        for part_i in range(n_parts) :
            success_file = options.out_dir + 'data' + options.save_suffix + '/tfrecords/' + 'success-' + label + '-part-' + str(part_i) + '.txt'
            
            part_start = part_i * options.part_size
            part_end = (part_i + 1) * options.part_size
            if part_end > seq_num:
                part_end = seq_num
            
            if os.path.isfile(success_file) :
                print('Skipping existing part %d' % part_i, file=sys.stderr)
            else:
                cmd = 'python write_data.py'
                if options.use_gtf :
                    cmd = 'python write_data_with_gtf.py'
                
                cmd += ' --save_suffix %s' % options.save_suffix
                cmd += ' --start %d' % part_start
                cmd += ' --end %d' % part_end
                cmd += ' --part %d' % part_i
                cmd += ' --seq_length %d' % options.seq_length
                cmd += ' --seq_stride %d' % options.seq_stride
                cmd += ' --record_size %d' % options.record_size
                if options.replace_n :
                    cmd += ' --replace_n'
                cmd += ' --label %s' % label
                if options.out_dir != '' :
                    cmd += ' --out_dir %s' % options.out_dir
                
                # enqueue jobs locally or on slurm
                if options.run_local :
                    write_jobs.append(cmd)
                else :
                    j = slurm.Job(
                        cmd,
                        name='write_%s-%d' % (label, part_i),
                        out_file='%sdata%s/tfrecords/%s-part-%d.out' % (options.out_dir, options.save_suffix, label, part_i),
                        err_file='%sdata%s/tfrecords/%s-part-%d.err' % (options.out_dir, options.save_suffix, label, part_i),
                        queue='standard', mem=15000, time='12:0:0',
                    ) 
                    write_jobs.append(j)

        # execute jobs
        if options.run_local:
            util.exec_par(write_jobs, options.processes, verbose=True)
        else:
            slurm.multi_run(write_jobs, options.processes, verbose=True, launch_sleep=1, update_sleep=5)
        

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
