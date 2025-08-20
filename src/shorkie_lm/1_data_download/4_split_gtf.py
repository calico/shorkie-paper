#!/usr/bin/env python
from optparse import OptionParser

import os

import pandas as pd
import numpy as np

import util
import shutil
import pysam
import csv

'''
split_gtf.py

Split gtf files by gene/transcript type (protein-coding / ncRNA / tRNA / rRNA / etc).
'''

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--save_suffix', dest='save_suffix', default='', type='str', help='Suffix of species file [Default: %default]')
    parser.add_option('--out_dir', dest='out_dir', default='', type='str', help='Output file path [Default: %default]')
    (options,args) = parser.parse_args()

    if options.out_dir != '' and options.out_dir[-1] != '/' :
        options.out_dir += '/'

    # read species table
    species_df = pd.read_csv(options.out_dir + "species" + options.save_suffix + ".cleaned.csv", sep=',')
    
    biotypes = ['protein_coding', 'rRNA', 'tRNA', 'ncRNA']
    
    # loop over species file rows
    for _, row in species_df.iterrows():
        accession_id = row['Accession']
        accession_id_cleaned = accession_id.replace(".", "_").replace("-", "_").replace("(", "").replace(")", "")
        
        print("Splitting GTF for Accession = '" + accession_id_cleaned + "'")
        
        gtf_file = options.out_dir + 'data' + options.save_suffix + '/gtf/' + accession_id_cleaned + '.59.gtf'
        
        if not os.path.isfile(gtf_file) :
            print(" - [skipping - missing gtf]")
            continue
        
        # load gene annotation file
        gtf_df = pd.read_csv(gtf_file, sep='\t', skiprows=5, names=['chrom', 'havana_str', 'feature', 'start', 'end', 'feat1', 'strand', 'feat2', 'id_str'])

        #gtf_df = gtf_df.query("feature == 'exon'").copy().reset_index(drop=True)

        gtf_df['gene_id'] = gtf_df['id_str'].apply(lambda x: x.split("gene_id \"")[1].split("\";")[0].split(".")[0])
        gtf_df['gene_biotype'] = gtf_df['id_str'].apply(lambda x: x.split("gene_biotype \"")[1].split("\";")[0].split(".")[0])
        #gtf_df['transcript_biotype'] = gtf_df['id_str'].apply(lambda x: x.split("transcript_biotype \"")[1].split("\";")[0].split(".")[0])

        # loop over each biotype to emit
        for biotype in biotypes :
            gtf_df_sub = gtf_df.query("gene_biotype == '" + biotype + "'").copy().reset_index(drop=True)
            
            # store filtered gtf file
            gtf_df_sub = gtf_df_sub.drop(columns=['gene_id', 'gene_biotype'])

            gtf_file_sub = options.out_dir + 'data' + options.save_suffix + '/gtf/' + accession_id_cleaned + '.' + biotype + '.59.gtf'
            gtf_df_sub.to_csv(gtf_file_sub, sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)
    
    print("Done.")


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
