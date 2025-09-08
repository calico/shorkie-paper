#!/usr/bin/env python
from optparse import OptionParser

import os
import sys
import time
import pandas as pd
import numpy as np
import pysam
from bed_helper import generate_beds

usage = 'usage: %prog [options] arg'
parser = OptionParser(usage)
parser.add_option('--save_suffix', dest='save_suffix', default='', type='str', help='Suffix of species file [Default: %default]')
parser.add_option('--out_dir', dest='out_dir', default='', type='str', help='Output file path [Default: %default]')
(options,args) = parser.parse_args()

print("options: ", options)
#Load species list
save_suffix = options.save_suffix
out_dir = options.out_dir

species_df = pd.read_csv(out_dir + 'species' + save_suffix + '.cleaned.csv', sep=',').drop(columns=['Unnamed: 0'])

print("len(species_df) = " + str(len(species_df)))

species_df = pd.read_csv(out_dir + 'species' + save_suffix + '.cleaned.csv', sep=',').drop(columns=['Unnamed: 0'])

# # Filter the DataFrame
# species_df = species_df[species_df["Accession"] == "GCA_000026945.1"]
# print("species_df: ", species_df )

species_df["Accession"] = species_df["Accession"].apply(lambda x: x.replace(".", "_").replace("-", "_").replace("(", "").replace(")", ""))


#Specify data generation parameters
seq_length = 16384
seq_stride = 4096
record_size = 32
max_rm_frac = 0.001
contig_padding = 512
biotype_constraints = {'' : 0.925, '.protein_coding' : 0.875, '.rRNA' : 0.03125, '.tRNA' : 0.03125}

# train_exclude_dict = {}
# if save_suffix == "_r64_gtf":
#     #Define species / chroms to be filtered out of training data (in addition to valid / test sets below)
#     train_exclude_dict = {}
# elif save_suffix == "_strains_gtf":
#     #Define species / chroms to be filtered out of training data (in addition to valid / test sets below)
#     train_exclude_dict = {
#         row['Accession'].replace(".", "_").replace("-", "_").replace("(", "").replace(")", "") : ['chrXV', 'chrXVI']
#         for _, row in species_df.iterrows()
#     }
# elif save_suffix == "_fungi_rm_gtf":
#     train_exclude_dict = {}


#Define species / chroms to be filtered out of training data (in addition to valid / test sets below)
train_exclude_dict = {
    row['Accession'].replace(".", "_").replace("-", "_").replace("(", "").replace(")", "") : ['chrXI', 'chrXII', 'chrXIII', 'chrXIV', 'chrXV', 'chrXVI']
    for _, row in species_df.iterrows()
}

#Define held-out validation and test species / chromosomes
#Validation species / chroms
valid_include_dict = {
    'GCA_000146045_2' : ['chrXI', 'chrXIII', 'chrXV']
}

#Test species / chroms
test_include_dict = {
    'GCA_000146045_2' : ['chrXII', 'chrXIV', 'chrXVI']
}

print("train_exclude_dict: ", train_exclude_dict)
print("valid_include_dict: ", valid_include_dict)
print("test_include_dict: ", test_include_dict)

#Generate sequence bed file(s)
generate_beds(
    species_df,
    save_suffix,
    out_dir,
    train_exclude_dict,
    valid_include_dict,
    test_include_dict,
    seq_length=seq_length,
    seq_stride=seq_stride,
    record_size=record_size,
    max_rm_frac=max_rm_frac,
    contig_padding=contig_padding,
    biotype_constraints=biotype_constraints,
)