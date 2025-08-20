#!/usr/bin/env python
from optparse import OptionParser

import os

import pandas as pd
import numpy as np

import util
import shutil
import pysam
import xml.etree.ElementTree as ET

'''
clean_fasta.py

Clean fasta files (keep only chromosomes and rename to unified name standard).
'''

# function to standardize chromosome names
def _stem_chrom_name(s) :
    if s in ['1'] :
        return 'I'
    if s in ['2'] :
        return 'II'
    if s in ['3'] :
        return'III'
    if s in ['4'] :
        return'IV'
    if s in ['5'] :
        return'V'
    if s in ['6'] :
        return'VI'
    if s in ['7'] :
        return'VII'
    if s in ['8'] :
        return'VIII'
    if s in ['9'] :
        return'IX'
    if s in ['10'] :
        return'X'
    if s in ['11'] :
        return'XI'
    if s in ['12'] :
        return'XII'
    if s in ['13'] :
        return'XIII'
    if s in ['14'] :
        return'XIV'
    if s in ['15'] :
        return'XV'
    if s in ['16'] :
        return'XVI'
    
    return s

# function to standardize assembly levels
def _stem_assembly_level(s) :
    if 'complete' in s or 'full' in s or 'chromosome' in s or 'genome' in s :
        return 'chromosome'
    if 'scaffold' in s or 'contig' in s or 'assembly' in s :
        return 'scaffold'
    
    return 'scaffold'

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('--save_suffix', dest='save_suffix', default='', type='str', help='Suffix of species file [Default: %default]')
    parser.add_option('--assembly_level', dest='assembly_level', default='chromosome', type='str', help='Minimum assembly level [Default: %default]')
    parser.add_option('--min_length', dest='min_length', default=32768, type='int', help='Minimum length of contig [Default: %default]')
    parser.add_option('--no_chrom_filter', dest='no_chrom_filter', default=False, action='store_true', help='Allow any non-yeast chromosome name [Default: %default]')
    parser.add_option('--out_dir', dest='out_dir', default='', type='str', help='Output file path [Default: %default]')
    (options,args) = parser.parse_args()

    if options.out_dir != '' and options.out_dir[-1] != '/' :
        options.out_dir += '/'

    # read species table
    species_df = pd.read_csv(options.out_dir + "species" + options.save_suffix + ".csv", sep=',')
    
    # reference yeast chromsomes (from R64-1-1)
    ref_chroms = [
        'I',
        'II',
        'III',
        'IV',
        'V',
        'VI',
        'VII',
        'VIII',
        'IX',
        'X',
        'XI',
        'XII',
        'XIII',
        'XIV',
        'XV',
        'XVI',
    ]

    assembly_levels = []
    n_chroms = []
    total_lengths = []
    keep_species = []
    
    # loop over species file rows
    for _, row in species_df.iterrows():
        accession_id = row['Accession']
        accession_id_cleaned = accession_id.replace(".", "_").replace("-", "_").replace("(", "").replace(")", "")
        
        print("Cleaning FASTA for Accession = '" + accession_id_cleaned + "'")
        
        fasta_file = options.out_dir + 'data' + options.save_suffix + '/fasta/' + accession_id_cleaned + '.fasta'
        clean_file = options.out_dir + 'data' + options.save_suffix + '/fasta/' + accession_id_cleaned + '.cleaned.fasta'
        xml_file = options.out_dir + 'data' + options.save_suffix + '/fasta/' + accession_id_cleaned + '.xml'

        print("fasta_file: ", fasta_file)
        
        if not os.path.isfile(fasta_file):
        # or not os.path.isfile(xml_file) :
            print(" - [skipping - missing fasta]")
            
            assembly_levels.append('missing')
            n_chroms.append(0)
            total_lengths.append(0)
            keep_species.append(False)
            
            continue
        
        # open fasta (pysam)
        fasta_pysam = pysam.Fastafile(fasta_file)
        
        # extract dictionary of references to lengths
        contig_refs = fasta_pysam.references
        contig_lengths = fasta_pysam.lengths
        
        length_dict = {contig_ref : int(contig_length) for contig_ref, contig_length in zip(contig_refs, contig_lengths)}
        
        # close fasta again
        fasta_pysam.close()
        
        # read xml
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # extract assembly level
        assembly_level = ''
        for child in root[0] :
            if child.tag == 'ASSEMBLY_LEVEL' :
                assembly_level = _stem_assembly_level(str(child.text))
                break
        
        print(" - assembly level  = " + str(assembly_level))
        
        # optionally skip scaffold assemblies (and extract whole chromosomes only)
        if (assembly_level == 'scaffold' and options.assembly_level == 'chromosome') :
            
            print(" - [skipping - scaffold assembly]")
            
            assembly_levels.append(assembly_level)
            n_chroms.append(0)
            total_lengths.append(0)
            keep_species.append(False)
            
            continue
        
        # skip scaffold assemblies for s. cerevisiae (difficult to preserve test set otherwise)
        if (assembly_level == 'scaffold' and 'saccharomyces cerevisiae' in row['Name'].lower()) :
            
            print(" - [skipping - scaffold assembly for s. cerevisiae]")
            
            assembly_levels.append(assembly_level)
            n_chroms.append(0)
            total_lengths.append(0)
            keep_species.append(False)
            
            continue
        
        # accumulate chrom counts and lengths
        n_chrom = 0
        total_length = 0
        
        marked_contigs = {}
        
        # open fasta to read
        with open(fasta_file, 'rt') as fasta_open_r :
            
            # open fasta to write
            with open(clean_file, 'wt') as fasta_open_w :
                write_to_file = False
                
                # loop over rows
                for line_raw in fasta_open_r :
                    line = line_raw.strip()
                    
                    # check if identifier row
                    if line[:1] == '>' :
                        
                        contig_ref = line[1:].split(" ")[0].split("\t")[0]
                        write_to_file = False
                        
                        if contig_ref not in marked_contigs :
                            marked_contigs[contig_ref] = True

                            if ((assembly_level == 'scaffold') or ('saccharomyces cerevisiae' not in row['Name'].lower()) or ('saccharomyces cerevisiae' in row['Name'].lower() and contig_ref in ref_chroms)) and length_dict[contig_ref] >= options.min_length :
                                write_to_file = True
                                
                                if assembly_level == 'chromosome' :
                                    fasta_open_w.write('>chr' + contig_ref + '\n')
                                else :
                                    fasta_open_w.write('>' + contig_ref + '\n')

                                n_chrom += 1
                                total_length += length_dict[contig_ref]
                    
                    elif write_to_file : # write sequence only if it is a valid contig
                        fasta_open_w.write(line + '\n')
        
        assembly_levels.append(assembly_level)
        n_chroms.append(n_chrom)
        total_lengths.append(total_length)
        
        if n_chrom > 0 and total_length > 0 :
        
            keep_species.append(True)
            
            # open cleaned fasta (pysam)
            clean_pysam = pysam.Fastafile(clean_file)

            # extract dictionary of references to lengths
            contig_refs = clean_pysam.references
            contig_lengths = clean_pysam.lengths

            # verify number of chromosomes and total length
            n_chrom_pysam = len(contig_refs)
            total_length_pysam = int(np.sum(np.array(total_length)))

            print(" ==> # chroms     = " + str(n_chrom_pysam))
            print(" ==> total length = " + str(total_length_pysam))

            if n_chrom_pysam == n_chrom and total_length_pysam == total_length :
                print(" ==> (matches running total)")
            else :
                print(" ==> (error - does not match running total!)")

            # close fasta again
            clean_pysam.close()
        else :
            
            keep_species.append(False)
            os.remove(clean_file)
            
            print(" - [skipping - empty cleaned fasta]")
    
    # store cleaned species list
    species_df['assembly_level'] = assembly_levels
    species_df['n_chroms'] = n_chroms
    species_df['total_length'] = total_lengths
    species_df['keep_species'] = keep_species
    
    species_df = species_df.loc[species_df['keep_species'] == True].copy().reset_index(drop=True)
    species_df = species_df.drop(columns=['keep_species'])
    
    print("# kept species = " + str(len(species_df)))
    
    species_df.to_csv(options.out_dir + "species" + options.save_suffix + ".cleaned.csv", index=False, sep=',')
    
    # if options.out_dir != '' :
    #     shutil.copy(options.out_dir + "species" + options.save_suffix + ".cleaned.csv", options.out_dir + "species" + options.save_suffix + ".cleaned.csv")
    
    print("Done.")


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
