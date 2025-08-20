#!/usr/bin/env python
from optparse import OptionParser

import os

import pandas as pd

import util
import shutil

'''
download_reference_genomes.py

Download reference genome fasta files from Ensembl Fungi.
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

    # read species table
    species_df = pd.read_csv(options.out_dir + "species" + options.save_suffix + ".csv", sep=',')
    print("species_df: ", species_df)

    ################################################
    # create output directory system if missing

    if options.out_dir != '' and options.out_dir[-1] != '/' :
        options.out_dir += '/'
    
    # if options.out_dir != '' :
    #     shutil.copy(options.out_dir + "species" + options.save_suffix + ".csv", options.out_dir + "species" + options.save_suffix + ".csv")

    os.makedirs(options.out_dir + 'data' + options.save_suffix, exist_ok=True)
    os.makedirs(options.out_dir + 'data' + options.save_suffix + '/fasta', exist_ok=True)

    download_jobs = []
    uncompress_jobs = []
    
    # loop over species file rows
    for _, row in species_df.iterrows():
        accession_id = row['Accession']
        accession_id_cleaned = accession_id.replace(".", "_").replace("-", "_").replace("(", "").replace(")", "")
        
        species_str = row['Species_str']
        assembly_str = row['assembly'].replace(' ', '_')
        core_str = row['Core_db'].split('_core_59')[0]
        
        species_str2 = species_str[:1].upper() + species_str[1:]
        
        url = ''
        
        if "sm" in options.save_suffix:
            if species_str == core_str :
                url = '\'https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-59/fungi/fasta/' + species_str + '/dna/' + species_str2 + '.' + assembly_str + '.dna_sm.toplevel.fa.gz\''
            else :
                url = '\'https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-59/fungi/fasta/' + core_str + '/' + species_str + '/dna/' + species_str2 + '.' + assembly_str + '.dna_sm.toplevel.fa.gz\''
        # elif "mm" in options.save_suffix:
        else:
            if species_str == core_str :
                url = '\'https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-59/fungi/fasta/' + species_str + '/dna/' + species_str2 + '.' + assembly_str + '.dna.toplevel.fa.gz\''
            else :
                url = '\'https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-59/fungi/fasta/' + core_str + '/' + species_str + '/dna/' + species_str2 + '.' + assembly_str + '.dna.toplevel.fa.gz\''
        
        print("url: ", url)
        local_path = options.out_dir + 'data' + options.save_suffix + '/fasta/' + accession_id_cleaned + '.fasta.gz'
        
        # enqueue fasta jobs
        if not os.path.isfile(local_path) and not os.path.isfile(local_path[:-3]) :
            download_jobs.append('curl -o %s %s' % (local_path, url))
            uncompress_jobs.append('gzip -d %s' % local_path)

        xml_url = '\'https://www.ebi.ac.uk/ena/browser/api/xml/' + accession_id + '?download=true\''
        xml_local_path = options.out_dir + 'data' + options.save_suffix + '/fasta/' + accession_id_cleaned + '.xml'

        # enqueue xml jobs
        if not os.path.isfile(xml_local_path):
            download_jobs.append('curl -o %s %s' % (xml_local_path, xml_url))
        
    util.exec_par(download_jobs, 4, verbose=True)
    util.exec_par(uncompress_jobs, 4, verbose=True)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
