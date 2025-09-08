from optparse import OptionParser

import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import h5py

#########################################
# FUNCTIONS FOR MOTIF TRIMMING & TOMTOM   #
#########################################

def write_meme_file(ppm, bg, fname):
    """
    Write a single-motif MEME file.
    """
    with open(fname, 'w') as f:
        f.write('MEME version 4\n\n')
        f.write('ALPHABET= ACGT\n\n')
        f.write('strands: + -\n\n')
        f.write('Background letter frequencies (from unknown source):\n')
        f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(list(bg)))
        f.write('MOTIF 1 TEMP\n\n')
        f.write('letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n' % ppm.shape[0])
        for s in ppm:
            f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))

def fetch_tomtom_matches(ppm, cwm, is_writing_tomtom_matrix, output_dir,
                          pattern_name, motifs_db, background=[0.25,0.25,0.25,0.25],
                          tomtom_exec_path='tomtom', trim_threshold=0.3, trim_min_length=3):
    """
    Trim the motif (based on cwm scores), write a temporary MEME file and run TomTom
    to match the trimmed motif against a motifs database.
    Returns the TomTom results (using only the first match).
    """
    # Determine trimming positions
    score = np.sum(np.abs(cwm), axis=1)
    trim_thresh = np.max(score) * trim_threshold
    pass_inds = np.where(score >= trim_thresh)[0]
    if len(pass_inds) == 0:
        return pd.DataFrame()
    trimmed = ppm[np.min(pass_inds): np.max(pass_inds) + 1]
    if trimmed.shape[0] < trim_min_length:
        # Do not trim if too short
        trimmed = ppm
    # Write a temporary MEME file for the trimmed PWM
    fd, fname = tempfile.mkstemp()
    os.close(fd)
    write_meme_file(trimmed, background, fname)
    # Create temporary file for TomTom results
    fd2, tomtom_fname = tempfile.mkstemp()
    os.close(fd2)
    # Check that TomTom is callable
    if not shutil.which(tomtom_exec_path):
        raise ValueError(f'`tomtom` executable not found. Please install it.')
    # Run TomTom (adjust options as needed)
    cmd = f'{tomtom_exec_path} -no-ssc -oc . --verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 {fname} {motifs_db} > {tomtom_fname}'
    os.system(cmd)
    try:
        # Read only the second and sixth columns (Target ID and q-value) from TomTom output.
        tomtom_results = pd.read_csv(tomtom_fname, sep="\t", usecols=(1, 5))
    except Exception as e:
        tomtom_results = pd.DataFrame()
    # Clean up temporary files
    os.remove(fname)
    if is_writing_tomtom_matrix:
        output_subdir = os.path.join(output_dir, "tomtom")
        os.makedirs(output_subdir, exist_ok=True)
        output_filepath = os.path.join(output_subdir, f"{pattern_name}.tomtom.tsv")
        shutil.move(tomtom_fname, output_filepath)
    else:
        os.remove(tomtom_fname)
    return tomtom_results

def generate_tomtom_dataframe(modisco_h5py, output_dir, meme_motif_db,
                              is_writing_tomtom_matrix, pattern_groups,
                              top_n_matches=3, tomtom_exec="tomtom",
                              trim_threshold=0.3, trim_min_length=3):
    """
    Iterate over modisco patterns and run TomTom for each.
    Returns a DataFrame with one row per pattern and columns:
      'pattern', 'match0', 'qval0', etc.
    Note: In this example we only use the first (best) match.
    """
    tomtom_results = {}
    for i in range(top_n_matches):
        tomtom_results[f'match{i}'] = []
        tomtom_results[f'qval{i}'] = []
    patterns_list = []
    with h5py.File(modisco_h5py, 'r') as modisco_results:
        for contribution_dir_name in pattern_groups:
            if contribution_dir_name not in modisco_results.keys():
                continue
            metacluster = modisco_results[contribution_dir_name]
            # Sorting patterns by a numeric suffix if available.
            key = lambda x: int(x[0].split("_")[-1])
            for idx, (pattern_name, pattern) in enumerate(sorted(metacluster.items(), key=key)):
                pattern_tag = f'{contribution_dir_name}.{pattern_name}'
                patterns_list.append(pattern_tag)
                ppm = np.array(pattern['sequence'][:])
                cwm = np.array(pattern["contrib_scores"][:])
                r = fetch_tomtom_matches(ppm, cwm,
                                          is_writing_tomtom_matrix=is_writing_tomtom_matrix,
                                          output_dir=output_dir,
                                          pattern_name=pattern_tag,
                                          motifs_db=meme_motif_db,
                                          tomtom_exec_path=tomtom_exec,
                                          trim_threshold=trim_threshold,
                                          trim_min_length=trim_min_length)
                # Use only the top match (if available)
                if not r.empty:
                    # r.iloc[0] contains the best match; adjust column names as needed.
                    target = r.iloc[0][r.columns[0]]
                    qval = r.iloc[0][r.columns[1]]
                else:
                    target = None
                    qval = None
                tomtom_results['match0'].append(target)
                tomtom_results['qval0'].append(qval)
                # (Fill in with None for additional columns if needed.)
                for i in range(1, top_n_matches):
                    tomtom_results[f'match{i}'].append(None)
                    tomtom_results[f'qval{i}'].append(None)
    df = pd.DataFrame({"pattern": patterns_list})
    for key, val in tomtom_results.items():
        df[key] = val
    return df


def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option("--modisco_h5", dest='modisco_h5',
                      help="Path to TF-MoDISco HDF5 results file.")
    parser.add_option('--out_dir', dest='out_dir', default='', type='str',
                      help='Output directory [Default: %default]')
    parser.add_option('--seq_bed', dest='seq_bed', default=None, type='str',
                      help='BED file of sequences')
    # New options for motif matching
    parser.add_option('--meme_db', dest='meme_db', default=None, type='str',
                      help='Known motif database in MEME format')
    parser.add_option('--trim_threshold', dest='trim_threshold', default=0.3, type='float',
                      help='Threshold for trimming seqlets [default: %default]')
    parser.add_option('--trim_min_length', dest='trim_min_length', default=3, type='int',
                      help='Minimum length after trimming [default: %default]')
    parser.add_option('--tomtom_exec', dest='tomtom_exec', default='tomtom', type='str',
                      help='Path to TomTom executable [default: %default]')
    
    (options, args) = parser.parse_args()
    print("Options:")
    print("  seq_bed:", options.seq_bed)
    print("Running TomTom motif matching with meme_db:", options.meme_db)
    pattern_groups = ['pos_patterns', 'neg_patterns']

    motif_mapping = {}
    if options.meme_db:
        tomtom_df = generate_tomtom_dataframe(options.modisco_h5, options.out_dir, options.meme_db,
                                                is_writing_tomtom_matrix=False,
                                                pattern_groups=pattern_groups,
                                                top_n_matches=1,
                                                tomtom_exec=options.tomtom_exec,
                                                trim_threshold=options.trim_threshold,
                                                trim_min_length=options.trim_min_length)
        # Build a mapping from modisco pattern ID (e.g. "pos_patterns/pattern_0") to the best match.
        for i, row in tomtom_df.iterrows():
            # In tomtom_df the pattern is stored as "pos_patterns.pattern_0"; convert to match modisco_df.
            pattern_id = row['pattern']  # e.g. "pos_patterns.pattern_0"
            modisco_pattern_id = pattern_id.replace(".", "/")
            best_match = row['match0'] if pd.notnull(row['match0']) else modisco_pattern_id
            motif_mapping[modisco_pattern_id] = best_match
    print("Motif mapping dictionary:")
    print(motif_mapping)


if __name__ == "__main__":
    main()
