# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

# logomaker import
import logomaker

def read_meme(filename):
	motifs = {}

	with open(filename, "r") as infile:
		motif, width, i = None, None, 0

		for line in infile:
			if motif is None:
				if line[:5] == 'MOTIF':
					motif = line.split()[1]
				else:
					continue

			elif width is None:
				if line[:6] == 'letter':
					width = int(line.split()[5])
					pwm = np.zeros((width, 4))

			elif i < width:
				pwm[i] = list(map(float, line.split()))
				i += 1

			else:
				motifs[motif] = pwm
				motif, width, i = None, None, 0

	return motifs

def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
          *ppm - (np.log(background)*background/np.log(2))[None,:])
    return np.sum(ic,axis=1)


def _plot_weights(array, path, figsize=(10,3), nucleotide_width=0.7, base_height=3):
    """Plot weights as a sequence logo and save to file."""
    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'

    num_positions = len(df)
    figure_width = num_positions * nucleotide_width
    figsize = (figure_width, base_height)
    print("figsize: ", figsize)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    
    crp_logo = logomaker.Logo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())

    # Remove x and y axis labels and ticks
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(path, dpi=300)
    plt.tight_layout()
    plt.close()

	
def make_logo(match, logo_dir, motifs, nucleotide_width, base_height):
	if match == 'NA':
		return

	background = np.array([0.25, 0.25, 0.25, 0.25])
	ppm = motifs[match]
	match = match.replace("/", "_")
	ic = compute_per_position_ic(ppm, background, 0.001)

	_plot_weights(ppm*ic[:, None], path='{}/{}.png'.format(logo_dir, match), nucleotide_width=nucleotide_width, base_height=base_height)
		

# Example usage
if __name__ == "__main__":

    pattern_groups = ['pos_patterns', 'neg_patterns']
    # model_archs = ['unet_small', 'unet_small_retry_1', 'unet_small_retry_2', 'unet_small_bert_aux_drop', 'unet_small_bert_aux_drop_retry_1', 'unet_small_bert_aux_drop_retry_2']#, 'averaged_models/unet_small', 'averaged_models/unet_small_bert_aux_drop']
	
    exps = ['gene_exp_motif_test_RP', 'gene_exp_motif_test_TSS']

    for exp in exps:
        modisco_h5py = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/experiments/motif_LM_fine_tuned_RP_TSS/1_modisco_analysis/{exp}/f0c0/logSED/modisco_results_10000_500.h5'
        is_file = os.path.isfile(modisco_h5py)
        print("is_file: ", is_file, '; ', modisco_h5py)

        nucleotide_width=0.7
        base_height=3
        #############################
        # Visualize TF-modisco motifs
        #############################
        results = {'pattern': [], 'num_seqlets': [], 'modisco_cwm_fwd': [], 'modisco_cwm_rev': []}
        img_path_suffix = 'self_modisco_logo'
        output_dir=f'viz_self_modisco/{exp}'

        top_n_matches=3
        trim_threshold=0.3
        trim_min_length=3
        os.makedirs(output_dir, exist_ok=True)
        modisco_logo_dir = output_dir
        modisco_results = h5py.File(modisco_h5py, 'r')
        tags = []
        for name in pattern_groups:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]
            key = lambda x: int(x[0].split("_")[-1])
            for pattern_name, pattern in sorted(metacluster.items(), key=key):
                print("pattern_name: ", pattern_name)
                print("pattern: ", pattern)
                
                tag = '{}.{}'.format(name, pattern_name)
                tags.append(tag)

                cwm_fwd = np.array(pattern['contrib_scores'][:])
                cwm_rev = cwm_fwd[::-1, ::-1]

                print('cwm_fwd: ', cwm_fwd.shape)
                print('cwm_rev: ', cwm_rev.shape)

                score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
                score_rev = np.sum(np.abs(cwm_rev), axis=1)

                trim_thresh_fwd = np.max(score_fwd) * trim_threshold
                trim_thresh_rev = np.max(score_rev) * trim_threshold

                print('score_fwd: ', score_fwd)
                pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
                pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]
                print('pass_inds_fwd: ', pass_inds_fwd)
                print('pass_inds_fwd: ', pass_inds_fwd.shape)
                print('pass_inds_rev: ', pass_inds_rev.shape)

                start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1)
                start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(np.max(pass_inds_rev) + 4 + 1, len(score_rev) + 1)

                # trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
                # trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

                trimmed_cwm_fwd = cwm_fwd#[start_fwd:end_fwd]
                trimmed_cwm_rev = cwm_rev#[start_rev:end_rev]

                _plot_weights(trimmed_cwm_fwd, path='{}/{}_{}_{}.cwm.fwd.png'.format(modisco_logo_dir, tag, nucleotide_width, base_height))
                _plot_weights(trimmed_cwm_rev, path='{}/{}_{}_{}.cwm.rev.png'.format(modisco_logo_dir, tag, nucleotide_width, base_height))
