import gc
import json
import os
import time
from optparse import OptionParser

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pysam
import pyfaidx
import pyranges as pr
import tensorflow as tf
from baskerville import dataset
from baskerville import gene as bgene
from baskerville import layers
from baskerville import seqnn
from yeast_helpers import *

# Suppress TF log messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

###############################################################################
# Global paths and constants
###############################################################################

root_dir = '/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML'
targets_file = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/cleaned_sheet_RNA-Seq_T0.txt'
gtf_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf'
fasta_file = f'{root_dir}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/fasta/GCA_000146045_2.cleaned.fasta'


def map_chromosome_to_roman(chromosome: str) -> str:
    """
    Map chromosome name from format 'chromosome1' to 'chrI' etc.
    If not found in the mapping, return original chromosome string.
    """
    roman_mapping = {
        'chromosome1': 'chrI',
        'chromosome2': 'chrII',
        'chromosome3': 'chrIII',
        'chromosome4': 'chrIV',
        'chromosome5': 'chrV',
        'chromosome6': 'chrVI',
        'chromosome7': 'chrVII',
        'chromosome8': 'chrVIII',
        'chromosome9': 'chrIX',
        'chromosome10': 'chrX',
        'chromosome11': 'chrXI',
        'chromosome12': 'chrXII',
        'chromosome13': 'chrXIII',
        'chromosome14': 'chrXIV',
        'chromosome15': 'chrXV',
        'chromosome16': 'chrXVI',
    }
    return roman_mapping.get(chromosome, chromosome)


def plot_attention_score(
    predicted_tracks,
    attention_scores,
    chrom='chr1',
    start=0,
    end=1024,
    track_crop=16,
    track_pool=1,
    use_gaussian=False,
    gaussian_sigma=8,
    gaussian_truncate=2,
    plot_start=0,
    plot_end=1024,
    save_suffix='',
    vmin=0.0001,
    vmax=0.005,
    highlight_area=False,
    annotate_features=[],
    annotation_df=None,
    highlight_start=0,
    highlight_end=1,
    highlight_start_y=None,
    highlight_end_y=None,
    example_ix=0,
    track_scale_qtl=0.95,
    track_scale_val=None,
    track_scale=0.02,
    track_clip=0.08,
    fold_index=[0, 1, 2, 3, 4, 5, 6, 7],
    layer_index=[5, 6], # two penultimate blocks
    head_index=[0, 1, 2, 3],
    figsize=(8, 8),
    fig_dpi=600,
    save_figs=False,
    out_dir=""
):
    """
    Plot the attention score matrix (2D heatmap) and the average predicted track
    around the edges, plus optional annotation features.

    Parameters
    ----------
    predicted_tracks : np.ndarray
        Shape: (batch, n_folds, seq_len).
    attention_scores : np.ndarray
        Shape: (batch, n_folds, n_layers, 1, seq_len, seq_len).
    fold_index : list
        Which fold indices to average across. If you pass only [f_ix],
        you'll get that single fold's attention map rather than an average.
    layer_index : list
        Which layers to average across.
    head_index : list
        Which heads to average across.
    """

    from scipy.ndimage import gaussian_filter
    # Constant for attention matrix size
    atten_size = 128

    # 1) Average over chosen folds
    yh = np.mean(predicted_tracks[:, fold_index, :], axis=1)[example_ix, ...]
    # Pool track bins if needed
    if track_pool > 1:
        yh = np.mean(np.reshape(yh, (yh.shape[0] // track_pool, track_pool)), axis=-1)

    # Next lines handle zero-padding and scaling
    track_crop_pooled = track_crop // (2**3)
    track_ratio = yh.shape[0] // (atten_size - 2 * track_crop_pooled)

    if track_scale_val is None:
        track_scale_val = round(np.quantile(yh, q=track_scale_qtl), 4)

    # Zero-pad the normalized track
    yh = np.concatenate(
        [
            np.zeros(track_crop_pooled * track_ratio),
            yh / track_scale_val,
            np.zeros(track_crop_pooled * track_ratio),
        ],
        axis=0,
    )

    # 2) Average attention map over folds, layers, heads
    # attention_scores shape ~ (batch, n_folds, n_layers, [1], seq_len, seq_len)
    att = np.mean(
        np.mean(
            np.mean(
                attention_scores[:, fold_index, ...],
                axis=1,  # average across the chosen folds
            )[
                :, layer_index, ...
            ],  # average across chosen layers
            axis=1,
        )[
            :, head_index, ...
        ],  # average across chosen heads
        axis=1,
    )[example_ix, ...]

    # Optionally apply Gaussian smoothing
    if use_gaussian:
        att = gaussian_filter(att, sigma=gaussian_sigma, truncate=gaussian_truncate)

    # Convert actual coordinates to attention matrix bins
    plot_start_bin = (plot_start - start) // 128
    plot_end_bin = (plot_end - start) // 128

    track_scale = int(track_scale * (plot_end_bin - plot_start_bin))
    track_clip = int(track_clip * (plot_end_bin - plot_start_bin))

    # Highlight region bin
    highlight_start_x_bin = (highlight_start - start) // 128
    highlight_end_x_bin = (highlight_end - start) // 128

    if highlight_start_y is not None and highlight_end_y is not None:
        highlight_start_y_bin = (highlight_start_y - start) // 128
        highlight_end_y_bin = (highlight_end_y - start) // 128
    else:
        highlight_start_y_bin = (highlight_start - start) // 128
        highlight_end_y_bin = (highlight_end - start) // 128

    # Clip attention at vmin
    att[att < vmin] = 0.0

    # Begin plotting
    plt.figure(figsize=figsize)

    plt.imshow(att, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')

    # Draw highlighted rectangle if needed
    if highlight_area:
        rect = patches.Rectangle(
            (highlight_start_x_bin, highlight_start_y_bin),
            min(highlight_end_x_bin - highlight_start_x_bin, atten_size - highlight_start_x_bin),
            min(highlight_end_y_bin - highlight_start_y_bin, atten_size - highlight_start_y_bin),
            linewidth=1,
            edgecolor='magenta',
            facecolor='none',
        )
        plt.gca().add_patch(rect)

    # Draw gene or feature annotations
    for z_order, annotate_dict in enumerate(annotate_features):
        feature_df = annotation_df.query(
            "Feature == '" + annotate_dict['feature'] + "'"
            + (
                (" and " + annotate_dict['filter_query'])
                if annotate_dict.get('filter_query')
                else ""
            )
        )

        for _, row in feature_df.iterrows():
            start_f = row['Start']
            end_f = row['End']
            strand_f = row.get('Strand', '.')

            # Skip if feature length not in [min_len, max_len]
            if (end_f - start_f) < annotate_dict['min_len'] or (end_f - start_f) > annotate_dict['max_len']:
                continue

            start_f_bin = (start_f - start) // 128
            end_f_bin = (end_f - start) // 128

            # Optionally plot gene name text
            if annotate_dict['feature'] == 'gene' and annotate_dict['annotate_text']:
                if start_f_bin > plot_start_bin and end_f_bin < plot_end_bin:
                    gene_name_text = row['gene_name'] if str(row['gene_name']) != "nan" else row['gene_id']
                    arrow_left = "<- " if strand_f == '-' else ""
                    arrow_right = " ->" if strand_f == '+' else ""
                    label_txt = f"{arrow_left}{gene_name_text}{arrow_right}"
                    plt.gca().text(
                        start_f_bin,
                        end_f_bin + 1,
                        label_txt,
                        ha="left",
                        va="bottom",
                        rotation=0,
                        size=8,
                        color=annotate_dict['color'],
                        fontweight='bold',
                    )

            # Draw box, line, or dots
            if annotate_dict['plot_type'] == 'box':
                rect = patches.Rectangle(
                    (start_f_bin, start_f_bin),
                    min(end_f_bin - start_f_bin, atten_size - start_f_bin),
                    min(end_f_bin - start_f_bin, atten_size - start_f_bin),
                    linewidth=2,
                    edgecolor=annotate_dict['color'],
                    facecolor='none',
                )
                plt.gca().add_patch(rect)
            elif annotate_dict['plot_type'] == 'line':
                plt.plot([start_f_bin, end_f_bin], [start_f_bin, end_f_bin],
                         linewidth=2, color=annotate_dict['color'], zorder=z_order)
            elif annotate_dict['plot_type'] == 'dots':
                plt.scatter([start_f_bin], [start_f_bin],
                            marker=annotate_dict['marker'],
                            s=annotate_dict['size'],
                            color=annotate_dict['color'],
                            edgecolor='black',
                            linewidth=0.5,
                            zorder=z_order)
                plt.scatter([end_f_bin], [end_f_bin],
                            marker=annotate_dict['marker'],
                            s=annotate_dict['size'],
                            color=annotate_dict['color'],
                            edgecolor='black',
                            linewidth=0.5,
                            zorder=z_order)

    # Plot predicted track around the matrix
    z_order = len(annotate_features)

    x_len = (plot_end_bin - plot_start_bin) * track_ratio
    x = np.arange(x_len) / track_ratio + plot_start_bin
    y = np.clip(track_scale * yh[plot_start_bin * track_ratio: plot_end_bin * track_ratio], 0, track_clip) + plot_end_bin

    # Fill top track (y-axis)
    plt.gca().fill_between(
        np.linspace(plot_start_bin, plot_end_bin + track_clip + 1, num=y.shape[0] + track_clip + 1),
        track_clip + plot_end_bin + 1,
        y2=plot_end_bin,
        color='white',
        zorder=z_order,
    )
    plt.gca().fill_between(
        x,
        y,
        y2=plot_end_bin,
        color='deepskyblue',
        zorder=z_order + 1,
    )
    plt.plot(
        x,
        y,
        linewidth=1,
        color='black',
        zorder=z_order + 2,
    )

    # Fill right track (x-axis)
    y_len = (plot_end_bin - plot_start_bin) * track_ratio
    y_vertical = np.arange(y_len) / track_ratio + plot_start_bin
    x_track = np.clip(track_scale * yh[plot_start_bin * track_ratio: plot_end_bin * track_ratio], 0, track_clip) + plot_end_bin

    plt.gca().fill_betweenx(
        np.linspace(plot_start_bin - track_clip, plot_end_bin, num=y.shape[0] + track_clip),
        track_clip + plot_end_bin + 1,
        x2=plot_end_bin,
        color='white',
        zorder=z_order,
    )
    plt.gca().fill_betweenx(
        y_vertical,
        x_track,
        x2=plot_end_bin,
        color='deepskyblue',
        zorder=z_order + 1,
    )
    plt.plot(
        x_track,
        y_vertical,
        linewidth=1,
        color='black',
        zorder=z_order + 2,
    )

    # Example bounding frames (optional debug lines)
    plt.xlim(plot_start_bin, plot_end_bin + track_clip + 1)
    plt.ylim(plot_start_bin, plot_end_bin + track_clip + 1)

    plt.xticks([], [])
    plt.yticks([], [])

    # Remove top and right spines
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('white')

    # Axes labels
    plt.gca().set_xlabel(f"Attended on ({chrom}:{plot_start}-{plot_end})  --->",
                         fontsize=12, loc='left')
    plt.gca().set_ylabel(f"Attended by ({chrom}:{plot_start}-{plot_end})  --->",
                         fontsize=12, loc='bottom')

    plt.tight_layout()

    if save_figs:
        save_path_png = os.path.join(
            out_dir, f"{chrom}_{plot_start}-{plot_end}{save_suffix}.png"
        )
        save_path_eps = os.path.join(
            out_dir, f"{chrom}_{plot_start}-{plot_end}{save_suffix}.eps"
        )
        plt.savefig(save_path_png, dpi=fig_dpi)
        plt.savefig(save_path_eps, format='eps')

    plt.show()
    plt.close()


def _get_attention_weights(self, inputs, training=False):
    """
    Custom method to inject into a model definition to extract raw attention weights.
    """
    # Initialise projection layers
    embedding_size = self._value_size * self._num_heads
    seq_len = inputs.shape[1]

    # Compute q, k, v
    q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K]
    k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K]
    v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V]

    # Scale query
    if self._scaling:
        q *= self._key_size ** -0.5

    # Compute content logits
    content_logits = tf.matmul(q + self._r_w_bias, k, transpose_b=True)

    # If no positional features:
    if self._num_position_features == 0:
        logits = content_logits
    else:
        # Project positions
        distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
        positional_encodings = layers.positional_features(
            positions=distances,
            feature_size=self._num_position_features,
            seq_length=seq_len,
            symmetric=self._relative_position_symmetric,
        )

        if training:
            positional_encodings = tf.nn.dropout(
                positional_encodings, rate=self._positional_dropout_rate
            )

        r_k = self._multihead_output(self._r_k_layer, positional_encodings)

        # Add shift for relative logits
        if self._content_position_bias:
            relative_logits = tf.matmul(q + self._r_r_bias, r_k, transpose_b=True)
        else:
            relative_logits = tf.matmul(self._r_r_bias, r_k, transpose_b=True)
            relative_logits = tf.broadcast_to(
                relative_logits,
                shape=(1, self._num_heads, seq_len, 2 * seq_len - 1),
            )

        relative_logits = layers.relative_shift(relative_logits)
        logits = content_logits + relative_logits

    weights = tf.nn.softmax(logits)
    return weights


def get_attention_model(seqnn_model, layer_ix=0, inital_offset=74, offset=11):
    """
    Build a new Keras model that outputs the attention weights from a particular
    attention layer (layer_ix) of the given seqnn_model.
    """
    print(f"layer_ix: {layer_ix}; inital_offset: {inital_offset}; offset: {offset}")
    attention_model = tf.keras.Model(
        seqnn_model.model.layers[1].inputs,
        _get_attention_weights(
            seqnn_model.model.layers[1].layers[inital_offset + offset * layer_ix],
            seqnn_model.model.layers[1].layers[inital_offset + offset * layer_ix - 1].output,
        ),
    )
    return attention_model


def get_lm_attention_weights(
    seqnn_model,
    x,                # (1, seq_len, 170) - final one-hot you want to examine
    do_rc=False,
    layer_ix=0,
    inital_offset=74, # your attention-layer offset
    offset=11,        # step offset between layers
):
    """
    Compute attention weights on an *unmasked* sequence input.
    This uses the final predicted or original sequence, as you like.
    We build an attention model for the desired layer_ix.

    Returns
    -------
    att_scores : np.ndarray
        Shape [1, num_heads, seq_len, seq_len]
    """
    print("* x.shape: ", x.shape)
    # Possibly RC the input
    x_fwd = x
    if do_rc:
        # Reverse-Complement the 4 DNA channels, but watch the other channels carefully
        x_rc = np.concatenate([
            x[0, ..., :4][::-1, ::-1],
            x[0, ..., 4:][::-1, :],
        ], axis=-1)[None, ...]
    
    # Build an attention model for the desired layer
    attention_model = tf.keras.Model(
        seqnn_model.model.layers[1].inputs,
        _get_attention_weights(
            seqnn_model.model.layers[1].layers[inital_offset + offset * layer_ix],
            seqnn_model.model.layers[1].layers[inital_offset + offset * layer_ix - 1].output,
        ),
    )

    # Predict attention
    att_scores_fwd = attention_model.predict(x=[x_fwd], batch_size=1)  # [1, H, T, T]
    if do_rc:
        att_scores_rc = attention_model.predict(x=[x_rc], batch_size=1)
        # average forward & RC
        att_scores = 0.5 * (att_scores_fwd + att_scores_rc[..., ::-1, ::-1])
    else:
        att_scores = att_scores_fwd

    return att_scores


def predict_tracks_and_attention_scores_LM(
    models,
    sequence_one_hot,
    track_scale=1.0,
    track_transform=1.0,
    clip_soft=None,
    n_layers=8,
    score_rc=True,
    attention_offset=74
):
    """
    For language-model type experiments: get attention from each of n_layers.
    Currently placeholders for predicted_tracks; returning zeros if not needed.
    """
    attention_scores = []
    predicted_tracks = []
    for fold_ix in range(len(models)):
        seqnn_model = models[fold_ix]
        print(f">> * {fold_ix}: {seqnn_model}")

        # (Here you could put code to produce coverage if needed)
        # We'll just produce a placeholder array of zeros for predicted_tracks
        # so the shape is (1, n_folds, seq_len).
        seq_len = sequence_one_hot.shape[0]
        placeholder_tracks = np.zeros((1, 1, seq_len), dtype='float32')
        predicted_tracks.append(placeholder_tracks)

        # Collect attention from each layer
        attention_scores_for_fold = []
        for layer_ix in range(n_layers):
            attention_model = get_attention_model(seqnn_model, layer_ix=layer_ix, inital_offset=attention_offset)
            att_scores = attention_model.predict(
                x=[sequence_one_hot[None, ...]], batch_size=1
            )
            if score_rc:
                att_scores_rc = attention_model.predict(
                    x=[sequence_one_hot[None, ::-1, ::-1]], batch_size=1
                )
                att_scores = (att_scores + att_scores_rc[..., ::-1, ::-1]) / 2.0

            # shape: (1, H, T, T)
            # We'll keep a dimension for layer, so expand dims accordingly
            attention_scores_for_fold.append(att_scores[:, None, None, ...])
            attention_model = None
            gc.collect()

        attention_scores_for_fold = np.concatenate(attention_scores_for_fold, axis=2)
        attention_scores.append(attention_scores_for_fold)

    predicted_tracks = np.concatenate(predicted_tracks, axis=1)  # shape (1, n_folds, seq_len)
    attention_scores = np.concatenate(attention_scores, axis=1) # shape (1, n_folds, n_layers, 1, T, T)
    print("predicted_tracks.shape =", predicted_tracks.shape)
    print("attention_scores.shape =", attention_scores.shape)
    return predicted_tracks, attention_scores


def predict_tracks_and_attention_scores(
    models,
    sequence_one_hot,
    track_scale=1.0,
    track_transform=1.0,
    clip_soft=None,
    n_layers=8,
    score_rc=True,
    attention_offset=74
):
    """
    Given a list of trained SeqNN models and a one-hot encoded sequence,
    predict coverage tracks and compute attention maps from each model.

    Returns
    -------
    predicted_tracks : np.ndarray
        Shape: (1, n_folds, seq_length).
    attention_scores : np.ndarray
        Shape: (1, n_folds, n_layers, 1, seq_len, seq_len).
    """
    print("attention_offset: ", attention_offset)
    attention_scores = []
    predicted_tracks = []
    for fold_ix in range(len(models)):
        seqnn_model = models[fold_ix]
        print(f">> * {fold_ix}: {seqnn_model}")
        
        print(f"sequence_one_hot.shape: {sequence_one_hot.shape}")
        # Predict coverage
        yh = seqnn_model(sequence_one_hot[None, ...])[:, None, ...].astype('float32')

        # Undo scale from training
        yh /= track_scale

        # Undo soft-clip
        if clip_soft is not None:
            yh_unclipped = (yh - clip_soft) ** 2 + clip_soft
            unclip_mask_h = (yh > clip_soft)
            yh[unclip_mask_h] = yh_unclipped[unclip_mask_h]

        # Undo sqrt or other transform
        yh = yh ** (1.0 / track_transform)

        predicted_tracks.append(yh)

        # Collect attention from each layer
        attention_scores_for_fold = []
        for layer_ix in range(n_layers):
            attention_model = get_attention_model(seqnn_model, layer_ix=layer_ix, inital_offset=attention_offset)
            att_scores = attention_model.predict(
                x=[sequence_one_hot[None, ...]], batch_size=1
            )
            if score_rc:
                att_scores_rc = attention_model.predict(
                    x=[sequence_one_hot[None, ::-1, ::-1]], batch_size=1
                )
                att_scores = (att_scores + att_scores_rc[..., ::-1, ::-1]) / 2.0

            attention_scores_for_fold.append(att_scores[:, None, None, ...])
            attention_model = None
            gc.collect()

        attention_scores_for_fold = np.concatenate(attention_scores_for_fold, axis=2)
        attention_scores.append(attention_scores_for_fold)

    # Concatenate across folds
    predicted_tracks = np.concatenate(predicted_tracks, axis=1)  # (1, n_folds, seq_len, possibly channels)
    attention_scores = np.concatenate(attention_scores, axis=1)  # (1, n_folds, n_layers, 1, seq_len, seq_len)

    # If your final dimension has >1 channels, average them:
    if predicted_tracks.ndim == 4:
        predicted_tracks = np.mean(predicted_tracks, axis=-1)

    print("predicted_tracks.shape =", predicted_tracks.shape)
    print("attention_scores.shape =", attention_scores.shape)
    return predicted_tracks, attention_scores


def process_seqs(
    start,
    end,
    chrom,
    seq_len,
    fasta_open,
    gene_pr,
    model_type
):
    """
    Obtain one-hot sequence plus annotation data (exons, genes, etc.).
    """
    sequence_one_hot, annotation_df = process_sequence(
        fasta_open, chrom, start, end, gene_pr, seq_len=seq_len, model_type=model_type
    )
    return sequence_one_hot, annotation_df


def main():
    """
    Entry point. Defines CLI options, loads models, loops over a list of genes,
    and plots the attention maps for each layer-set & fold.
    """
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option(
        "--out_dir",
        dest="out_dir",
        default="",
        type="str",
        help="Output file path [Default: %default]",
    )
    parser.add_option(
        "--model_type",
        dest="model_type",
        default="supervised",
        type="str",
        help="Model type: 'self_supervised' or 'supervised'.",
    )
    parser.add_option(
        "--attention_offset",
        dest="attention_offset",
        default=74,
        type=int,
        help="attention offset for the first layer (transformer block).",
    )
    parser.add_option(
        "--LM_exp",
        dest="LM_exp",
        default=False,
        action="store_true",
        help="LM experiment type (if relevant).",
    )
    parser.add_option(
        "--seq_len",
        dest="seq_len",
        default=16384,
        type=int,
        help="Sequence length",
    )
    parser.add_option(
        "--n_reps",
        dest="n_reps",
        default=1,
        type=int,
        help="Number of models (folds) to ensemble",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions",
    )
    parser.add_option(
        "--vmin",
        dest="vmin",
        default=0.0001,
        type=float,
        help="Minimum attention score for plotting",
    )
    parser.add_option(
        "--vmax",
        dest="vmax",
        default=0.05,
        type=float,
        help="Maximum attention score for plotting",
    )
    (options, args) = parser.parse_args()

    # Model config
    seq_len = options.seq_len
    n_reps = options.n_reps
    num_species = 165

    # Read targets
    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    target_index = targets_df.index
    print("target_index: ", target_index)
    print("len(target_index) =", len(target_index))

    # Basic gene annotation as PyRanges
    gene_pr = pr.read_gtf(gtf_file)
    gene_pr = gene_pr[gene_pr.Feature.isin(["gene", "exon", "five_prime_UTR", "three_prime_UTR"])]
    print("len(gene_pr) =", len(gene_pr))

    # Initialize model ensemble
    models = []
    for rep_ix in range(n_reps):
        tf.keras.backend.clear_session()
        print("** Clear tf session")

        if options.LM_exp:
            # Example: using the same model file each time
            params_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train/params.json'
            model_file = f'/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train/model_best.h5'
        else:
            # For supervised or self_supervised
            params_file = f'{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/{options.model_type}/train/f{rep_ix}c0/train/params.json'
            model_file = f"{root_dir}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp/{options.model_type}/train/f{rep_ix}c0/train/model_best.h5"

        print(f"* {options.model_type}; model_file: {model_file}")
        print(f"* {params_file}")

        # Load training parameters
        with open(params_file) as params_open:
            params = json.load(params_open)
            params_model = params["model"]
            params_train = params["train"]

        # Adjust param fields as needed
        params_model["num_features"] = 4
        params_model["seq_length"] = seq_len

        # If self-supervised or LM => Adjust to reflect the additional channels
        if "self_supervised" in options.model_type or "LM" in options.model_type:
            params_model["num_features"] = num_species + 5
            params_train['r64_idx'] = 109

        print("params_model: ", params_model)
        print("params_train: ", params_train)

        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, trunk=False)
        seqnn_model.build_slice(target_index)
        seqnn_model.build_ensemble(options.rc, [0])
        models.append(seqnn_model)
        print(f"> Loaded model {rep_ix}")

    # Initialize FASTA
    fasta_open = pysam.Fastafile(fasta_file)

    # Load full transcriptome
    transcriptome = bgene.Transcriptome(gtf_file)

    # Make output directory
    os.makedirs("attention_viz", exist_ok=True)

    # Example set of genes
    gene_ls = [
        # "YBR091C",
        # "YBR111W-A",
        # "YCL005W-A",
        # "YCR097W",
        # "YDR424C",
        # "YGL033W",
        "YGL076C",
        "YGR001C",
        # "YLR316C",
        # "YPL198W",
        # "YPL283C",
        # "YPR010C-A",
    ]

    # Annotation style
    annotate_features = [
        {
            "feature": "gene",
            "annotate_text": True,
            "filter_query": "Strand == '+'",
            "plot_type": "box",
            "color": "lightgreen",
            "marker": None,
            "min_len": 500,
            "max_len": 1e9,
        },
        {
            "feature": "gene",
            "annotate_text": True,
            "filter_query": "Strand == '-'",
            "plot_type": "box",
            "color": "deepskyblue",
            "marker": None,
            "min_len": 500,
            "max_len": 1e9,
        },
        {
            "feature": "five_prime_UTR",
            "annotate_text": False,
            "filter_query": None,
            "plot_type": "line",
            "color": "magenta",
            "marker": None,
            "min_len": 0,
            "max_len": 1e9,
        },
        {
            "feature": "three_prime_UTR",
            "annotate_text": False,
            "filter_query": None,
            "plot_type": "line",
            "color": "magenta",
            "marker": None,
            "min_len": 0,
            "max_len": 1e9,
        },
        {
            "feature": "exon",
            "annotate_text": False,
            "filter_query": None,
            "plot_type": "line",
            "color": "deepskyblue",
            "marker": None,
            "min_len": 0,
            "max_len": 1e9,
        },
    ]

    # We define sets of layers we'd like to visualize:
    #  - Each of the 8 individually
    #  - All 8 together
    #  - Last 7 (layers 1..7)
    #  - Last 6 (layers 2..7)
    #  - ...
    layer_sets = {
        "Layer1": [0],
        "Layer2": [1],
        "Layer3": [2],
        "Layer4": [3],
        "Layer5": [4],
        "Layer6": [5],
        "Layer7": [6],
        "Layer8": [7],
        "AllLayers": list(range(8)),       # [0..7]
        "Last7": list(range(1, 8)),        # [1..7]
        "Last6": list(range(2, 8)),        # [2..7]
        "Last5": list(range(3, 8)),        # [3..7]
        "Last4": list(range(4, 8)),        # [4..7]
        "Last3": list(range(5, 8)),        # [5..7]
        "Last2": list(range(6, 8)),        # [6..7]
    }

    # For each gene, predict and plot
    for search_gene in gene_ls:
        output_gene_dir = os.path.join(options.out_dir, search_gene)
        os.makedirs(output_gene_dir, exist_ok=True)

        gene_keys = [gk for gk in transcriptome.genes.keys() if search_gene in gk]
        if not gene_keys:
            continue

        gene = transcriptome.genes[gene_keys[0]]
        gene_start = gene.get_exons()[0][0]
        gene_end = gene.get_exons()[-1][1]
        center_pos = (gene_start + gene_end) // 2
        start = center_pos - seq_len // 2
        end = center_pos + seq_len // 2
        chrom = gene.chrom

        print("search_gene:", search_gene)
        print("gene:", gene)
        print("start:", start)
        print("end:", end)
        print("chrom:", chrom)

        sequence_one_hot_wt, annotation_df = process_seqs(
            start,
            end,
            chrom,
            seq_len,
            fasta_open,
            gene_pr,
            model_type=options.model_type
        )
        print("sequence_one_hot_wt:", sequence_one_hot_wt.shape)

        # Predict coverage tracks and attention
        if "LM" in options.model_type:
            predicted_tracks_wt, attention_scores_wt = predict_tracks_and_attention_scores_LM(
                models,
                sequence_one_hot_wt,
                score_rc=options.rc,
                attention_offset=options.attention_offset
            )
        else:
            predicted_tracks_wt, attention_scores_wt = predict_tracks_and_attention_scores(
                models,
                sequence_one_hot_wt,
                score_rc=options.rc,
                attention_offset=options.attention_offset
            )
        print("predicted_tracks_wt:", predicted_tracks_wt.shape)
        print("attention_scores_wt:", attention_scores_wt.shape)

        # We define some track scaling (tweak as appropriate)
        track_scale = 0.1
        track_clip = 0.15

        # Loop over each set of layers and each fold
        for layer_set_name, layer_inds in layer_sets.items():
            layer_dir = os.path.join(output_gene_dir, layer_set_name)
            os.makedirs(layer_dir, exist_ok=True)

            # Plot each fold separately
            for f_ix in range(n_reps):
                suffix = f"_fold_{f_ix}_layers_{layer_set_name}"
                plot_attention_score(
                    predicted_tracks=predicted_tracks_wt,
                    attention_scores=attention_scores_wt,
                    chrom=chrom,
                    start=start,
                    end=end,
                    track_crop=64,
                    track_pool=1,
                    plot_start=start,
                    plot_end=end,
                    save_suffix=suffix,
                    annotate_features=annotate_features,
                    annotation_df=annotation_df,
                    vmin=float(options.vmin),
                    vmax=float(options.vmax),
                    track_scale=track_scale,
                    track_clip=track_clip,
                    # Pass only one fold => no averaging
                    fold_index=[f_ix],
                    # Pass the chosen layers
                    layer_index=layer_inds,
                    track_scale_qtl=0.98,
                    save_figs=True,
                    out_dir=layer_dir
                )

            # Also plot the average of all folds
            suffix = f"_fold_AVG_layers_{layer_set_name}"
            plot_attention_score(
                predicted_tracks=predicted_tracks_wt,
                attention_scores=attention_scores_wt,
                chrom=chrom,
                start=start,
                end=end,
                track_crop=64,
                track_pool=1,
                plot_start=start,
                plot_end=end,
                save_suffix=suffix,
                annotate_features=annotate_features,
                annotation_df=annotation_df,
                vmin=float(options.vmin),
                vmax=float(options.vmax),
                track_scale=track_scale,
                track_clip=track_clip,
                # Average across all folds
                fold_index=list(range(n_reps)),
                # Pass the chosen layers
                layer_index=layer_inds,
                track_scale_qtl=0.98,
                save_figs=True,
                out_dir=layer_dir
            )


if __name__ == "__main__":
    main()
