import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

import baskerville
from baskerville import seqnn
from baskerville import dna

import pysam

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

import intervaltree
import pyBigWig
from scipy.special import rel_entr
import gc

# Helper functions (prediction, attribution, visualization)

# Make one-hot coded sequence
def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    print("make_seq_1hot ", chrm, start, end, seq_len)
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # Extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna.dna_1hot(seq_dna)
    return seq_1hot


# Predict tracks
def predict_tracks(models, sequence_one_hot):

    predicted_tracks = []
    for fold_ix in range(len(models)):

        yh = models[fold_ix](sequence_one_hot[None, ...])[:, None, ...].astype(
            "float16"
        )

        predicted_tracks.append(yh)

    predicted_tracks = np.concatenate(predicted_tracks, axis=1)

    return predicted_tracks


# Helper function to get (padded) one-hot
def process_sequence(fasta_open, chrom, start, end, gene_pr, seq_len=16384, model_type='self_supervised'):
    print(f"In process_sequence, model_type: {model_type}")
    seq_len_actual = end - start

    # Pad sequence to input window size
    start -= (seq_len - seq_len_actual) // 2
    end += (seq_len - seq_len_actual) // 2

    # Get one-hot
    sequence_one_hot = make_seq_1hot(fasta_open, chrom, start, end, seq_len)
    sequence_one_hot = sequence_one_hot.astype("float32")
    num_species = 165
    print("sequence_one_hot.shape = " + str(sequence_one_hot.shape))

    if model_type == 'self_supervised' or 'LM' in model_type:
        new_shape = sequence_one_hot.shape[:-1] + (num_species+1,)
        # Copy the original tensor into the first 4 positions
        x_new = tf.concat([
            sequence_one_hot, 
            tf.zeros(new_shape)
        ], axis=-1)
        print("x_new.shape = " + str(x_new.shape))
        # # Use TensorFlow indexing to set the desired column to 1
        # x_new = tf.tensor_scatter_nd_update(
        #     x_new,
        #     indices=tf.constant([[i, 114] for i in range(x_new.shape[0])]),
        #     updates=tf.ones((x_new.shape[0] * x_new.shape[1],))
        # )
        # Set the 114th column to 1
        x_new = tf.Variable(x_new)  # Convert to a mutable tensor
        x_new[:, 114].assign(tf.ones([tf.shape(x_new)[0]]))
        print("x_new.shape = " + str(x_new.shape))
    elif model_type == 'supervised':
        x_new = sequence_one_hot

    annotation_df = gene_pr.df.query("Chromosome == '" + chrom + "' and ((End >= " + str(int(start)) + " and End < " + str(int(end)) + ") or (Start >= " + str(int(start)) + " and Start < " + str(int(end)) + "))")

    return x_new, annotation_df
    # x = x_new

    # # !!!Change the dimension of the X for fine-tuning
    # # Create a new tensor filled with zeros of the desired shape
    # new_shape = sequence_one_hot.shape[:-1] + (num_species+1,)
    # # Copy the original tensor into the first 4 positions
    # sequence_one_hot_new = tf.concat([
    #     sequence_one_hot, 
    #     tf.zeros(new_shape)
    # ], axis=-1)
    # # Use TensorFlow indexing to set the desired column to 1
    # sequence_one_hot_new = tf.tensor_scatter_nd_update(
    #     sequence_one_hot_new,
    #     indices=tf.constant([[i, j, 114] for i in range(sequence_one_hot_new.shape[0]) for j in range(sequence_one_hot_new.shape[1])]),
    #     updates=tf.ones((sequence_one_hot_new.shape[0] * sequence_one_hot_new.shape[1],))
    # )

    # print("sequence_one_hot.shape = " + str(sequence_one_hot.shape))
    # print("sequence_one_hot_new.dtype = " + str(sequence_one_hot_new.dtype))
    # return sequence_one_hot_new
    # return sequence_one_hot.astype("float32")


def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

    fp = FontProperties(family="DejaVu Sans", weight="bold")

    globscale = 1.35

    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
        "UP": TextPath((-0.488, 0), "$\\Uparrow$", size=1, prop=fp),
        "DN": TextPath((-0.488, 0), "$\\Downarrow$", size=1, prop=fp),
        "(": TextPath((-0.25, 0), "(", size=1, prop=fp),
        ".": TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")": TextPath((-0.1, 0), ")", size=1, prop=fp),
    }

    COLOR_SCHEME = {
        "G": "orange",
        "A": "green",
        "C": "blue",
        "T": "red",
        "UP": "green",
        "DN": "red",
        "(": "black",
        ".": "black",
        ")": "black",
    }

    text = LETTERS[letter]

    chosen_color = COLOR_SCHEME[letter]
    if color is not None:
        chosen_color = color

    t = (
        mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale)
        + mpl.transforms.Affine2D().translate(x, y)
        + ax.transData
    )
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)

    if ax != None:
        ax.add_artist(p)
    return p


def _prediction_input_grad(
    input_sequence,
    model,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft,
    use_mean,
    use_ratio,
    use_logodds,
    subtract_avg,
    prox_bin_index,
    dist_bin_index,
):

    mean_dist_prox_ratio = None
    with tf.GradientTape() as tape:
        tape.watch(input_sequence)

        # predict
        preds = tf.gather(
            model(input_sequence, training=False),
            tf.tile(
                tf.constant(np.array(track_index))[None, :],
                (tf.shape(input_sequence)[0], 1),
            ),
            axis=2,
            batch_dims=1,
        )

        # undo scale
        preds = preds / track_scale

        # undo soft_clip
        if clip_soft is not None:
            preds = tf.where(
                preds > clip_soft, (preds - clip_soft) ** 2 + clip_soft, preds
            )

        # undo sqrt
        preds = preds ** (1.0 / track_transform)

        # aggregate over tracks (average)
        pred = tf.reduce_mean(preds, axis=2)

        if not use_mean:
            if dist_bin_index is None:
                mean_dist = tf.reduce_sum(pred[:, dist_bin_start:dist_bin_end], axis=1)
            else:
                mean_dist = tf.reduce_sum(
                    tf.gather(pred, dist_bin_index, axis=1), axis=1
                )
            if prox_bin_index is None:
                mean_prox = tf.reduce_sum(pred[:, prox_bin_start:prox_bin_end], axis=1)
            else:
                mean_prox = tf.reduce_sum(
                    tf.gather(pred, prox_bin_index, axis=1), axis=1
                )
        else:
            if dist_bin_index is None:
                mean_dist = tf.reduce_mean(pred[:, dist_bin_start:dist_bin_end], axis=1)
            else:
                mean_dist = tf.reduce_mean(
                    tf.gather(pred, dist_bin_index, axis=1), axis=1
                )
            if prox_bin_index is None:
                mean_prox = tf.reduce_mean(pred[:, prox_bin_start:prox_bin_end], axis=1)
            else:
                mean_prox = tf.reduce_mean(
                    tf.gather(pred, prox_bin_index, axis=1), axis=1
                )
        if not use_ratio:
            mean_dist_prox_ratio = tf.math.log(mean_dist + 1e-6)
        else:
            if not use_logodds:
                mean_dist_prox_ratio = tf.math.log(mean_dist / mean_prox + 1e-6)
            else:
                mean_dist_prox_ratio = tf.math.log(
                    (mean_dist / mean_prox) / (1.0 - (mean_dist / mean_prox)) + 1e-6
                )

    input_grad = tape.gradient(mean_dist_prox_ratio, input_sequence)
    if subtract_avg:
        input_grad = input_grad - tf.reduce_mean(input_grad, axis=-1, keepdims=True)
    else:
        input_grad = input_grad

    return input_grad


def get_prediction_gradient_w_rc(
    models,
    sequence_one_hots,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft=None,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    subtract_avg=False,
    fold_index=[0, 1, 2, 3],
):

    # Get gradients for fwd
    pred_grads = get_prediction_gradient(
        models,
        sequence_one_hots,
        prox_bin_start,
        prox_bin_end,
        dist_bin_start,
        dist_bin_end,
        track_index,
        track_scale,
        track_transform,
        clip_soft,
        prox_bin_index,
        dist_bin_index,
        use_mean,
        use_ratio,
        use_logodds,
        subtract_avg,
        fold_index,
    )

    # Get gradients for rev
    sequence_one_hots_rc = [
        sequence_one_hots[example_ix][::-1, ::-1]
        for example_ix in range(len(sequence_one_hots))
    ]

    prox_bin_start_rc = models[0].target_lengths[0] - prox_bin_start - 1
    prox_bin_end_rc = models[0].target_lengths[0] - prox_bin_end - 1

    dist_bin_start_rc = models[0].target_lengths[0] - dist_bin_start - 1
    dist_bin_end_rc = models[0].target_lengths[0] - dist_bin_end - 1

    prox_bin_index_rc = None
    if prox_bin_index is not None:
        prox_bin_index_rc = [
            models[0].target_lengths[0] - prox_bin - 1 for prox_bin in prox_bin_index
        ]

    dist_bin_index_rc = None
    if dist_bin_index is not None:
        dist_bin_index_rc = [
            models[0].target_lengths[0] - dist_bin - 1 for dist_bin in dist_bin_index
        ]

    pred_grads_rc = get_prediction_gradient(
        models,
        sequence_one_hots_rc,
        prox_bin_end_rc,
        prox_bin_start_rc,
        dist_bin_end_rc,
        dist_bin_start_rc,
        track_index,
        track_scale,
        track_transform,
        clip_soft,
        prox_bin_index_rc,
        dist_bin_index_rc,
        use_mean,
        use_ratio,
        use_logodds,
        subtract_avg,
        fold_index,
    )

    pred_grads_avg = [
        (pred_grads[example_ix] + pred_grads_rc[example_ix][::-1, ::-1]) / 2.0
        for example_ix in range(len(sequence_one_hots))
    ]

    return pred_grads, pred_grads_rc, pred_grads_avg


def get_prediction_gradient(
    models,
    sequence_one_hots,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft=None,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    subtract_avg=False,
    fold_index=[0, 1, 2, 3],
):

    pred_grads = np.zeros((len(sequence_one_hots), len(fold_index), 16384, 4))

    for fold_i, fold_ix in enumerate(fold_index):

        prediction_model = models[fold_ix].model.layers[1]

        input_sequence = tf.keras.layers.Input(shape=(16384, 4), name="sequence")

        input_grad = tf.keras.layers.Lambda(
            lambda x: _prediction_input_grad(
                x,
                prediction_model,
                prox_bin_start,
                prox_bin_end,
                dist_bin_start,
                dist_bin_end,
                track_index,
                track_scale,
                track_transform,
                clip_soft,
                use_mean,
                use_ratio,
                use_logodds,
                subtract_avg,
                prox_bin_index,
                dist_bin_index,
            ),
            name="inp_grad",
        )(input_sequence)

        grad_model = tf.keras.models.Model(input_sequence, input_grad)

        with tf.device("/cpu:0"):
            for example_ix in range(len(sequence_one_hots)):
                pred_grads[example_ix, fold_i, ...] = (
                    sequence_one_hots[example_ix]
                    * grad_model.predict(
                        x=[sequence_one_hots[example_ix][None, ...]],
                        batch_size=1,
                        verbose=True,
                    )[0, ...]
                )

        # Run garbage collection before next fold
        prediction_model = None
        gc.collect()

    pred_grads = np.mean(pred_grads, axis=1)
    pred_grads = [
        np.sum(pred_grads[example_ix, ...], axis=-1, keepdims=True)
        * sequence_one_hots[example_ix]
        for example_ix in range(len(sequence_one_hots))
    ]

    return pred_grads


def get_prediction_gradient_noisy_w_rc(
    models,
    sequence_one_hots,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft=None,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    subtract_avg=False,
    fold_index=[0, 1, 2, 3],
    n_samples=5,
    sample_prob=0.75,
):

    # Get gradients for fwd
    pred_grads = get_prediction_gradient_noisy(
        models,
        sequence_one_hots,
        prox_bin_start,
        prox_bin_end,
        dist_bin_start,
        dist_bin_end,
        track_index,
        track_scale,
        track_transform,
        clip_soft,
        prox_bin_index,
        dist_bin_index,
        use_mean,
        use_ratio,
        use_logodds,
        subtract_avg,
        fold_index,
        n_samples,
        sample_prob,
    )

    # Get gradients for rev
    sequence_one_hots_rc = [
        sequence_one_hots[example_ix][::-1, ::-1]
        for example_ix in range(len(sequence_one_hots))
    ]

    prox_bin_start_rc = models[0].target_lengths[0] - prox_bin_start - 1
    prox_bin_end_rc = models[0].target_lengths[0] - prox_bin_end - 1

    dist_bin_start_rc = models[0].target_lengths[0] - dist_bin_start - 1
    dist_bin_end_rc = models[0].target_lengths[0] - dist_bin_end - 1

    prox_bin_index_rc = None
    if prox_bin_index is not None:
        prox_bin_index_rc = [
            models[0].target_lengths[0] - prox_bin - 1 for prox_bin in prox_bin_index
        ]

    dist_bin_index_rc = None
    if dist_bin_index is not None:
        dist_bin_index_rc = [
            models[0].target_lengths[0] - dist_bin - 1 for dist_bin in dist_bin_index
        ]

    pred_grads_rc = get_prediction_gradient_noisy(
        models,
        sequence_one_hots_rc,
        prox_bin_end_rc,
        prox_bin_start_rc,
        dist_bin_end_rc,
        dist_bin_start_rc,
        track_index,
        track_scale,
        track_transform,
        clip_soft,
        prox_bin_index_rc,
        dist_bin_index_rc,
        use_mean,
        use_ratio,
        use_logodds,
        subtract_avg,
        fold_index,
        n_samples,
        sample_prob,
    )

    pred_grads_avg = [
        (pred_grads[example_ix] + pred_grads_rc[example_ix][::-1, ::-1]) / 2.0
        for example_ix in range(len(sequence_one_hots))
    ]

    return pred_grads, pred_grads_rc, pred_grads_avg


def get_prediction_gradient_noisy(
    models,
    sequence_one_hots,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft=None,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
    subtract_avg=False,
    fold_index=[0, 1, 2, 3],
    n_samples=5,
    sample_prob=0.75,
):

    pred_grads = np.zeros((len(sequence_one_hots), len(fold_index), 16384, 4))

    for fold_i, fold_ix in enumerate(fold_index):

        print("fold_ix = " + str(fold_ix))

        prediction_model = models[fold_ix].model.layers[1]

        input_sequence = tf.keras.layers.Input(shape=(16384, 4), name="sequence")

        input_grad = tf.keras.layers.Lambda(
            lambda x: _prediction_input_grad(
                x,
                prediction_model,
                prox_bin_start,
                prox_bin_end,
                dist_bin_start,
                dist_bin_end,
                track_index,
                track_scale,
                track_transform,
                clip_soft,
                use_mean,
                use_ratio,
                use_logodds,
                subtract_avg,
                prox_bin_index,
                dist_bin_index,
            ),
            name="inp_grad",
        )(input_sequence)

        grad_model = tf.keras.models.Model(input_sequence, input_grad)

        with tf.device("/cpu:0"):
            for example_ix in range(len(sequence_one_hots)):

                print("example_ix = " + str(example_ix))

                inp = sequence_one_hots[example_ix][None, ...]

                for sample_ix in range(n_samples):

                    print("sample_ix = " + str(sample_ix))

                    inp_corrupted = np.copy(inp)

                    corrupt_index = np.nonzero(
                        np.random.rand(inp.shape[1]) >= sample_prob
                    )[0]

                    rand_nt_index = np.random.choice(
                        [0, 1, 2, 3], size=(corrupt_index.shape[0],)
                    )

                    inp_corrupted[0, corrupt_index, :] = 0.0
                    inp_corrupted[0, corrupt_index, rand_nt_index] = 1.0

                    pred_grads[example_ix, fold_i, ...] += (
                        sequence_one_hots[example_ix]
                        * grad_model.predict(
                            x=[inp_corrupted], batch_size=1, verbose=True
                        )[0, ...]
                    )

                pred_grads[example_ix, fold_i, ...] /= float(n_samples)

            # Run garbage collection before next example
            gc.collect()

        # Run garbage collection before next fold
        prediction_model = None
        gc.collect()

    pred_grads = np.mean(pred_grads, axis=1)
    pred_grads = [
        np.sum(pred_grads[example_ix, ...], axis=-1, keepdims=True)
        * sequence_one_hots[example_ix]
        for example_ix in range(len(sequence_one_hots))
    ]

    return pred_grads


def _prediction_ism_score(
    pred,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    use_mean,
    use_ratio,
    use_logodds,
    prox_bin_index,
    dist_bin_index,
):

    if not use_mean:
        if dist_bin_index is None:
            mean_dist = np.sum(pred[:, dist_bin_start:dist_bin_end], axis=1)
        else:
            mean_dist = np.sum(pred[:, dist_bin_index], axis=1)
        if prox_bin_index is None:
            mean_prox = np.sum(pred[:, prox_bin_start:prox_bin_end], axis=1)
        else:
            mean_prox = np.sum(pred[:, prox_bin_index], axis=1)
    else:
        if dist_bin_index is None:
            mean_dist = np.mean(pred[:, dist_bin_start:dist_bin_end], axis=1)
        else:
            mean_dist = np.mean(pred[:, dist_bin_index], axis=1)
        if prox_bin_index is None:
            mean_prox = np.mean(pred[:, prox_bin_start:prox_bin_end], axis=1)
        else:
            mean_prox = np.mean(pred[:, prox_bin_index], axis=1)

    if not use_ratio:
        mean_dist_prox_ratio = np.log(mean_dist + 1e-6)
    else:
        if not use_logodds:
            mean_dist_prox_ratio = np.log(mean_dist / mean_prox + 1e-6)
        else:
            mean_dist_prox_ratio = np.log(
                (mean_dist / mean_prox) / (1.0 - (mean_dist / mean_prox)) + 1e-6
            )

    return mean_dist_prox_ratio


def get_ism(
    models,
    sequence_one_hots,
    ism_start,
    ism_end,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft,
    prox_bin_index=None,
    dist_bin_index=None,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
):

    pred_ism = np.zeros((len(sequence_one_hots), len(models), 16384, 4))

    bases = [0, 1, 2, 3]

    for example_ix in range(len(sequence_one_hots)):

        print("example_ix = " + str(example_ix))

        sequence_one_hot_wt = sequence_one_hots[example_ix]

        # get pred
        y_wt = predict_tracks(models, sequence_one_hot_wt)[0, ...][
            ..., track_index
        ].astype("float32")

        # undo scale
        y_wt /= track_scale

        # undo soft_clip
        if clip_soft is not None:
            y_wt_unclipped = (y_wt - clip_soft) ** 2 + clip_soft
            unclip_mask_wt = y_wt > clip_soft

            y_wt[unclip_mask_wt] = y_wt_unclipped[unclip_mask_wt]

        # undo sqrt
        y_wt = y_wt ** (1.0 / track_transform)

        # aggregate over tracks (average)
        y_wt = np.mean(y_wt, axis=-1)

        score_wt = _prediction_ism_score(
            y_wt,
            prox_bin_start,
            prox_bin_end,
            dist_bin_start,
            dist_bin_end,
            use_mean,
            use_ratio,
            use_logodds,
            prox_bin_index,
            dist_bin_index,
        )

        for j in range(ism_start, ism_end):
            for b in bases:
                if sequence_one_hot_wt[j, b] != 1.0:
                    sequence_one_hot_mut = np.copy(sequence_one_hot_wt)
                    sequence_one_hot_mut[j, :] = 0.0
                    sequence_one_hot_mut[j, b] = 1.0

                    # get pred
                    y_mut = predict_tracks(models, sequence_one_hot_mut)[0, ...][
                        ..., track_index
                    ].astype("float32")

                    # undo scale
                    y_mut /= track_scale

                    # undo soft_clip
                    if clip_soft is not None:
                        y_mut_unclipped = (y_mut - clip_soft) ** 2 + clip_soft
                        unclip_mask_mut = y_mut > clip_soft

                        y_mut[unclip_mask_mut] = y_mut_unclipped[unclip_mask_mut]

                    # undo sqrt
                    y_mut = y_mut ** (1.0 / track_transform)

                    # aggregate over tracks (average)
                    y_mut = np.mean(y_mut, axis=-1)

                    score_mut = _prediction_ism_score(
                        y_mut,
                        prox_bin_start,
                        prox_bin_end,
                        dist_bin_start,
                        dist_bin_end,
                        use_mean,
                        use_ratio,
                        use_logodds,
                        prox_bin_index,
                        dist_bin_index,
                    )

                    pred_ism[example_ix, :, j, b] = score_wt - score_mut

        pred_ism[example_ix, ...] = (
            np.tile(np.mean(pred_ism[example_ix, ...], axis=-1)[..., None], (1, 1, 4))
            * sequence_one_hots[example_ix][None, ...]
        )

    pred_ism = np.mean(pred_ism, axis=1)
    pred_ism = [
        pred_ism[example_ix, ...] for example_ix in range(len(sequence_one_hots))
    ]

    return pred_ism


def get_ism_shuffle(
    models,
    sequence_one_hots,
    ism_start,
    ism_end,
    prox_bin_start,
    prox_bin_end,
    dist_bin_start,
    dist_bin_end,
    track_index,
    track_scale,
    track_transform,
    clip_soft,
    prox_bin_index=None,
    dist_bin_index=None,
    window_size=5,
    n_samples=8,
    mononuc_shuffle=False,
    dinuc_shuffle=False,
    use_mean=False,
    use_ratio=True,
    use_logodds=False,
):

    pred_shuffle = np.zeros((len(sequence_one_hots), len(models), 16384, n_samples))
    pred_ism = np.zeros((len(sequence_one_hots), len(models), 16384, 4))

    bases = [0, 1, 2, 3]

    for example_ix in range(len(sequence_one_hots)):

        print("example_ix = " + str(example_ix))

        sequence_one_hot_wt = sequence_one_hots[example_ix]

        # get pred
        y_wt = predict_tracks(models, sequence_one_hot_wt)[0, ...][
            ..., track_index
        ].astype("float32")

        # undo scale
        y_wt /= track_scale

        # undo soft_clip
        if clip_soft is not None:
            y_wt_unclipped = (y_wt - clip_soft) ** 2 + clip_soft
            unclip_mask_wt = y_wt > clip_soft

            y_wt[unclip_mask_wt] = y_wt_unclipped[unclip_mask_wt]

        # undo sqrt
        y_wt = y_wt ** (1.0 / track_transform)

        # aggregate over tracks (average)
        y_wt = np.mean(y_wt, axis=-1)

        score_wt = _prediction_ism_score(
            y_wt,
            prox_bin_start,
            prox_bin_end,
            dist_bin_start,
            dist_bin_end,
            use_mean,
            use_ratio,
            use_logodds,
            prox_bin_index,
            dist_bin_index,
        )

        for j in range(ism_start, ism_end):

            j_start = j - window_size // 2
            j_end = j + window_size // 2 + 1

            pos_index = np.arange(j_end - j_start) + j_start

            for sample_ix in range(n_samples):
                sequence_one_hot_mut = np.copy(sequence_one_hot_wt)
                sequence_one_hot_mut[j_start:j_end, :] = 0.0

                if not mononuc_shuffle and not dinuc_shuffle:
                    nt_index = np.random.choice(bases, size=(j_end - j_start,)).tolist()
                    sequence_one_hot_mut[pos_index, nt_index] = 1.0
                elif mononuc_shuffle:
                    shuffled_pos_index = np.copy(pos_index)
                    np.random.shuffle(shuffled_pos_index)

                    sequence_one_hot_mut[shuffled_pos_index, :] = sequence_one_hot_wt[
                        pos_index, :
                    ]
                else:  # dinuc-shuffle
                    if sample_ix % 2 == 0:
                        shuffled_pos_index = [
                            [pos_index[pos_j], pos_index[pos_j + 1]]
                            if pos_j + 1 < pos_index.shape[0]
                            else [pos_index[pos_j]]
                            for pos_j in range(0, pos_index.shape[0], 2)
                        ]
                    else:
                        pos_index_rev = np.copy(pos_index)[::-1]
                        shuffled_pos_index = [
                            [pos_index_rev[pos_j], pos_index_rev[pos_j + 1]]
                            if pos_j + 1 < pos_index_rev.shape[0]
                            else [pos_index_rev[pos_j]]
                            for pos_j in range(0, pos_index_rev.shape[0], 2)
                        ]

                    shuffled_shuffle_index = np.arange(
                        len(shuffled_pos_index), dtype="int32"
                    )
                    np.random.shuffle(shuffled_shuffle_index)

                    shuffled_pos_index_new = []
                    for pos_tuple_i in range(len(shuffled_pos_index)):
                        shuffled_pos_index_new.extend(
                            shuffled_pos_index[shuffled_shuffle_index[pos_tuple_i]]
                        )

                    shuffled_pos_index = np.array(shuffled_pos_index_new, dtype="int32")
                    sequence_one_hot_mut[shuffled_pos_index, :] = sequence_one_hot_wt[
                        pos_index, :
                    ]

                # get pred
                y_mut = predict_tracks(models, sequence_one_hot_mut)[0, ...][
                    ..., track_index
                ].astype("float32")

                # undo scale
                y_mut /= track_scale

                # undo soft_clip
                if clip_soft is not None:
                    y_mut_unclipped = (y_mut - clip_soft) ** 2 + clip_soft
                    unclip_mask_mut = y_mut > clip_soft

                    y_mut[unclip_mask_mut] = y_mut_unclipped[unclip_mask_mut]

                # undo sqrt
                y_mut = y_mut ** (1.0 / track_transform)

                # aggregate over tracks (average)
                y_mut = np.mean(y_mut, axis=-1)

                score_mut = _prediction_ism_score(
                    y_mut,
                    prox_bin_start,
                    prox_bin_end,
                    dist_bin_start,
                    dist_bin_end,
                    use_mean,
                    use_ratio,
                    use_logodds,
                    prox_bin_index,
                    dist_bin_index,
                )

                pred_shuffle[example_ix, :, j, sample_ix] = score_wt - score_mut

        pred_ism[example_ix, ...] = (
            np.tile(
                np.mean(pred_shuffle[example_ix, ...], axis=-1)[..., None], (1, 1, 4)
            )
            * sequence_one_hots[example_ix][None, ...]
        )

    pred_ism = np.mean(pred_ism, axis=1)
    pred_ism = [
        pred_ism[example_ix, ...] for example_ix in range(len(sequence_one_hots))
    ]

    return pred_ism


def plot_seq_scores(
    importance_scores,
    figsize=(16, 2),
    plot_y_ticks=True,
    y_min=None,
    y_max=None,
    save_figs=False,
    fig_name="default",
):

    importance_scores = importance_scores.T

    fig = plt.figure(figsize=figsize)

    ref_seq = ""
    for j in range(importance_scores.shape[1]):
        argmax_nt = np.argmax(np.abs(importance_scores[:, j]))

        if argmax_nt == 0:
            ref_seq += "A"
        elif argmax_nt == 1:
            ref_seq += "C"
        elif argmax_nt == 2:
            ref_seq += "G"
        elif argmax_nt == 3:
            ref_seq += "T"

    ax = plt.gca()

    for i in range(0, len(ref_seq)):
        mutability_score = np.sum(importance_scores[:, i])
        color = None
        dna_letter_at(ref_seq[i], i + 0.5, 0, mutability_score, ax, color=color)

    plt.sca(ax)
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    plt.xlim((0, len(ref_seq)))

    # plt.axis('off')

    if plot_y_ticks:
        plt.yticks(fontsize=11)
    else:
        plt.yticks([], [])

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        plt.ylim(
            np.min(importance_scores) - 0.1 * np.max(np.abs(importance_scores)),
            np.max(importance_scores) + 0.1 * np.max(np.abs(importance_scores)),
        )

    plt.axhline(y=0.0, color="black", linestyle="-", linewidth=1)

    # for axis in fig.axes :
    #    axis.get_xaxis().set_visible(False)
    #    axis.get_yaxis().set_visible(False)

    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".eps")

    plt.show()


def visualize_input_gradient_pair(
    att_grad_wt, att_grad_mut, plot_start=0, plot_end=100, save_figs=False, fig_name=""
):

    scores_wt = att_grad_wt[plot_start:plot_end, :]
    scores_mut = att_grad_mut[plot_start:plot_end, :]

    y_min = min(np.min(scores_wt), np.min(scores_mut))
    y_max = max(np.max(scores_wt), np.max(scores_mut))

    y_max_abs = max(np.abs(y_min), np.abs(y_max))

    y_min = y_min - 0.05 * y_max_abs
    y_max = y_max + 0.05 * y_max_abs

    print("--- WT ---")
    plot_seq_scores(
        scores_wt,
        y_min=y_min,
        y_max=y_max,
        figsize=(8, 1),
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + "_wt",
    )

    print("--- Mut ---")
    plot_seq_scores(
        scores_mut,
        y_min=y_min,
        y_max=y_max,
        figsize=(8, 1),
        plot_y_ticks=False,
        save_figs=save_figs,
        fig_name=fig_name + "_mut",
    )

def coverage_track_pair_bins_w_ref_compute_scores(
    y_wt,
    y_mut,
    chrom,
    start,
    search_gene,
    center_pos,
    gene_start,
    gene_end,
    poses,
    plot_window=4096,
    normalize_window=4096,
    bin_size=32,
    pad=16,
    gene_slice=None,
):
    """
    Compute scores for a given gene slice region without any track selection 
    or visualization. This function is simplified to only compute scores at 
    the given gene region using the `compute_scores` function.
    """

    import numpy as np

    print("start: ", start)
    print("y_wt.shape: ", y_wt.shape)
    print("y_mut.shape: ", y_mut.shape)

    # Determine plotting windows and bins (even though we don't plot, we need the indices)
    plot_start = center_pos - plot_window // 2
    plot_end = center_pos + plot_window // 2
    plot_start_bin = (plot_start - start) // bin_size - pad
    plot_end_bin = (plot_end - start) // bin_size - pad

    normalize_start = center_pos - normalize_window // 2
    normalize_end = center_pos + normalize_window // 2
    normalize_start_bin = (normalize_start - start) // bin_size - pad
    normalize_end_bin = (normalize_end - start) // bin_size - pad

    center_bin = (center_pos - start) // bin_size - pad
    gene_start_bin = (gene_start - start) // bin_size - pad
    gene_end_bin = (gene_end - start) // bin_size - pad
    mut_bin = (poses[0] - start) // bin_size - pad

    # For this simplified version, we assume y_wt and y_mut are already in final form
    # without needing per-track scaling or transformations.
    # In the original code, tracks were averaged over axes (0, 1, 3). We maintain that.
    y_wt_curr = np.mean(y_wt, axis=(0, 1, 3))
    y_mut_curr = np.mean(y_mut, axis=(0, 1, 3))

    # We do not do normalization or transformations here, as requested.

    if gene_slice is not None:
        print("y_wt_curr[gene_slice].shape: ", y_wt_curr[gene_slice].shape)
        print("y_mut_curr[gene_slice].shape: ", y_mut_curr[gene_slice].shape)

        sum_wt = np.sum(y_wt_curr[gene_slice])
        sum_mut = np.sum(y_mut_curr[gene_slice])
        print(" - sum_wt = " + str(round(sum_wt, 4)))
        print(" - sum_mut = " + str(round(sum_mut, 4)))

        scores = ['SUM','logSUM','sqrtSUM','SAX','D1','logD1','sqrtD1','D2','logD2','sqrtD2','JS','logJS']
        computed_scores = compute_scores(y_wt_curr[gene_slice], y_mut_curr[gene_slice], scores)
        return computed_scores
    else:
        print("No gene_slice provided; no scores computed.")
        return None


def plot_coverage_tracks(
    # Required data
    y_1_in,
    track_indices,
    
    # Basic track naming / color
    track_name='MyTrack',
    track_color='green',
    
    # Coverage transformation parameters
    track_scale=1.0,
    track_transform=1.0, 
    clip_soft=None,
    untransform_old=False,  # if you need that older style of "clip+sqrt" reversion
    
    # Genomic region settings
    start=0,            # Genomic coordinate corresponding to the first bin's left edge
    plot_start_rel=512,
    plot_end_rel=524288 - 512,
    normalize_start_rel=512,
    normalize_end_rel=524288 - 512,
    normalize_counts=False,
    
    # Binning parameters
    bin_size=32,
    pad=16,
    
    # Gene annotation
    gene_slice=None,        # list/array of bin indices for gene exons
    gene_strand='+',        # '+' or '-' or None
    chrom='chrX',
    search_gene='MyGene',
    gene_color='deepskyblue',
    
    # Plot controls
    log_scale=False,
    plot_as_bars=False,
    
    # Figure / saving
    save_figs=False,
    save_suffix='default',
    save_dir='./',
    fig_size=(12, 2),
    dpi=300
):
    """
    Plots a single coverage track y_1_in over a user-defined region.

    Parameters
    ----------
    y_1_in : np.ndarray
        Shape (batch, replicate, length, channels). Coverage output from model.
    track_indices : list of ints
        Which channel indices to average across for the single track.
    track_name : str
        Name for the track (for labeling).
    track_color : str
        Color for the coverage area/bars.
    track_scale : float
        Undo scaling factor. If your coverage was multiplied by track_scale, 
        we do coverage /= track_scale.
    track_transform : float
        Undo a power transform, e.g. coverage **= 1.0 / track_transform.
    clip_soft : float or None
        If used, we remove soft clipping in old style or new style.
    untransform_old : bool
        Whether to use the older style of untransform (clip_soft => (x-clip_soft)**2 + clip_soft).
    start : int
        Genomic coordinate that corresponds to the left edge of y_1_in. 
    plot_start_rel : int
        Number of bases from `start` to begin plotting.
    plot_end_rel : int
        Number of bases from `start` to end plotting.
    normalize_start_rel : int
        Start coordinate (relative) for coverage normalization.
    normalize_end_rel : int
        End coordinate (relative) for coverage normalization.
    normalize_counts : bool
        If True, normalizes coverage by total counts in [normalize_start_rel, normalize_end_rel].
    bin_size : int
        Bin size used by y_1_in in base pairs.
    pad : int
        Additional offset used by your model (subtract from the bin index).
    gene_slice : array-like or None
        Bin indices that define exons for the gene of interest.
    gene_strand : str
        '+' or '-' (just for small arrow annotations, if needed).
    chrom : str
        Chromosome label for final annotation.
    search_gene : str
        Gene name label.
    gene_color : str
        Color used for exon shading.
    log_scale : bool
        If True, apply log2(coverage + 1) transform before plotting.
    plot_as_bars : bool
        If True, use `plt.bar`; else `fill_between`.
    save_figs : bool
        If True, save the figure as PNG/PDF with `save_suffix`.
    fig_size : tuple
        (width, height) in inches.
    dpi : int
        Dots per inch for the figure saving.
    """
    print("* y_1_in shape:", y_1_in.shape)

    # 1) Compute bin-based start/end
    plot_start_bin = plot_start_rel // bin_size - pad
    plot_end_bin   = plot_end_rel   // bin_size - pad

    normalize_start_bin = normalize_start_rel // bin_size - pad
    normalize_end_bin   = normalize_end_rel   // bin_size - pad

    # 2) Extract exons from gene_slice as contiguous blocks
    gene_exons = []
    if gene_slice is not None:
        gene_slice = np.array(gene_slice)
        if len(gene_slice) > 0:
            current_exon = [gene_slice[0]]
            for ix in gene_slice[1:]:
                if ix == current_exon[-1] + 1:
                    current_exon.append(ix)
                else:
                    gene_exons.append(current_exon)
                    current_exon = [ix]
            # Final one
            if current_exon:
                gene_exons.append(current_exon)

    # 3) Copy coverage tensor
    y_1 = np.array(y_1_in, dtype=np.float32)

    # 4) Undo transformations (scale, clip, sqrt, etc.)
    #    We'll do it specifically for the selected channels in track_indices below,
    #    but first let's see if we do old or new logic:
    if untransform_old:
        # Old style
        y_1 /= track_scale
        if clip_soft is not None:
            # old style: (x - clip_soft)**2 + clip_soft
            y_1_unclip = (y_1 - clip_soft)**2 + clip_soft
            mask = (y_1 > clip_soft)
            y_1[mask] = y_1_unclip[mask]
        y_1 = y_1 ** (1.0 / track_transform)
    else:
        # New style
        if clip_soft is not None:
            # new style: (x - clip_soft + 1)**2 + clip_soft - 1
            y_1_unclip = (y_1 - clip_soft + 1)**2 + clip_soft - 1
            mask = (y_1 > clip_soft)
            y_1[mask] = y_1_unclip[mask]
        y_1 = (y_1 + 1) ** (1.0 / track_transform) - 1
        y_1 /= track_scale

    # 5) Reduce to a single coverage track across the chosen channels
    #    e.g. mean over (batch, replicate, channels)
    cov_1_list = []
    for idx in track_indices:
        # shape is (batch, replicate, length, channels)
        # we average over batch, replicate, channel dimension
        print("y_1.shape: ", y_1.shape)
        print("y_1[..., idx].shape: ", y_1[..., idx].shape)
        tmp = np.mean(y_1[..., idx], axis=(0,1))  # shape => (length,)
        cov_1_list.append(tmp[:, None])  # shape => (length,1)

    cov_1 = np.concatenate(cov_1_list, axis=-1)  # shape => (length, #channels_in_track_indices)
    # If multiple channels in track_indices, you can average across them all:
    cov_1 = np.mean(cov_1, axis=-1)  # shape => (length,)

    # 6) Optionally normalize coverage by total counts in [normalize_start_bin, normalize_end_bin]
    if normalize_counts:
        count_1 = np.sum(cov_1[normalize_start_bin:normalize_end_bin])
        # Convert to densities (divide by total) then scale back with that same total
        cov_1 = (cov_1 / count_1) * count_1  # effectively no change if all used the same factor

    # 7) Slice out the plotting region
    cov_1_plot = cov_1[plot_start_bin:plot_end_bin]
    if log_scale:
        cov_1_plot = np.log2(cov_1_plot + 1.0)

    max_y = np.max(cov_1_plot) if len(cov_1_plot) > 0 else 1.0

    # 8) Plot
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)

    x_vals = np.arange(plot_start_bin, plot_start_bin + len(cov_1_plot))

    if plot_as_bars:
        ax.bar(x_vals, cov_1_plot, width=1.0, color=track_color, alpha=0.6, label=track_name)
    else:
        ax.fill_between(x_vals, cov_1_plot, color=track_color, alpha=0.6, label=track_name)

    # 9) Annotate gene exons (shaded rectangles). Each exon block is [exon_start_bin..exon_end_bin].
    for exon in gene_exons:
        exon_start_bin = exon[0] - 0.5
        exon_end_bin   = exon[-1] + 0.5
        # Only shade if overlaps the plotted region
        if exon[-1] >= plot_start_bin and exon[0] < plot_end_bin:
            ax.fill_between(
                [exon_start_bin, exon_end_bin],
                max_y * 0.98,   # top
                max_y,          # bottom
                color=gene_color,
                alpha=0.3,
                zorder=10
            )

    # 10) Label axis
    ax.set_xlim([plot_start_bin, plot_end_bin])
    ax.set_ylim([0, max_y * 1.05])
    ax.set_ylabel("Coverage", fontsize=8)
    ax.set_xlabel(f"{chrom}:{start + plot_start_rel}-{start + plot_end_rel}", fontsize=8)
    ax.set_title(f"{search_gene} ({gene_strand}) - {track_name}", fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()

    # 11) Save or show
    if save_figs:
        fname_png = f"{save_dir}/borzoi_{save_suffix}_{track_name}.png"
        fname_pdf = f"{save_dir}/borzoi_{save_suffix}_{track_name}.pdf"
        plt.savefig(fname_png)
        plt.savefig(fname_pdf)
    plt.show()








def plot_3_coverage(
    y_gt,            # shape (1, 1, length, channels) or (1, length, channels)
    y_wt_sup,        # shape (1, 1, length, channels)
    y_wt_selfsup,    # shape (1, 1, length, channels)
    chrom,
    start,
    center_pos,
    gene_start,
    gene_end,
    plot_window=4096,
    bin_size=32,
    pad=16,
    region_mode='full',  # or 'gene'
    invert_16bp_sum=False,
    # Possibly handle re-scaling if you stored coverage in 16-sums, etc.
    # For now, assume you have already done any needed un-scaling.
    save_figs=False,
    save_suffix="default",
    save_dir="./",

    # Gene annotation
    gene_slice=None,        # list/array of bin indices for gene exons
    gene_strand='+',        # '+' or '-' or None
    search_gene='MyGene',
    gene_color='deepskyblue',
):
    """
    Plot:
      Row 1: Average ground truth
      Row 2: Average supervised coverage
      Row 3: Average self-supervised coverage
    Using a shared Y-axis.
    """

    # 2) Extract exons from gene_slice as contiguous blocks
    gene_exons = []
    if gene_slice is not None:
        gene_slice = np.array(gene_slice)
        if len(gene_slice) > 0:
            current_exon = [gene_slice[0]]
            for ix in gene_slice[1:]:
                if ix == current_exon[-1] + 1:
                    current_exon.append(ix)
                else:
                    gene_exons.append(current_exon)
                    current_exon = [ix]
            # Final one
            if current_exon:
                gene_exons.append(current_exon)

    # Decide on the plotting region
    if region_mode == 'full':
        plot_start = center_pos - plot_window // 2
        plot_end   = center_pos + plot_window // 2
    elif region_mode == 'gene':
        buffer_bp  = 50
        plot_start = gene_start - buffer_bp
        plot_end   = gene_end   + buffer_bp
    else:
        raise ValueError("region_mode must be either 'full' or 'gene'")

    # Convert region to bin coordinates
    plot_start_bin = (plot_start - start) // bin_size - pad
    plot_end_bin   = (plot_end   - start) // bin_size - pad

    # Make sure shapes are consistent
    # y_gt, y_wt_sup, y_wt_selfsup might be shape (1, 1, length, channels)
    # We'll just average over batch, replicate, and channels for this demonstration
    # Or if you do channel-based coverage, adapt as needed.

    y_gt_arr       = np.mean(y_gt, axis=(0,1,3))  # shape: (length,)
    y_sup_arr      = np.mean(y_wt_sup, axis=(0,1,3))
    y_selfsup_arr  = np.mean(y_wt_selfsup, axis=(0,1,3))

    print("y_gt.shape: ", y_gt.shape)
    print("y_wt_sup.shape: ", y_wt_sup.shape)
    print("y_wt_selfsup.shape: ", y_wt_selfsup.shape)

    print("y_gt_arr.shape: ", y_gt_arr.shape)
    print("y_sup_arr.shape: ", y_sup_arr.shape) 
    print("y_selfsup_arr.shape: ", y_selfsup_arr.shape)

    # Slice to region
    y_gt_arr      = y_gt_arr[plot_start_bin:plot_end_bin]
    y_sup_arr     = y_sup_arr[plot_start_bin:plot_end_bin]
    y_selfsup_arr = y_selfsup_arr[plot_start_bin:plot_end_bin]

    # Build save directory
    if save_figs:
        os.makedirs(save_dir, exist_ok=True)

    # Create figure with 3 subplots, share y-axis
    fig, axes = plt.subplots(3, 1, figsize=(20, 3.5), sharex=True, sharey=True)

    # Row 1: Ground truth
    ax_gt = axes[0]
    ax_gt.bar(
        x=np.arange(len(y_gt_arr)),
        height=y_gt_arr,
        width=1.0,
        color="purple",
        alpha=0.9,
        label="Experiment Ground Truth (Avg)"
    )
    # ax_gt.legend(fontsize=11)
    # ax_gt.set_ylabel("GT", rotation=0, labelpad=20)
    ax_gt.spines["top"].set_visible(False)
    ax_gt.spines["right"].set_visible(False)
    ax_gt.spines["left"].set_visible(False)
    # Hide x-axis ticks and labels for the top two
    ax_gt.set_xticks([])
    ax_gt.tick_params(axis='x', which='both', length=0)

    # Row 2: Self-supervised
    ax_selfsup = axes[1]
    ax_selfsup.bar(
        x=np.arange(len(y_selfsup_arr)),
        height=y_selfsup_arr,
        width=1.0,
        color="#ff7f0e",
        alpha=0.9,
        label="Fine-tuned model (Avg)"
    )
    # ax_selfsup.legend(fontsize=11)
    # ax_selfsup.set_ylabel("Self-Sup", rotation=0, labelpad=20)
    ax_selfsup.spines["top"].set_visible(False)
    ax_selfsup.spines["right"].set_visible(False)
    ax_selfsup.spines["left"].set_visible(False)

    # Row 3: Supervised
    ax_sup = axes[2]
    ax_sup.bar(
        x=np.arange(len(y_sup_arr)),
        height=y_sup_arr,
        width=1.0,
        color="#1f77b4",
        alpha=0.9,
        label="Scratch-trained (Avg)"
    )
    # ax_sup.legend(fontsize=11)
    # ax_sup.set_ylabel("Sup", rotation=0, labelpad=20)
    ax_sup.spines["top"].set_visible(False)
    ax_sup.spines["right"].set_visible(False)
    ax_sup.spines["left"].set_visible(False)
    ax_sup.set_xticks([])
    ax_sup.tick_params(axis='x', which='both', length=0)

    # The bottom axis shows the X labels
    # ax_selfsup.set_xlabel(f"{chrom}:{plot_start}-{plot_end} ({region_mode})\nBin index", fontsize=8)

    # # 9) Annotate gene exons (shaded rectangles). Each exon block is [exon_start_bin..exon_end_bin].
    # max_y = np.max([np.max(y_gt_arr), np.max(y_sup_arr), np.max(y_selfsup_arr)])
    # print("gene_exons: ", gene_exons)
    # print("max_y     : ", max_y)
    # for exon in gene_exons:
    #     exon_start_bin = exon[0] - 0.5
    #     exon_end_bin   = exon[-1] + 0.5
    #     # Only shade if overlaps the plotted region
    #     if exon[-1] >= plot_start_bin and exon[0] < plot_end_bin:
    #         ax_gt.fill_between(
    #             [exon_start_bin, exon_end_bin],
    #             max_y * 0.95,   # top
    #             max_y,          # bottom
    #             color=gene_color,
    #             alpha=0.3,
    #             zorder=10
    #         )

    # Title
    fig.suptitle(f"{chrom}:{plot_start}-{plot_end}", fontsize=18)

    plt.tight_layout()

    if save_figs:
        plot_fname = f"averaged_cov{save_suffix}.png"
        plt.savefig(os.path.join(save_dir, plot_fname), dpi=300, bbox_inches='tight')

    plt.show()






def plot_coverage_track_bins(
    y_wt,
    chrom,
    start,
    center_pos,
    gene_start,
    gene_end,
    track_indices,        # list of lists or list of ints for model channels
    track_names,          # not strictly needed now, but kept for consistency
    track_scales,         # same length as track_indices
    track_transforms,     # same length as track_indices
    # Ground truth optional
    y_ground_truth=None,
    ref_track_indices=None,  # list of ints (or list of lists) for reference channels
    # Windowing / region options
    plot_window=4096,
    bin_size=32,
    pad=16,
    region_mode='full',  # 'full' or 'gene'
    invert_16bp_sum=False,
    # Plot controls
    rescale_tracks=True,
    save_figs=False,
    save_suffix="default",
    save_dir="./",
    ax=None
):
    """
    Plot two coverage tracks:
      1) AVERAGE of all reference channels in y_ground_truth (first row).
      2) AVERAGE of all model-predicted tracks in y_wt (second row).

    Parameters
    ----------
    y_wt : np.ndarray
        Model-predicted coverage (WT). Shape typically (batch, replicate, length, channels).
    chrom : str
        Chromosome name for labeling.
    start : int
        Genomic coordinate of the start of y_wt coverage region.
    center_pos : int
        Center position used if region_mode='full'.
    gene_start : int
        Gene start coordinate (if region_mode='gene').
    gene_end : int
        Gene end coordinate (if region_mode='gene').
    track_indices : list
        List of channel indices for the model. If multiple sets, each can be a list of ints or a single int.
        We will un-transform and then average *all* these channels together.
    track_names : list
        Names for each track (not strictly needed here, but retained for consistency).
    track_scales : list
        Scaling factors, same length as track_indices. Used to invert the scaling if rescale_tracks=True.
    track_transforms : list
        Power transforms, same length as track_indices. Used to invert the transform if rescale_tracks=True.
    y_ground_truth : np.ndarray, optional
        Ground truth coverage, shape matched to y_wt. Default=None.
    ref_track_indices : list, optional
        Channel indices for ground truth. We'll average these into one coverage. Default=None.
    plot_window : int
        Window size if region_mode='full'.
    bin_size : int
        Binning factor used in the coverage arrays.
    pad : int
        Index offset in model's output.
    region_mode : str
        'full' => use center_pos ± plot_window/2.
        'gene' => gene_start-50 to gene_end+50.
    invert_16bp_sum : bool
        If True, invert the 16-base window sum to approximate original coverage.
    rescale_tracks : bool
        Whether to apply inverse scaling/power transforms. Default=True.
    save_figs : bool
        If True, save the figure.
    save_suffix : str
        Filename suffix for saving.
    save_dir : str
        Directory in which to save figures.

    Returns
    -------
    None
    """

    if ax is None:
        fig = plt.figure(figsize=(15, 4))
        ax = plt.gca()

    # Decide on plotting region
    if region_mode == 'full':
        plot_start = center_pos - plot_window // 2
        plot_end   = center_pos + plot_window // 2
    elif region_mode == 'gene':
        buffer_bp  = 50
        plot_start = gene_start - buffer_bp
        plot_end   = gene_end + buffer_bp
    else:
        raise ValueError("region_mode must be either 'full' or 'gene'")

    # Convert region to "bin coordinates"
    plot_start_bin = (plot_start - start) // bin_size - pad
    plot_end_bin   = (plot_end   - start) // bin_size - pad

    # Build save directory if needed
    if save_figs:
        os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Prepare figure with 2 rows:
    #   Row 1 for reference coverage (avg of all reference channels)
    #   Row 2 for model coverage (avg of all model tracks)
    # -------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True, sharey=True)

    # If user didn't provide ground truth or ref indices, we skip the top row
    ax_gt = axes[0]
    ax_model = axes[1]

    # 1) Compute and plot reference coverage (averaged)
    if (y_ground_truth is not None) and (ref_track_indices is not None):
        y_gt = np.copy(y_ground_truth)  # shape: (batch, replicate, length, channels) typically

        # If ref_track_indices is a list of lists or a single list, flatten them into one list
        flat_ref_inds = []
        for item in ref_track_indices:
            if isinstance(item, int):
                flat_ref_inds.append(item)
            else:
                # assume it's an iterable
                flat_ref_inds.extend(item)

        # Average across (batch, replicate, those reference channels)
        # Steps:
        #   1) slice the channels
        #   2) optionally average across them
        #   3) average across batch & replicate
        y_gt = y_gt[..., flat_ref_inds]  # shape: (batch, replicate, length, len(flat_ref_inds))
        y_gt = np.mean(y_gt, axis=(0,1,3))  # => shape: (length,)

        if invert_16bp_sum:
            y_gt = invert_16sum(y_gt)

        # Slice to the region
        y_gt = y_gt[plot_start_bin:plot_end_bin]

        ax_gt.bar(
            x=np.arange(len(y_gt)),
            height=y_gt,
            width=1.0,
            color="blue",
            alpha=0.5,
            label="Reference (avg)"
        )
        ax_gt.set_ylabel("Ref.\nAvg", fontsize=8, rotation=0, labelpad=30)
        ax_gt.legend(fontsize=8)
    else:
        # Clear or just say "No ground truth"
        ax_gt.text(0.5, 0.5, "No Ground Truth Available", ha='center', va='center')
        ax_gt.set_ylabel("Ref.\nN/A", fontsize=8, rotation=0, labelpad=25)

    # Hide x-axis labels for the top row
    ax_gt.set_xticklabels([])
    ax_gt.tick_params(axis='x', which='both', length=0)
    ax_gt.spines['top'].set_visible(False)
    ax_gt.spines['right'].set_visible(False)

    # 2) Compute and plot model coverage (AVERAGE of all tracks)
    y_wt_copy = np.copy(y_wt)  # shape: (batch, replicate, length, channels)

    # We'll accumulate untransformed coverage from each track, then average
    # For safety, keep a running sum in an array of shape (length,)
    # We'll also count how many track sets we have
    # Each track in track_indices has a corresponding scale and transform
    # We'll untransform each track and accumulate the coverage.
    coverage_sum = None
    coverage_count = 0

    for t_idx, scale, power_tf in zip(track_indices, track_scales, track_transforms):
        # t_idx could be int or list of ints
        if isinstance(t_idx, int):
            channel_list = [t_idx]
        else:
            channel_list = list(t_idx)

        # Slice out channels
        y_curr = y_wt_copy[..., channel_list]  # shape ~ (batch, rep, length, #channels)
        # Average across these channels
        y_curr = np.mean(y_curr, axis=3)  # shape ~ (batch, rep, length)

        # Undo scale/transform if requested
        if rescale_tracks:
            y_curr /= scale          # shape ~ (batch, rep, length)
            y_curr = y_curr ** (1.0 / power_tf)

        # Now average across batch & replicate
        y_curr = np.mean(y_curr, axis=(0,1))  # shape: (length,)

        # Possibly invert 16-bp sums
        if invert_16bp_sum:
            y_curr = invert_16sum(y_curr)

        # Accumulate
        if coverage_sum is None:
            coverage_sum = np.zeros_like(y_curr)
        coverage_sum += y_curr
        coverage_count += 1

    # Now we have the sum of each track's coverage. Average it
    if coverage_count > 0:
        coverage_avg = coverage_sum / coverage_count
    else:
        coverage_avg = np.zeros(y_wt.shape[2])  # fallback if no tracks

    # Slice to region
    coverage_avg = coverage_avg[plot_start_bin:plot_end_bin]

    # Plot in second row
    ax_model.bar(
        x=np.arange(len(coverage_avg)),
        height=coverage_avg,
        width=1.0,
        color="green",
        alpha=0.5,
        label="Model (avg)"
    )
    ax_model.set_ylabel("Model\nAvg", fontsize=8, rotation=0, labelpad=30)
    ax_model.legend(fontsize=8)
    ax_model.spines['top'].set_visible(False)
    ax_model.spines['right'].set_visible(False)

    # The bottom subplot can show the tick labels
    ax_model.set_xlabel(f"{chrom}:{plot_start}-{plot_end} ({region_mode} mode)\nBin index", fontsize=8)

    # Overall title
    fig.suptitle(f"Averaged Reference vs Averaged Model Coverage\n{chrom}:{plot_start}-{plot_end} ({region_mode})", 
                 fontsize=10)

    # Make layout nicer
    fig.tight_layout()

    # Save if requested
    if save_figs:
        fname = f"yeast_{save_suffix}_avg_ref_model_{region_mode}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')

    plt.show()


# def plot_coverage_track_bins(
#     y_wt,
#     chrom,
#     start,
#     center_pos,
#     gene_start,
#     gene_end,
#     track_indices,
#     track_names,
#     track_scales,
#     track_transforms,
#     # Ground truth optional
#     y_ground_truth=None,
#     ref_track_indices=None,
#     # Windowing / region options
#     plot_window=4096,
#     bin_size=32,
#     pad=16,
#     region_mode='full',  # 'full' or 'gene'
#     invert_16bp_sum=False,
#     # Plot controls
#     rescale_tracks=True,
#     save_figs=False,
#     save_suffix="default",
#     save_dir="./"
# ):
#     """
#     Plot coverage tracks with an option to invert the 16-base window sums
#     and to choose between plotting the full window or the gene region ±50bp.

#     Parameters
#     ----------
#     y_wt : np.ndarray
#         Model-predicted coverage (WT). Expected shape e.g. (batch, 1, length, channels).
#     chrom : str
#         Chromosome name for labeling.
#     start : int
#         Genomic coordinate of the start of y_wt coverage region.
#     center_pos : int
#         Center position used if region_mode='full'.
#     gene_start : int
#         Gene start coordinate (if region_mode='gene').
#     gene_end : int
#         Gene end coordinate (if region_mode='gene').
#     track_indices : list
#         Indices (or channel sets) to slice from y_wt.
#     track_names : list
#         Names for each track (for labeling).
#     track_scales : list
#         Scaling factors applied to each track (inverse will be used if rescale_tracks=True).
#     track_transforms : list
#         Power transforms applied to each track.
#     y_ground_truth : np.ndarray, optional
#         Ground truth coverage, shape matched similarly to y_wt. Default=None.
#     ref_track_indices : list, optional
#         Channel sets for ground truth. Default=None.
#     plot_window : int
#         Window size if region_mode='full'.
#     bin_size : int
#         Binning factor.
#     pad : int
#         Index offset in model's output.
#     region_mode : str
#         'full' => plot a window around `center_pos`.
#         'gene' => plot (gene_start-50, gene_end+50).
#     invert_16bp_sum : bool
#         If True, attempt to invert a 16-base window sum to approximate original coverage.
#     rescale_tracks : bool
#         Whether to apply inverse scaling/power transforms. Default=True.
#     save_figs : bool
#         If True, save the figure.
#     save_suffix : str
#         Filename suffix for saving.
#     save_dir : str
#         Directory in which to save figures.
#     """

#     # Decide on plotting region
#     if region_mode == 'full':
#         plot_start = center_pos - plot_window // 2
#         plot_end   = center_pos + plot_window // 2
#     elif region_mode == 'gene':
#         buffer_bp  = 50
#         plot_start = gene_start - buffer_bp
#         plot_end   = gene_end + buffer_bp
#     else:
#         raise ValueError("region_mode must be either 'full' or 'gene'")

#     # Convert region to "bin coordinates"
#     plot_start_bin = (plot_start - start) // bin_size - pad
#     plot_end_bin   = (plot_end   - start) // bin_size - pad

#     # Build save directory if needed
#     os.makedirs(save_dir, exist_ok=True)

#     # --------------
#     # Create figure and subplots
#     # --------------
#     # We'll have 1 row for ground-truth + 1 row per track
#     n_tracks = len(track_names)
#     n_subplots = 1 + n_tracks  # first row for GT, subsequent for model tracks
#     fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 2 * n_subplots),
#                              sharex=True, sharey=False)

#     # If there's only 1 track, axes may not be a list; force it
#     if n_subplots == 1:
#         axes = [axes]

#     # --------------
#     # 1) Plot ground-truth in the first row
#     # --------------
#     ax_gt = axes[0]

#     # Process ground-truth if needed:
#     y_gt_plot = np.copy(y_ground_truth)  # shape expected ~ (batch, replicate, length, channel)
#     # For demonstration, we do a simple average across batch/replicate/channel
#     y_gt_plot = np.mean(y_gt_plot, axis=(0,1,3))  # adapt as needed

#     # Possibly invert 16-bp sums if you store them that way
#     if invert_16bp_sum:
#         y_gt_plot = invert_16sum(y_gt_plot)

#     # Slice to plotting window
#     y_gt_plot = y_gt_plot[plot_start_bin:plot_end_bin]

#     # Plot the ground truth coverage
#     ax_gt.bar(
#         x=np.arange(len(y_gt_plot)),
#         height=y_gt_plot,
#         width=1.0,
#         color="blue",
#         alpha=0.5,
#         label="Ground Truth"
#     )
#     ax_gt.set_ylabel("GT", fontsize=8, rotation=0, labelpad=25)
#     ax_gt.legend(fontsize=8)

#     # Remove top/right spines
#     ax_gt.spines['top'].set_visible(False)
#     ax_gt.spines['right'].set_visible(False)
#     # Hide x-axis tick labels for the ground-truth subplot
#     ax_gt.set_xticklabels([])
#     ax_gt.tick_params(axis='x', which='both', length=0)

#     # --------------
#     # 2) Plot model coverage tracks in subsequent rows
#     # --------------
#     for i, (track_name, t_idx) in enumerate(zip(track_names, track_indices)):
#         ax = axes[i + 1]  # ground-truth was at index 0, so tracks start at 1

#         y_wt_curr = np.copy(y_wt)
#         # Undo scaling/power transform if requested
#         if (track_scales is not None) and (track_transforms is not None) and rescale_tracks:
#             scale = track_scales[i]
#             power_tf = track_transforms[i]
#             y_wt_curr /= scale
#             y_wt_curr = y_wt_curr ** (1.0 / power_tf)

#         # Average across batch/replicate/channel
#         y_wt_curr = np.mean(y_wt_curr[..., t_idx], axis=(0,1,3))

#         # Possibly invert 16-bp sums
#         if invert_16bp_sum:
#             y_wt_curr = invert_16sum(y_wt_curr)

#         # Slice
#         y_wt_curr = y_wt_curr[plot_start_bin:plot_end_bin]

#         # Plot
#         ax.bar(
#             x=np.arange(len(y_wt_curr)),
#             height=y_wt_curr,
#             width=1.0,
#             color="green",
#             alpha=0.5,
#             label=track_name
#         )

#         # Remove top/right spines
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#         # Hide x-axis labels except for the last subplot
#         if i < n_tracks - 1:
#             ax.set_xticklabels([])
#             ax.tick_params(axis='x', which='both', length=0)

#         # Label on y-axis (optional)
#         ax.set_ylabel(track_name, fontsize=8, rotation=0, labelpad=25)
#         ax.legend(fontsize=8)

#     # --------------
#     # Set an overall title, add axis label on the last subplot, etc.
#     # --------------
#     fig.suptitle(f"{chrom}:{plot_start}-{plot_end} ({region_mode} mode)", fontsize=10)

#     # Only the bottom axes gets the visible X label
#     axes[-1].set_xlabel("Bin index (relative to plot_start_bin)", fontsize=8)

#     # Adjust layout
#     fig.tight_layout()

#     # --------------
#     # Save if requested
#     # --------------
#     if save_figs:
#         os.makedirs(save_dir, exist_ok=True)
#         fname = f"yeast_{save_suffix}_GT_first_{region_mode}.png"
#         plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')

#     plt.show()





# Helper functions (measured RNA-seq coverage loader)


def get_coverage_reader(
    cov_files, target_length, crop_length, blacklist_bed, blacklist_pct=0.5
):

    # open genome coverage files
    cov_opens = [CovFace(cov_file) for cov_file in cov_files]

    # read blacklist regions
    black_chr_trees = read_blacklist(blacklist_bed)

    def _read_coverage(
        chrom,
        start,
        end,
        clip_soft=None,
        clip=None,
        scale=0.01,
        blacklist_pct=blacklist_pct,
        cov_opens=cov_opens,
        target_length=target_length,
        crop_length=crop_length,
        black_chr_trees=black_chr_trees,
    ):

        n_targets = len(cov_opens)

        targets = []

        # for each targets
        for target_i in range(n_targets):

            # extract sequence as BED style
            if start < 0:
                seq_cov_nt = np.concatenate(
                    [np.zeros(-start), cov_opens[target_i].read(chrom, 0, end)], axis=0
                )
            else:
                seq_cov_nt = cov_opens[target_i].read(chrom, start, end)  # start - 1

            # extend to full length
            if seq_cov_nt.shape[0] < end - start:
                seq_cov_nt = np.concatenate(
                    [seq_cov_nt, np.zeros((end - start) - seq_cov_nt.shape[0])], axis=0
                )

            # read coverage
            seq_cov_nt = cov_opens[target_i].read(chrom, start, end)

            # determine baseline coverage
            if target_length >= 8:
                baseline_cov = np.percentile(seq_cov_nt, 100 * blacklist_pct)
                baseline_cov = np.nan_to_num(baseline_cov)
            else:
                baseline_cov = 0

            # set blacklist to baseline
            if chrom in black_chr_trees:
                for black_interval in black_chr_trees[chrom][start:end]:
                    # adjust for sequence indexes
                    black_seq_start = black_interval.begin - start
                    black_seq_end = black_interval.end - start
                    black_seq_values = seq_cov_nt[black_seq_start:black_seq_end]
                    seq_cov_nt[black_seq_start:black_seq_end] = np.clip(
                        black_seq_values, -baseline_cov, baseline_cov
                    )

            # set NaN's to baseline
            nan_mask = np.isnan(seq_cov_nt)
            seq_cov_nt[nan_mask] = baseline_cov

            # sum pool
            seq_cov = (
                seq_cov_nt.reshape(target_length, -1).sum(axis=1, dtype="float32")
                ** 0.75
            )

            # crop
            seq_cov = seq_cov[crop_length:-crop_length]

            # clip
            if clip_soft is not None:
                clip_mask = seq_cov > clip_soft
                seq_cov[clip_mask] = clip_soft + np.sqrt(seq_cov[clip_mask] - clip_soft)
            if clip is not None:
                seq_cov = np.clip(seq_cov, -clip, clip)

            # scale
            seq_cov = scale * seq_cov

            # clip float16 min/max
            seq_cov = np.clip(
                seq_cov, np.finfo(np.float16).min, np.finfo(np.float16).max
            )

            # append to targets
            targets.append(seq_cov.astype("float16")[:, None])

        return np.concatenate(targets, axis=-1)

    def _close_coverage(cov_opens=cov_opens):
        # close genome coverage files
        for cov_open in cov_opens:
            cov_open.close()

    return _read_coverage, _close_coverage


def read_blacklist(blacklist_bed, black_buffer=20):
    """Construct interval trees of blacklist
    regions for each chromosome."""
    black_chr_trees = {}

    if blacklist_bed is not None and os.path.isfile(blacklist_bed):
        for line in open(blacklist_bed):
            a = line.split()
            chrm = a[0]
            start = max(0, int(a[1]) - black_buffer)
            end = int(a[2]) + black_buffer

            if chrm not in black_chr_trees:
                black_chr_trees[chrm] = intervaltree.IntervalTree()

            black_chr_trees[chrm][start:end] = True

    return black_chr_trees


class CovFace:
    def __init__(self, cov_file):
        self.cov_file = cov_file
        self.bigwig = False
        self.bed = False

        cov_ext = os.path.splitext(self.cov_file)[1].lower()
        if cov_ext == ".gz":
            cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

        if cov_ext in [".bed", ".narrowpeak"]:
            self.bed = True
            self.preprocess_bed()

        elif cov_ext in [".bw", ".bigwig"]:
            self.cov_open = pyBigWig.open(self.cov_file, "r")
            self.bigwig = True

        elif cov_ext in [".h5", ".hdf5", ".w5", ".wdf5"]:
            self.cov_open = h5py.File(self.cov_file, "r")

        else:
            print(
                'Cannot identify coverage file extension "%s".' % cov_ext,
                file=sys.stderr,
            )
            exit(1)

    def preprocess_bed(self):
        # read BED
        bed_df = pd.read_csv(
            self.cov_file, sep="\t", usecols=range(3), names=["chr", "start", "end"]
        )

        # for each chromosome
        self.cov_open = {}
        for chrm in bed_df.chr.unique():
            bed_chr_df = bed_df[bed_df.chr == chrm]

            # find max pos
            pos_max = bed_chr_df.end.max()

            # initialize array
            self.cov_open[chrm] = np.zeros(pos_max, dtype="bool")

            # set peaks
            for peak in bed_chr_df.itertuples():
                self.cov_open[peak.chr][peak.start : peak.end] = 1

    def read(self, chrm, start, end):
        if self.bigwig:
            cov = self.cov_open.values(chrm, start, end, numpy=True).astype("float16")

        else:
            if chrm in self.cov_open:
                cov = self.cov_open[chrm][start:end]
                pad_zeros = end - start - len(cov)
                if pad_zeros > 0:
                    cov_pad = np.zeros(pad_zeros, dtype="bool")
                    cov = np.concatenate([cov, cov_pad])
            else:
                print(
                    "WARNING: %s doesn't see %s:%d-%d. Setting to all zeros."
                    % (self.cov_file, chrm, start, end),
                    file=sys.stderr,
                )
                cov = np.zeros(end - start, dtype="float16")

        return cov

    def close(self):
        if not self.bed:
            self.cov_open.close()


def compute_scores(ref_preds, alt_preds, snp_stats, strand_transform=None):
    """Compute SNP scores from reference and alternative predictions for 1D arrays.
    
    Args:
        ref_preds (np.array): Reference allele predictions of shape (seq_length,).
        alt_preds (np.array): Alternative allele predictions of shape (seq_length,).
        snp_stats [str]: List of SAD stats to compute.
        strand_transform (scipy.sparse): Strand transform matrix (seq_length x seq_length).
        
    Returns:
        dict: Dictionary of computed scores keyed by stat name.
    """
    seq_length = ref_preds.shape[0]

    print("* ref_preds.shape: ", ref_preds.shape)
    print("* alt_preds.shape: ", alt_preds.shape)

    # log/sqrt transformations
    ref_preds_log = np.log2(ref_preds + 1)
    alt_preds_log = np.log2(alt_preds + 1)
    ref_preds_sqrt = np.sqrt(ref_preds)
    alt_preds_sqrt = np.sqrt(alt_preds)

    # sums across length (no averaging over shifts since none)
    ref_preds_sum = ref_preds.sum()
    alt_preds_sum = alt_preds.sum()
    ref_preds_log_sum = ref_preds_log.sum()
    alt_preds_log_sum = alt_preds_log.sum()
    ref_preds_sqrt_sum = ref_preds_sqrt.sum()
    alt_preds_sqrt_sum = alt_preds_sqrt.sum()

    # difference arrays
    altref_diff = alt_preds - ref_preds
    altref_log_diff = alt_preds_log - ref_preds_log
    altref_sqrt_diff = alt_preds_sqrt - ref_preds_sqrt

    # initialize scores dict
    scores = {}

    def strand_clip_save(key, score, d2=False):
        # score expected to be 1D
        if strand_transform is not None:
            if d2:
                # For D2 variants: transform squared values, then sqrt after transform
                score = np.power(score, 2)
                score = score @ strand_transform
                score = np.sqrt(score)
            else:
                score = score @ strand_transform
        score = np.clip(score, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores[key] = score.astype("float16")

    # SUM-based stats
    if "SUM" in snp_stats:
        sad = alt_preds_sum - ref_preds_sum
        # scalar
        strand_clip_save("SUM", np.array([sad]))
    if "logSUM" in snp_stats:
        log_sad = alt_preds_log_sum - ref_preds_log_sum
        strand_clip_save("logSUM", np.array([log_sad]))
    if "sqrtSUM" in snp_stats:
        sqrt_sad = alt_preds_sqrt_sum - ref_preds_sqrt_sum
        strand_clip_save("sqrtSUM", np.array([sqrt_sad]))

    # SAX: max absolute difference position
    if "SAX" in snp_stats:
        # altref_diff is 1D now; just pick the position of max abs diff
        max_i = np.argmax(np.abs(altref_diff))
        sax = altref_diff[max_i]
        strand_clip_save("SAX", np.array([sax]))

    # L1 norm of difference vector
    if "D1" in snp_stats:
        sad_d1 = np.linalg.norm(altref_diff, ord=1)
        strand_clip_save("D1", np.array([sad_d1]))
    if "logD1" in snp_stats:
        log_d1 = np.linalg.norm(altref_log_diff, ord=1)
        strand_clip_save("logD1", np.array([log_d1]))
    if "sqrtD1" in snp_stats:
        sqrt_d1 = np.linalg.norm(altref_sqrt_diff, ord=1)
        strand_clip_save("sqrtD1", np.array([sqrt_d1]))

    # L2 norm of difference vector
    if "D2" in snp_stats:
        sad_d2 = np.linalg.norm(altref_diff, ord=2)
        strand_clip_save("D2", np.array([sad_d2]), d2=True)
    if "logD2" in snp_stats:
        log_d2 = np.linalg.norm(altref_log_diff, ord=2)
        strand_clip_save("logD2", np.array([log_d2]), d2=True)
    if "sqrtD2" in snp_stats:
        sqrt_d2 = np.linalg.norm(altref_sqrt_diff, ord=2)
        strand_clip_save("sqrtD2", np.array([sqrt_d2]), d2=True)

    # JS divergence
    # We will add pseudocounts and normalize
    if "JS" in snp_stats:
        pseudocount = np.percentile(ref_preds, 25)
        ref_preds_norm = ref_preds + pseudocount
        ref_preds_norm /= ref_preds_norm.sum()
        alt_preds_norm = alt_preds + pseudocount
        alt_preds_norm /= alt_preds_norm.sum()

        js_dist = (rel_entr(ref_preds_norm, alt_preds_norm).sum() 
                   + rel_entr(alt_preds_norm, ref_preds_norm).sum()) / 2
        strand_clip_save("JS", np.array([js_dist]))

    if "logJS" in snp_stats:
        # For logJS, use the log-transformed arrays
        # Compute pseudocounts from the log distributions
        pseudocount = np.percentile(ref_preds_log, 25)
        ref_preds_log_norm = ref_preds_log + pseudocount
        ref_preds_log_norm /= ref_preds_log_norm.sum()
        alt_preds_log_norm = alt_preds_log + pseudocount
        alt_preds_log_norm /= alt_preds_log_norm.sum()

        log_js_dist = (rel_entr(ref_preds_log_norm, alt_preds_log_norm).sum()
                       + rel_entr(alt_preds_log_norm, ref_preds_log_norm).sum()) / 2
        strand_clip_save("logJS", np.array([log_js_dist]))

    # Predictions
    if "REF" in snp_stats:
        ref_out = np.clip(ref_preds, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores["REF"] = ref_out.astype("float16")
    if "ALT" in snp_stats:
        alt_out = np.clip(alt_preds, np.finfo(np.float16).min, np.finfo(np.float16).max)
        scores["ALT"] = alt_out.astype("float16")

    return scores
