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
def process_sequence(fasta_open, chrom, start, end, gene_pr, seq_len=16384, model_type=None):
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

    if 'self_supervised' in model_type or 'LM' in model_type:
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
    elif 'supervised' in model_type:
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
        plt.yticks(fontsize=12)
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


def plot_coverage_track_pair_bins_w_ref(
    y_wt,
    y_mut,
    chrom,
    start,
    search_gene,
    center_pos,
    gene_start,
    gene_end,
    poses,
    track_indices,
    track_names,
    track_scales,
    track_transforms,
    clip_softs,
    logsum_avg,
    snpweight,
    y_ground_truth=None,  # New parameter
    ref_track_indices=None,
    log_scale=False,
    sqrt_scale=False,
    plot_mut=True,
    plot_window=4096,
    normalize_window=4096,
    bin_size=32,
    pad=16,
    rescale_tracks=True,
    normalize_counts=False,
    save_figs=False,
    save_suffix="default",
    save_dir="./",
    gene_slice=None,
    anno_df=None,
):  
    print("start: ", start)
    print("y_wt.shape: ", y_wt.shape)
    print("y_mut.shape: ", y_mut.shape)
    print("y_ground_truth.shape: ", y_ground_truth.shape)

    print("Plotting coverage track pair bins...")
    print("\tplot_window: " + str(plot_window))
    print("\tnormalize_window: " + str(normalize_window))
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

    # Get annotation positions
    anno_poses = []
    if anno_df is not None:
        anno_poses = anno_df.query(
            "chrom == '"
            + chrom
            + "' and position_hg38 >= "
            + str(plot_start)
            + " and position_hg38 < "
            + str(plot_end)
        )["position_hg38"].values.tolist()

    for track_name, track_index, ref_track_index, track_scale, track_transform, clip_soft in zip(
        track_names, track_indices, ref_track_indices, track_scales, track_transforms, clip_softs
    ):
        print(" - track_name = " + track_name)
        print(" - track_index = " + str(track_index))
        print(" - ref_track_index = " + str(ref_track_index))
        print(" - track_scale = " + str(track_scale))
        print(" - track_transform = " + str(track_transform))
        print(" - clip_soft = " + str(clip_soft))

        # Process WT and MUT
        y_wt_curr = np.array(np.copy(y_wt), dtype=np.float32)
        y_mut_curr = np.array(np.copy(y_mut), dtype=np.float32)

        if y_ground_truth is not None:
            y_gt_curr = np.array(np.copy(y_ground_truth), dtype=np.float32)

        # Undo transformations if needed
        if rescale_tracks:
            # undo scale
            y_wt_curr /= track_scale
            y_mut_curr /= track_scale
            if y_ground_truth is not None:
                y_gt_curr /= track_scale

            # undo soft_clip
            if clip_soft is not None:
                y_wt_curr_unclipped = (y_wt_curr - clip_soft) ** 2 + clip_soft
                y_mut_curr_unclipped = (y_mut_curr - clip_soft) ** 2 + clip_soft
                if y_ground_truth is not None:
                    y_gt_curr_unclipped = (y_gt_curr - clip_soft) ** 2 + clip_soft

                unclip_mask_wt = y_wt_curr > clip_soft
                unclip_mask_mut = y_mut_curr > clip_soft
                if y_ground_truth is not None:
                    unclip_mask_gt = y_gt_curr > clip_soft

                y_wt_curr[unclip_mask_wt] = y_wt_curr_unclipped[unclip_mask_wt]
                y_mut_curr[unclip_mask_mut] = y_mut_curr_unclipped[unclip_mask_mut]
                if y_ground_truth is not None:
                    y_gt_curr[unclip_mask_gt] = y_gt_curr_unclipped[unclip_mask_gt]

            # undo sqrt/power transform
            y_wt_curr = y_wt_curr ** (1.0 / track_transform)
            y_mut_curr = y_mut_curr ** (1.0 / track_transform)
            if y_ground_truth is not None:
                y_gt_curr = y_gt_curr ** (1.0 / track_transform)

        y_wt_curr = np.mean(y_wt_curr[..., track_index], axis=(0, 1, 3))
        y_mut_curr = np.mean(y_mut_curr[..., track_index], axis=(0, 1, 3))
        if y_ground_truth is not None:
            y_gt_curr = np.mean(y_gt_curr[..., ref_track_index], axis=(0, 1, 3))

        # Normalize if requested
        if normalize_counts:
            wt_count = np.sum(y_wt_curr[normalize_start_bin:normalize_end_bin])
            mut_count = np.sum(y_mut_curr[normalize_start_bin:normalize_end_bin])

            y_wt_curr /= wt_count
            y_mut_curr /= mut_count
            y_wt_curr *= wt_count
            y_mut_curr *= wt_count

            if y_ground_truth is not None:
                # For ground truth, we assume we normalize similarly to WT
                gt_count = np.sum(y_gt_curr[normalize_start_bin:normalize_end_bin])
                y_gt_curr /= gt_count
                y_gt_curr *= wt_count

        if gene_slice is not None:
            print("y_wt_curr[gene_slice].shape: ", y_wt_curr[gene_slice].shape)
            print("y_mut_curr[gene_slice].shape: ", y_mut_curr[gene_slice].shape)

            sum_wt = np.sum(y_wt_curr[gene_slice])
            sum_mut = np.sum(y_mut_curr[gene_slice])
            print(" - sum_wt = " + str(round(sum_wt, 4)))
            print(" - sum_mut = " + str(round(sum_mut, 4)))

            scores = ['SUM','logSUM','sqrtSUM','SAX','D1','logD1','sqrtD1','D2','logD2','sqrtD2','JS','logJS']
            computed_scores = compute_scores(y_wt_curr[gene_slice], y_mut_curr[gene_slice], scores)
            print("computed_scores: ", computed_scores)

        # Slice data to the plotting window
        y_wt_curr = y_wt_curr[plot_start_bin:plot_end_bin]
        y_mut_curr = y_mut_curr[plot_start_bin:plot_end_bin]
        if y_ground_truth is not None:
            y_gt_curr = y_gt_curr[plot_start_bin:plot_end_bin]

        if log_scale:
            y_wt_curr = np.log2(y_wt_curr + 1.0)
            y_mut_curr = np.log2(y_mut_curr + 1.0)
            if y_ground_truth is not None:
                y_gt_curr = np.log2(y_gt_curr + 1.0)
        elif sqrt_scale:
            y_wt_curr = np.sqrt(y_wt_curr + 1.0)
            y_mut_curr = np.sqrt(y_mut_curr + 1.0)
            if y_ground_truth is not None:
                y_gt_curr = np.sqrt(y_gt_curr + 1.0)

        print("* y_wt_curr.shape = " + str(y_wt_curr.shape))
        print("* y_mut_curr.shape = " + str(y_mut_curr.shape))

        max_y_wt = np.max(y_wt_curr)
        max_y_mut = np.max(y_mut_curr)
        max_y = max(max_y_wt, max_y_mut)
        if y_ground_truth is not None:
            max_y_gt = np.max(y_gt_curr)
            max_y = max(max_y, max_y_gt)

        print(" - max_y_wt = " + str(round(max_y_wt, 4)))
        print(" - max_y_mut = " + str(round(max_y_mut, 4)))
        print(" -- (max_y = " + str(round(max_y, 4)) + ")")

        # Create two subplots: top for WT/Mut, bottom for Ground Truth
        if y_ground_truth is not None:
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        else:
            f, ax1 = plt.subplots(1, 1, figsize=(12, 2))
            ax2 = None

        print(" - track_indices = " + str(track_index))
        print(" - ref_track_indices = " + str(ref_track_index))
        print(" - plot_start_bin = " + str(plot_start_bin))
        print(" - plot_end_bin = " + str(plot_end_bin))
        print(" - center_bin = " + str(center_bin))
        print(" - gene_start_bin = " + str(gene_start_bin)) 
        print(" - gene_end_bin = " + str(gene_end_bin))
        print(" - mut_bin = " + str(mut_bin))
        print(" - y_wt_curr: " + str(y_wt_curr.shape))

        # Annotate logsum_avg and snpweight in the first subplot
        annotation_x = plot_start_bin + 0.1 * (plot_end_bin - plot_start_bin)  # Example x-coordinate
        annotation_y = 0.96 * max_y  # Example y-coordinate based on plot scale

        ax1.text(
            annotation_x,
            annotation_y,
            f"logsum_avg: {logsum_avg:.2f}; snpweight: {snpweight:.2f}",
            fontsize=8,
            color="blue",
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.3'),
        )

        # Plot WT/Mut on the top axis
        ax1.bar(
            np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
            y_wt_curr,
            width=1.0,
            color="green",
            alpha=0.5,
            label="Ref",
        )

        if plot_mut:
            ax1.bar(
                np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                y_mut_curr,
                width=1.0,
                color="red",
                alpha=0.5,
                label="Alt",
            )

        # Plot annotations (vertical lines)
        for pas_ix, anno_pos in enumerate(anno_poses):
            pas_bin = int((anno_pos - start) // 32) - 16
            ax1.axvline(
                x=pas_bin,
                color="cyan",
                linewidth=2,
                alpha=0.5,
                linestyle="-",
                zorder=-1,
            )

        # Mark mutation
        ax1.scatter(
            [mut_bin],
            [0.075 * max_y],
            color="black",
            s=125,
            marker="*",
            zorder=100,
            label="SNP",
        )

        # Vertical reference lines
        ax1.vlines(
            center_bin,
            0,
            max_y,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Center",
        )
        ax1.vlines(
            gene_start_bin,
            0,
            max_y,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"{search_gene} Gene Start",
        )
        ax1.vlines(
            gene_end_bin,
            0,
            max_y,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"{search_gene} Gene End",
        )

        ax1.set_xlim(plot_start_bin, plot_end_bin - 1)
        # Show y-axis ticks and label
        ax1.set_ylabel("Signal", fontsize=8)
        ax1.set_title("Track(s): " + str(track_name), fontsize=8)
        ax1.legend(fontsize=8)

        # If ground truth provided, plot on second (bottom) axis
        if y_ground_truth is not None:
            ax2.bar(
                np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                y_gt_curr,
                width=1.0,
                color="blue",
                alpha=0.5,
                label="Ground Truth",
            )

            # Optionally add vertical lines or SNP markers if desired
            # ax2.axvline(...), etc.

            ax2.set_xlim(plot_start_bin, plot_end_bin - 1)
            ax2.set_ylabel("GT Signal", fontsize=8)
            ax2.set_xlabel(
                chrom
                + ":"
                + str(plot_start)
                + "-"
                + str(plot_end)
                + " ("
                + str(plot_window)
                + "bp window)",
                fontsize=8,
            )
            ax2.legend(fontsize=8)

        else:
            # If no ground truth, just put labels on ax1
            ax1.set_xlabel(
                chrom
                + ":"
                + str(plot_start)
                + "-"
                + str(plot_end)
                + " ("
                + str(plot_window)
                + "bp window)",
                fontsize=8,
            )

        plt.tight_layout()

        if save_figs:
            plt.savefig(
                save_dir
                + "yeast_"
                + save_suffix
                + "_track_"
                + str(track_index[0])
                + "_to_"
                + str(track_index[-1])
                + ".png",
                dpi=300,
                transparent=False,
            )
            plt.savefig(
                save_dir
                + "yeast_"
                + save_suffix
                + "_track_"
                + str(track_index[0])
                + "_to_"
                + str(track_index[-1])
                + ".eps"
            )
        # plt.show()


def plot_coverage_track_pair_bins(
    y_wt,
    y_mut,
    chrom,
    start,
    search_gene,
    center_pos,
    gene_start,
    gene_end,
    poses,
    track_indices,
    track_names,
    track_scales,
    track_transforms,
    clip_softs,
    log_scale=False,
    sqrt_scale=False,
    plot_mut=True,
    plot_window=4096,
    normalize_window=4096,
    bin_size=32,
    pad=16,
    rescale_tracks=True,
    normalize_counts=False,
    save_figs=False,
    save_suffix="default",
    save_dir="./",
    gene_slice=None,
    anno_df=None,
):

    print("Plotting coverage track pair bins...")
    print("\tplot_window: " + str(plot_window))
    print("\tnormalize_window: " + str(normalize_window))
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

    # Get annotation positions
    anno_poses = []
    if anno_df is not None:
        anno_poses = anno_df.query(
            "chrom == '"
            + chrom
            + "' and position_hg38 >= "
            + str(plot_start)
            + " and position_hg38 < "
            + str(plot_end)
        )["position_hg38"].values.tolist()

    # Plot each tracks
    for track_name, track_index, track_scale, track_transform, clip_soft in zip(
        track_names, track_indices, track_scales, track_transforms, clip_softs
    ):
        print(" - track_name = " + track_name)
        print(" - track_index = " + str(track_index))
        print(" - track_scale = " + str(track_scale))
        print(" - track_transform = " + str(track_transform))
        print(" - clip_soft = " + str(clip_soft))

        # Plot track densities (bins)
        y_wt_curr = np.array(np.copy(y_wt), dtype=np.float32)
        y_mut_curr = np.array(np.copy(y_mut), dtype=np.float32)

        if rescale_tracks:
            # undo scale
            y_wt_curr /= track_scale
            y_mut_curr /= track_scale

            # undo soft_clip
            if clip_soft is not None:
                y_wt_curr_unclipped = (y_wt_curr - clip_soft) ** 2 + clip_soft
                y_mut_curr_unclipped = (y_mut_curr - clip_soft) ** 2 + clip_soft

                unclip_mask_wt = y_wt_curr > clip_soft
                unclip_mask_mut = y_mut_curr > clip_soft

                y_wt_curr[unclip_mask_wt] = y_wt_curr_unclipped[unclip_mask_wt]
                y_mut_curr[unclip_mask_mut] = y_mut_curr_unclipped[unclip_mask_mut]

            # undo sqrt
            y_wt_curr = y_wt_curr ** (1.0 / track_transform)
            y_mut_curr = y_mut_curr ** (1.0 / track_transform)

        y_wt_curr = np.mean(y_wt_curr[..., track_index], axis=(0, 1, 3))
        y_mut_curr = np.mean(y_mut_curr[..., track_index], axis=(0, 1, 3))

        if normalize_counts:
            wt_count = np.sum(y_wt_curr[normalize_start_bin:normalize_end_bin])
            mut_count = np.sum(y_mut_curr[normalize_start_bin:normalize_end_bin])

            # Normalize to densities
            y_wt_curr /= wt_count
            y_mut_curr /= mut_count

            # Bring back to count space (wt reference)
            y_wt_curr *= wt_count
            y_mut_curr *= wt_count

        if gene_slice is not None:
            sum_wt = np.sum(y_wt_curr[gene_slice])
            sum_mut = np.sum(y_mut_curr[gene_slice])

            print(" - sum_wt = " + str(round(sum_wt, 4)))
            print(" - sum_mut = " + str(round(sum_mut, 4)))

        y_wt_curr = y_wt_curr[plot_start_bin:plot_end_bin]
        y_mut_curr = y_mut_curr[plot_start_bin:plot_end_bin]

        if log_scale:
            y_wt_curr = np.log2(y_wt_curr + 1.0)
            y_mut_curr = np.log2(y_mut_curr + 1.0)
        elif sqrt_scale:
            y_wt_curr = np.sqrt(y_wt_curr + 1.0)
            y_mut_curr = np.sqrt(y_mut_curr + 1.0)

        max_y_wt = np.max(y_wt_curr)
        max_y_mut = np.max(y_mut_curr)

        if plot_mut:
            max_y = max(max_y_wt, max_y_mut)
        else:
            max_y = max_y_wt

        print(" - max_y_wt = " + str(round(max_y_wt, 4)))
        print(" - max_y_mut = " + str(round(max_y_mut, 4)))
        print(" -- (max_y = " + str(round(max_y, 4)) + ")")

        f = plt.figure(figsize=(12, 2))

        print(" - track_indices = " + str(track_index))
        print(" - plot_start_bin = " + str(plot_start_bin))
        print(" - plot_end_bin = " + str(plot_end_bin))
        print(" - center_bin = " + str(center_bin))
        print(" - gene_start_bin = " + str(gene_start_bin)) 
        print(" - gene_end_bin = " + str(gene_end_bin))
        print(" - mut_bin = " + str(mut_bin))
        print(" - y_wt_curr: " + str(y_wt_curr.shape))


        plt.bar(
            np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
            y_wt_curr,
            width=1.0,
            color="green",
            alpha=0.5,
            label="Ref",
        )

        if plot_mut:
            plt.bar(
                np.arange(plot_end_bin - plot_start_bin) + plot_start_bin,
                y_mut_curr,
                width=1.0,
                color="red",
                alpha=0.5,
                label="Alt",
            )

        xtick_vals = []

        for pas_ix, anno_pos in enumerate(anno_poses):

            pas_bin = int((anno_pos - start) // 32) - 16

            xtick_vals.append(pas_bin)

            bin_end = pas_bin + 3 - 0.5
            bin_start = bin_end - 5

            plt.axvline(
                x=pas_bin,
                color="cyan",
                linewidth=2,
                alpha=0.5,
                linestyle="-",
                zorder=-1,
            )

        plt.scatter(
            [mut_bin],
            [0.075 * max_y],
            color="black",
            s=125,
            marker="*",
            zorder=100,
            label="SNP",
        )

        plt.vlines(
            center_bin,
            0,
            max_y,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Center",
        )
        plt.vlines(
            gene_start_bin,
            0,
            max_y,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"{search_gene} Gene Start",
        )
        plt.vlines(
            gene_end_bin,
            0,
            max_y,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"{search_gene} Gene End",
        )

        plt.xlim(plot_start_bin, plot_end_bin - 1)

        plt.xticks([], [])
        plt.yticks([], [])

        plt.xlabel(
            chrom
            + ":"
            + str(plot_start)
            + "-"
            + str(plot_end)
            + " ("
            + str(plot_window)
            + "bp window)",
            fontsize=8,
        )
        plt.ylabel("Signal (log)" if not rescale_tracks else "Signal", fontsize=8)

        plt.title("Track(s): " + str(track_name), fontsize=8)

        plt.legend(fontsize=8)

        plt.tight_layout()

        if save_figs:
            plt.savefig(
                save_dir
                + "yeast_"
                + save_suffix
                + "_track_"
                + str(track_index[0])
                + "_to_"
                + str(track_index[-1])
                + ".png",
                dpi=300,
                transparent=False,
            )
            plt.savefig(
                save_dir
                + "yeast_"
                + save_suffix
                + "_track_"
                + str(track_index[0])
                + "_to_"
                + str(track_index[-1])
                + ".eps"
            )

        # plt.show()


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
