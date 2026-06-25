"""Shorkie model ensemble loading + variant scoring.

Unifies the f0c0..f7c0 ensemble loading, sequence preparation, and logSED
scoring that was duplicated in ``minimal_example/run_shorkie_variant.py`` and the
eQTL variant scorers (``analysis/shorkie/eqtl/2_variant_scoring/score_variants_*.py``).

Input channel layout: 4 DNA one-hot channels + 166 species-identity channels
(= 170 features); the S. cerevisiae channel is index 114.
"""
from __future__ import annotations

import json
import os

import numpy as np

N_DNA = 4
N_SPECIES = 166
NUM_FEATURES = N_DNA + N_SPECIES        # 170
SCEREVISIAE_COL = 114                   # species channel for S. cerevisiae


def _tf():
    import tensorflow as tf
    return tf


def fetch_1hot(fasta, chrom, start, end, seq_len):
    """Fetch a DNA window, one-hot encode, left-pad with N for negative starts."""
    from baskerville import dna

    seq = ("N" * -start + fasta.fetch(chrom, 0, end)) if start < 0 \
        else fasta.fetch(chrom, start, end)
    seq = seq.ljust(seq_len, "N")
    return dna.dna_1hot(seq).astype("float32")


def make_input(fasta, chrom, start, end, seq_len=16384,
               species_col=SCEREVISIAE_COL, mask_pos=None):
    """Build a ``(seq_len, NUM_FEATURES)`` model input tensor.

    Channels 0-3 are the DNA one-hot; channels 4-169 are species identity (all
    zero except ``species_col`` = 1 for S. cerevisiae). The window is centered on
    ``[start, end)`` and padded symmetrically to ``seq_len``. If ``mask_pos`` (a
    0-based index within the window) is given, the DNA channels at that position
    are zeroed — the LM-style variant masking used by the language-model scorer.
    """
    tf = _tf()
    seq_len_actual = end - start
    pad = (seq_len - seq_len_actual) // 2
    x = fetch_1hot(fasta, chrom, start - pad, end + pad, seq_len)
    x = tf.Variable(tf.concat([x, tf.zeros((seq_len, N_SPECIES), dtype=tf.float32)], axis=-1))
    x[:, species_col].assign(tf.ones(seq_len))
    if mask_pos is not None and 0 <= mask_pos < seq_len:
        x[mask_pos, :N_DNA].assign(tf.zeros(N_DNA))
    return x


def load_ensemble(model_dir, params_file, target_index, num_folds=8,
                  fold_subdir="f{fold}c0"):
    """Load an N-fold Shorkie ensemble.

    Each fold checkpoint is expected at
    ``{model_dir}/train/{fold_subdir}/train/model_best.h5``. Returns a list of
    built ``baskerville.seqnn.SeqNN`` models, sliced to ``target_index`` and
    ensembled over (rc, shift-0).
    """
    from baskerville import seqnn

    with open(params_file) as fh:
        params = json.load(fh)
    params["model"]["num_features"] = NUM_FEATURES

    models = []
    for fold in range(num_folds):
        path = os.path.join(model_dir, "train",
                            fold_subdir.format(fold=fold), "train", "model_best.h5")
        print(f"  Loading fold {fold}: {path}")
        m = seqnn.SeqNN(params["model"])
        m.restore(path, trunk=False, by_name=False)
        m.build_slice(target_index)
        m.build_ensemble(True, [0])
        models.append(m)
    return models


def ensemble_predict(models, x):
    """Average predictions across all fold models. Returns ``(1, 1, bins, tracks)``."""
    preds = [m(x[None, ...])[:, None, ...].astype("float32") for m in models]
    return np.mean(preds, axis=0)


# Backwards-compatible alias (minimal_example used the name ``predict``).
predict = ensemble_predict


def logSED(y_ref, y_alt, gene_slice):
    """Aggregated logSED = log2(Σ_alt + 1) − log2(Σ_ref + 1) over gene-body bins,
    averaged across tracks. ``y_*`` have shape ``(1, 1, bins, tracks)``."""
    cov_ref = np.mean(y_ref, axis=(0, 1, 3))   # (bins,)
    cov_alt = np.mean(y_alt, axis=(0, 1, 3))
    s_ref = cov_ref[gene_slice].sum()
    s_alt = cov_alt[gene_slice].sum()
    return float(np.log2(s_alt + 1) - np.log2(s_ref + 1))


def logSED_per_track(y_ref, y_alt, gene_slice):
    """Per-track logSED (the eQTL-scorer extension); returns an array over tracks."""
    cov_ref = np.mean(y_ref, axis=(0, 1))      # (bins, tracks)
    cov_alt = np.mean(y_alt, axis=(0, 1))
    s_ref = cov_ref[gene_slice].sum(axis=0)
    s_alt = cov_alt[gene_slice].sum(axis=0)
    return np.log2(s_alt + 1) - np.log2(s_ref + 1)
