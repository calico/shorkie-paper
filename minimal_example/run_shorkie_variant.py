#!/usr/bin/env python3
"""
run_shorkie_variant.py — Minimal Shorkie variant effect prediction (logSED)

Usage:
  python run_shorkie_variant.py --model_dir /path/to/model_dir
  python run_shorkie_variant.py --model_dir /path/to/model_dir --chrom chrXI --pos 128987 --ref A --alt G --gene YKL152C

All genomic resource paths default to the Salzberg cluster paths.
"""
import os, sys, json, argparse
import numpy as np
import pysam
import tensorflow as tf
from baskerville import seqnn, dna
from baskerville import gene as bgene

# ── Default paths (Salzberg cluster) ────────────────────────────────────────
_ROOT = ""
_EXP  = ""
_DATA = ""

DEFAULT_PARAMS   = f"{_EXP}/params.json"
DEFAULT_TARGETS  = f"{_EXP}/sheet.txt"
DEFAULT_GTF      = f"{_DATA}/gtf/GCA_000146045_2.59.gtf"
DEFAULT_FASTA    = f"{_DATA}/fasta/GCA_000146045_2.cleaned.fasta"
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Minimal Shorkie logSED scoring")
    # ── Model ──
    p.add_argument("--model_dir",   required=True,
                   help="Directory containing fold sub-dirs, e.g. .../self_supervised_unet_small_bert_drop")
    p.add_argument("--num_folds",   type=int, default=8)
    # ── Variant ──
    p.add_argument("--chrom", default="chrI")
    p.add_argument("--pos",   type=int, default=124373, help="1-based position")
    p.add_argument("--ref",   default="T")
    p.add_argument("--alt",   default="C")
    p.add_argument("--gene",  default="YAL016C-B")
    # ── Resources (defaults to cluster paths) ──
    p.add_argument("--params_file",  default=DEFAULT_PARAMS)
    p.add_argument("--targets_file", default=DEFAULT_TARGETS)
    p.add_argument("--gtf_file",     default=DEFAULT_GTF)
    p.add_argument("--fasta_file",   default=DEFAULT_FASTA)
    p.add_argument("--seq_len",      type=int, default=16384)
    return p.parse_args()


# ── Sequence helpers ─────────────────────────────────────────────────────────

def fetch_1hot(fasta, chrom, start, end, seq_len):
    """Fetch DNA window, one-hot encode, pad with N if needed."""
    seq = ("N" * -start + fasta.fetch(chrom, 0, end)) if start < 0 \
          else fasta.fetch(chrom, start, end)
    seq = seq.ljust(seq_len, "N")
    return dna.dna_1hot(seq).astype("float32")


def make_input(fasta, chrom, start, end, seq_len=16384):
    """
    Build model input tensor (seq_len, 170):
      channels 0-3:    DNA one-hot (A/C/G/T)
      channels 4-169:  species identity (zeros = S. cerevisiae, except col 114 = 1)
    """
    seq_len_actual = end - start
    pad = (seq_len - seq_len_actual) // 2
    x = fetch_1hot(fasta, chrom, start - pad, end + pad, seq_len)
    x = tf.Variable(tf.concat([x, tf.zeros((seq_len, 166), dtype=tf.float32)], axis=-1))
    x[:, 114].assign(tf.ones(seq_len))          # S. cerevisiae species channel
    return x


# ── Model loading ────────────────────────────────────────────────────────────

def load_ensemble(model_dir, params_file, target_index, num_folds):
    with open(params_file) as f:
        params = json.load(f)
    params["model"]["num_features"] = 165 + 5   # species + DNA channels

    models = []
    for fold in range(num_folds):
        path = os.path.join(model_dir, "train", f"f{fold}c0", "train", "model_best.h5")
        print(f"  Loading fold {fold}: {path}")
        m = seqnn.SeqNN(params["model"])
        m.restore(path, trunk=False, by_name=False)
        m.build_slice(target_index)
        m.build_ensemble(True, [0])
        models.append(m)
    return models


def predict(models, x):
    """Average predictions across all fold models."""
    preds = [m(x[None, ...])[: , None, ...].astype("float32") for m in models]
    return np.mean(preds, axis=0)   # shape: (1, 1, seq_bins, num_tracks)


# ── logSED ───────────────────────────────────────────────────────────────────

def logSED(y_ref, y_alt, gene_slice):
    """
    logSED (aggregated) = log2(Σ_alt + 1) - log2(Σ_ref + 1)
    summed over output bins overlapping the gene body, averaged across tracks.
    """
    cov_ref = np.mean(y_ref, axis=(0, 1, 3))   # (seq_bins,)
    cov_alt = np.mean(y_alt, axis=(0, 1, 3))
    s_ref = cov_ref[gene_slice].sum()
    s_alt = cov_alt[gene_slice].sum()
    return float(np.log2(s_alt + 1) - np.log2(s_ref + 1))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import pandas as pd

    # 1. Targets
    targets_df   = pd.read_csv(args.targets_file, index_col=0, sep="\t")
    target_index = targets_df.index
    print(f"Tracks: {len(target_index)}")

    # 2. Load ensemble
    print(f"Loading {args.num_folds}-fold ensemble from {args.model_dir}...")
    models = load_ensemble(args.model_dir, args.params_file, target_index, args.num_folds)
    m0 = models[0]

    # 3. Genomic resources
    fasta         = pysam.Fastafile(args.fasta_file)
    transcriptome = bgene.Transcriptome(args.gtf_file)

    # 4. Look up gene
    keys = [k for k in transcriptome.genes if args.gene in k]
    assert keys, f"Gene '{args.gene}' not found in GTF"
    gene = transcriptome.genes[keys[0]]
    gene_start, gene_end = gene.span()
    print(f"Gene {args.gene}: {gene.chrom}:{gene_start}-{gene_end}")

    # 5. Compute window placement
    gc    = gene.midpoint()
    off   = m0.model_strides[0] * m0.target_crops[0]
    olen  = m0.model_strides[0] * m0.target_lengths[0]
    pos   = args.pos

    lo = max(pos - args.seq_len + 1, gc - off - olen + 1)
    hi = min(pos - 1,                gc - off)
    start = int((lo + hi) // 2) if lo <= hi else int(gc - args.seq_len // 2)
    end   = start + args.seq_len

    gene_slice = gene.output_slice(start + off, int(olen), m0.model_strides[0], False)

    # 6. Build ref & alt sequences
    chrom = args.chrom if args.chrom.startswith("chr") else "chr" + args.chrom
    x_ref = make_input(fasta, chrom, start, end, args.seq_len)

    # Verify reference allele
    ci  = pos - start - 1
    if 0 <= ci < args.seq_len:
        nt  = {0: "A", 1: "C", 2: "G", 3: "T"}
        ext = nt.get(int(np.argmax(x_ref.numpy()[ci, :4])), "N")
        if ext != args.ref.upper():
            print(f"WARNING: genome ref = {ext}, supplied ref = {args.ref}", file=sys.stderr)

        # Apply mutation
        alt_ix = {"A": 0, "C": 1, "G": 2, "T": 3}[args.alt.upper()]
        x_alt  = np.copy(x_ref.numpy())
        x_alt[ci, :4] = 0.; x_alt[ci, alt_ix] = 1.
        x_alt  = tf.constant(x_alt)
    else:
        print(f"WARNING: Variant {pos} is outside the 16kb window [{start}, {end}) for gene {args.gene}.", file=sys.stderr)
        x_alt = x_ref

    # 7. Predict & score
    print("Predicting reference...")
    y_ref = predict(models, x_ref)
    print("Predicting alternate...")
    y_alt = predict(models, x_alt)

    score = logSED(y_ref, y_alt, gene_slice)

    # 8. Report
    print(f"\n{'='*50}")
    print(f"  Variant  : {chrom}:{pos} {args.ref}>{args.alt}")
    print(f"  Gene     : {args.gene}")
    print(f"  logSED   : {score:+.4f}")
    print(f"{'='*50}")
    print("  logSED > 0 → alt increases predicted expression")
    print("  logSED < 0 → alt decreases predicted expression")

    fasta.close()


if __name__ == "__main__":
    main()
