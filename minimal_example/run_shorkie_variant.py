#!/usr/bin/env python3
"""
run_shorkie_variant.py — Minimal Shorkie variant effect prediction (logSED)

Usage:
  python run_shorkie_variant.py --model_dir /path/to/model_dir
  python run_shorkie_variant.py --model_dir /path/to/model_dir --chrom chrXI --pos 128987 --ref A --alt G --gene YKL152C

The model-loading, sequence-building and logSED helpers now live in the
installable package (``shorkie.models.ensemble``); this script is a thin CLI
showcase around them. Install the package with ``pip install -e .`` from the
repo root, or rely on the src/ fallback below.
"""
import os, sys, json, argparse, pathlib
import numpy as np
import pysam

# Use the shared package implementation; fall back to repo src/ if not installed.
try:
    from shorkie.models.ensemble import load_ensemble, make_input, predict, logSED
except ImportError:
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from shorkie.models.ensemble import load_ensemble, make_input, predict, logSED

from baskerville import gene as bgene

# ── Default resource paths ──────────────────────────────────────────────────
# params.json / sheet.txt ship next to this script, so they resolve with no setup.
# The genome FASTA/GTF are not committed (too large) — they resolve through
# `shorkie.config` (keys genome.fasta / genome.gtf), which is what
# `data/download.sh --genome` populates. Every one of these is overridable by flag.
_HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_PARAMS  = str(_HERE / "params.json")
DEFAULT_TARGETS = str(_HERE / "sheet.txt")


def _config_path(key):
    """Resolve a genome path via shorkie.config; None if unset/unavailable."""
    try:
        from shorkie import config
        val = config.path(key)
    except Exception:
        return None
    return str(val) if val else None


DEFAULT_GTF   = _config_path("genome.gtf")
DEFAULT_FASTA = _config_path("genome.fasta")
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
    # ── Resources ──
    # params/targets ship beside this script; genome files come from shorkie.config
    # (populated by `data/download.sh --genome`).
    p.add_argument("--params_file",  default=DEFAULT_PARAMS,
                   help="model architecture params.json (default: alongside this script)")
    p.add_argument("--targets_file", default=DEFAULT_TARGETS,
                   help="tab-separated track metadata (default: alongside this script)")
    p.add_argument("--gtf_file",     default=DEFAULT_GTF,
                   help="yeast GTF (default: config key genome.gtf)")
    p.add_argument("--fasta_file",   default=DEFAULT_FASTA,
                   help="indexed yeast FASTA (default: config key genome.fasta)")
    p.add_argument("--seq_len",      type=int, default=16384)
    return p.parse_args()


def _require(path, flag, config_key=None):
    """Fail early and helpfully when a required resource can't be resolved."""
    if not path:
        hint = (f"Set config key '{config_key}' in config/paths.yaml, run "
                f"`data/download.sh --genome`, or pass {flag} explicitly."
                if config_key else f"Pass {flag} explicitly.")
        sys.exit(f"error: no path for {flag}. {hint}")
    if not os.path.exists(path):
        sys.exit(f"error: {flag} not found: {path}\n"
                 f"       Run `data/download.sh --genome` to fetch the reference files.")
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import pandas as pd

    # 0. Resolve + validate resources up front, so failures are actionable.
    _require(args.params_file,  "--params_file")
    _require(args.targets_file, "--targets_file")
    _require(args.gtf_file,     "--gtf_file",   "genome.gtf")
    _require(args.fasta_file,   "--fasta_file", "genome.fasta")

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
        import tensorflow as tf
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
