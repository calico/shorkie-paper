#!/usr/bin/env python3
"""Revision experiment 07, step 1 — recompute promoter ISM restricting the bin sum to the
target gene.

The question: when mutating a promoter, is the intended quantity the effect on THAT gene's
expression? If so the ISM should sum only the bins covering the target gene, because
otherwise predicted coverage from neighbouring genes contributes to the score -- and could
affect the comparison across induction time points as well as a single saliency map.

The mechanics are these. Equation 17 sums over B = all 896 output bins
(~14.3 kb of genomic coverage), so a promoter mutation is scored against everything in the
window, neighbours included. The paper's *variant* scoring does not do this -- ``hound_snp``
and the gene-level evaluation both restrict to gene-body bins (Methods L1175) -- so the two
conventions genuinely differ.

The time-course comparison matters as much as the static one, and is easy to overlook:
neighbouring-gene coverage could affect Figure 5C, not just a single saliency map. So this
script does NOT average over a single T0 track subset. It computes logSED per bin scope AND
per induction time point, using the same track partition the published Figure 5 uses
(``reproduction/figure_05/recheck/fig05_lib.tp_tracks``), which lets step 2 rebuild the 8x8
Euclidean-distance matrix under each scope and compare them directly.

Three bin scopes over the same mutations, in the same forward passes:

  all_bins    every output bin                      (the published Equation 17)
  gene_body   bins overlapping the target gene      (the gene-restricted alternative)
  tss_window  bins within +/-1 kb of the TSS        (a middle ground: local, but not
                                                     dependent on annotation extent)

Window placement, input construction and the logSED definition are the published ones,
ported from ``reproduction/figure_07/panels/run_ism_eqtl.py``.

Writes ``results/ism/<name>.npz`` per locus, each holding
``grid_<scope>`` of shape (n_positions, 4, n_timepoints).

GPU strongly recommended (roughly 4 x scan_length forward passes per locus through an
8-fold ensemble); CPU works but is slow.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from baskerville import gene as bgene

from shorkie import config
from shorkie.models.ensemble import load_ensemble, make_input, ensemble_predict

NT = ["A", "C", "G", "T"]
SEQ_LEN = 16384
TSS_FLANK_BP = 1000          # half-width of the tss_window scope

# Default loci. `tf` names the induced transcription factor whose time course the published
# figure shows; loci without one are scored at T0 only (a single "timepoint").
DEFAULT_LOCI = [
    dict(name="atg42", gene="YBR139W", up=450, down=50, tf="MSN2"),   # Figure 5A
    dict(name="tsl1", gene="YML100W", up=450, down=50, tf="MSN4"),    # Figure 5F
    dict(name="fun12", gene="YAL035W", up=450, down=50, tf=None),     # Figure 4B, RRB
    dict(name="kre33", gene="YNL132W", up=450, down=50, tf=None),     # Figure 4C, RRB
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Promoter ISM under three output-bin scopes, per induction time point.")
    parser.add_argument("--genes", default=None,
                        help="Comma-separated systematic ORF names; default is the "
                             "Figure 4/5 promoters this experiment is about")
    parser.add_argument("--num_folds", type=int, default=None,
                        help="Folds to ensemble [default: config models.num_folds]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--targets_file", default=None,
                        help="RNA-seq targets sheet [default: the 3,053-track "
                             "cleaned_sheet_RNA-Seq.txt the published Figure 5 uses]")
    return parser.parse_args()

def slice_to_idx(gs):
    """baskerville returns either a slice or an index array; normalise to an index array."""
    if isinstance(gs, slice):
        return np.arange(gs.start or 0, gs.stop)
    return np.asarray(gs, dtype=int)

def log_sed(y_ref, y_alt, bin_idx, track_idx):
    """logSED over a bin set and a track set: log2(sum_alt + 1) - log2(sum_ref + 1).

    Note this averages the PER-TRACK log ratios, which differs from
    ``shorkie.models.ensemble.logSED`` (average-then-log). Per-track-then-average is
    deliberate: it is the convention ``fig05_lib.load_locus`` applies to the released
    logSED HDF5, and the whole point of this script is to be comparable to Figure 5.
    """
    r = np.asarray(y_ref)[0, 0][np.ix_(bin_idx, track_idx)].sum(axis=0)
    a = np.asarray(y_alt)[0, 0][np.ix_(bin_idx, track_idx)].sum(axis=0)
    return float(np.mean(np.log2(a + 1) - np.log2(r + 1)))

def timepoint_tracks(tf, fig05):
    """{timepoint: track indices} for a TF, or {0: all tracks} when there is no time course."""
    if tf is None or fig05 is None:
        return None
    tps = fig05.tp_tracks(tf)
    return {t: np.asarray(v, dtype=int) for t, v in tps.items() if len(v)}

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    (out_dir / "ism").mkdir(parents=True, exist_ok=True)

    # The published Figure-5 track partition, reused rather than re-derived.
    fig05 = None
    try:
        sys.path.insert(0, str(Path(config.repo_root()) /
                               "reproduction" / "figure_05" / "recheck"))
        import fig05_lib as fig05
    except Exception as e:                                     # time course unavailable
        print(f"NOTE: fig05_lib unavailable ({e}); scoring T0 only", file=sys.stderr)

    model_dir = config.path("models.shorkie_finetuned")
    params_file = str(model_dir / "params.json")
    targets_file = args.targets_file or str(
        Path(config.path("datasets.supervised_root")) / "cleaned_sheet_RNA-Seq.txt")
    fasta_file = str(config.path("genome.fasta"))
    gtf_file = str(config.path("genome.gtf"))
    n_folds = args.num_folds if args.num_folds else int(config.get("models.num_folds", 8))
    for p, what in ((params_file, "params.json"), (targets_file, "RNA-seq targets sheet"),
                    (fasta_file, "genome FASTA"), (gtf_file, "genome GTF")):
        if not Path(p).exists():
            sys.exit(f"error: {what} not found at {p}")

    loci = DEFAULT_LOCI
    if args.genes:
        loci = [dict(name=g.lower(), gene=g, up=450, down=50, tf=None)
                for g in args.genes.split(",")]

    print(f"model_dir : {model_dir}\nfolds     : {n_folds}\ntargets   : {targets_file}",
          flush=True)
    # index_col=0 is load-bearing. build_slice() gathers POSITIONALLY into the model's
    # 5,215-track output, so target_index must hold the sheet's `index` column (1148..4200
    # for the RNA-seq subset), not a RangeIndex. With a RangeIndex this silently scores
    # tracks 0..3052 = 1,128 ChIP-exo + 20 ChIP-MNase + only 1,905 RNA-seq, with the right
    # shape and no error. Matches reproduction/figure_07/panels/run_ism_eqtl.py:71-72.
    target_index = pd.read_csv(targets_file, index_col=0, sep="\t").index.to_numpy()
    models = load_ensemble(str(model_dir), params_file, target_index, num_folds=n_folds)
    m0 = models[0]
    off = m0.model_strides[0] * m0.target_crops[0]
    olen = m0.model_strides[0] * m0.target_lengths[0]
    stride = m0.model_strides[0]

    fasta = pysam.Fastafile(fasta_file)
    transcriptome = bgene.Transcriptome(gtf_file)

    for L in loci:
        try:
            keys = [k for k in transcriptome.genes if L["gene"] in k]
            if not keys:
                print(f"SKIPPED: {L['gene']} not in the GTF", file=sys.stderr)
                continue
            gene = transcriptome.genes[keys[0]]
            chrom = gene.chrom if gene.chrom.startswith("chr") else "chr" + gene.chrom
            gc = gene.midpoint()
            start = int(gc - SEQ_LEN // 2)
            end = start + SEQ_LEN
            seq_out_start = int(start + off)

            # --- the three bin scopes -------------------------------------------------
            gene_idx = slice_to_idx(gene.output_slice(seq_out_start, int(olen), stride, False))
            n_bins = int(m0.target_lengths[0])
            all_idx = np.arange(n_bins)
            strand = getattr(gene, "strand", "+")
            # baskerville's Gene has no.start/.end (only chrom/strand/exons/kv plus
            # span()/midpoint()); reading them raises AttributeError, which the per-locus
            # except below would swallow into a silent zero-output run.
            gene_start, gene_end = gene.span()
            tss = gene_start if strand == "+" else gene_end
            lo = (tss - TSS_FLANK_BP - seq_out_start) // stride
            hi = (tss + TSS_FLANK_BP - seq_out_start) // stride
            tss_idx = np.arange(max(0, lo), min(n_bins, hi + 1))
            if gene_idx.size == 0 or tss_idx.size == 0:
                print(f"SKIPPED: {L['name']} has an empty bin scope "
                      f"(gene {gene_idx.size}, tss {tss_idx.size})", file=sys.stderr)
                continue
            scopes = {"all_bins": all_idx, "gene_body": gene_idx, "tss_window": tss_idx}

            # --- track partition: per induction timepoint, or all tracks at once -------
            tp = timepoint_tracks(L.get("tf"), fig05)
            if tp is None:
                tp = {0: np.arange(len(target_index))}
            timepoints = sorted(tp)
            print(f"{L['name']}: tf={L.get('tf')} timepoints={timepoints}", flush=True)

            # --- promoter scan region (-up / +down around the TSS) ---------------------
            scan_lo = tss - L["up"] if strand == "+" else tss - L["down"]
            scan_hi = tss + L["down"] if strand == "+" else tss + L["up"]
            positions = np.arange(scan_lo, scan_hi)

            x_ref = make_input(fasta, chrom, start, end, SEQ_LEN)
            x_ref_np = x_ref.numpy()
            y_ref = ensemble_predict(models, x_ref)
            cov_ref = np.mean(np.asarray(y_ref)[0, 0], axis=-1)      # (bins,)

            grids = {k: np.full((len(positions), 4, len(timepoints)), np.nan,
                                dtype=np.float32) for k in scopes}
            ref_bases = []
            for i, gpos in enumerate(positions):
                ci = int(gpos - start)
                if not (0 <= ci < SEQ_LEN):
                    ref_bases.append("N")
                    continue
                col = x_ref_np[ci, :4]
                ref_bases.append(NT[int(np.argmax(col))] if col.sum() > 0 else "N")
                for aix, alt in enumerate(NT):
                    if ref_bases[-1] == alt:
                        for k in scopes:
                            grids[k][i, aix, :] = 0.0
                        continue
                    xm = x_ref_np.copy()
                    xm[ci, :4] = 0.0
                    xm[ci, aix] = 1.0
                    y_alt = ensemble_predict(models, xm)
                    for k, bidx in scopes.items():
                        for ti, t in enumerate(timepoints):
                            grids[k][i, aix, ti] = log_sed(y_ref, y_alt, bidx, tp[t])
                if (i + 1) % 50 == 0:
                    print(f"  {L['name']}: {i+1}/{len(positions)} positions", flush=True)

            outside = np.setdiff1d(all_idx, gene_idx)
            contamination = (float(cov_ref[outside].sum() / cov_ref.sum())
                             if cov_ref.sum() > 0 else np.nan)
            np.savez_compressed(
                out_dir / "ism" / f"{L['name']}.npz",
                positions=positions, ref_bases=np.array(ref_bases),
                chrom=chrom, window_start=start, window_end=end, strand=strand, tss=tss,
                gene_id=keys[0], cov_ref=cov_ref, tf=str(L.get("tf")),
                timepoints=np.array(timepoints),
                gene_bin_idx=gene_idx, tss_bin_idx=tss_idx, n_bins=n_bins,
                coverage_fraction_outside_gene=contamination,
                **{f"grid_{k}": v for k, v in grids.items()})
            print(f"{L['name']}: {len(positions)} positions x {len(timepoints)} timepoints, "
                  f"gene bins {gene_idx.size}/{n_bins}, TSS-window bins {tss_idx.size}/{n_bins}, "
                  f"reference coverage outside the gene = {100*contamination:.1f}%",
                  flush=True)
        except Exception as e:                                  # fail-soft per locus
            print(f"SKIPPED: {L['name']} ({e})", file=sys.stderr)

    print(f"\nwrote {out_dir/'ism'}")

if __name__ == "__main__":
    main()
