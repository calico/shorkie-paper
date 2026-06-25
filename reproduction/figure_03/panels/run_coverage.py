#!/usr/bin/env python3
"""GPU coverage prediction for Figure 3 panels H/I/J — RNA-seq coverage at three
ribosomal-protein loci, Shorkie (self_supervised, fine-tuned) vs Random_Init
(supervised LR=5e-4, scratch-trained) vs observed bigwig.

Faithful to scripts/03_eval/supervised/track_prediction_eval/3_viz_rnaseq_tracks/
2_yeast_rna_seq_models.py: the same two model trees, the same RNA-seq target slice,
process_sequence(model_type=...) input encoding, transform=None (no untransform on
the no_norm tree), and observed coverage from the released bigwigs. As in the
published panels (titled "fold 3" / "fold 6", legend "(Avg)") each locus is scored
with the SINGLE held-out fold whose test partition contains it, averaged over the
RNA-seq tracks.

IMPORTANT — one model per process. Loading a 170-feature (self_supervised) SeqNN and
a 4-feature (supervised) SeqNN in the same Python process corrupts the second model's
weight restore (Keras cross-architecture layer-name collision → the second model
silently keeps init weights and predicts the softplus floor ln2≈0.693 everywhere).
So this script loads exactly ONE model per invocation (--tree T --fold F), writes a
per-(locus,tree) partial, and --merge stitches them + adds observed coverage.

Run via run_coverage.sbatch (5 sequential invocations: random_init f3/f6, shorkie
f3/f6, merge). Outputs per locus to reproduced/coverage/<name>.npz:
  cov_self / cov_ri : per-bin mean predicted coverage (held-out fold, mean over RNA-seq tracks)
  cov_obs           : per-bin mean observed coverage (mean over an RNA-seq subset)
  fold, seq_out_start, stride, win, chrom  for genomic alignment.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import pysam
import pyranges as pr
import tensorflow as tf
from baskerville import seqnn

from shorkie import config
from shorkie.helpers.yeast_helpers import process_sequence, predict_tracks
from shorkie.viz.load_cov import read_coverage, seq_norm

SEQLEN = 16384
OBS_N = 40  # observed bigwig subsample (RNA-seq tracks) — mean coverage reference

# Each locus scored with the held-out fold whose test set contains it (published panel titles).
LOCI = [
    dict(name="rpl7a",          chrom="chrVII", win=(362180, 366023), fold=3),  # panel H
    dict(name="rps16b_rpl13a",  chrom="chrIV",  win=(305657, 310505), fold=3),  # panel I
    dict(name="efm5",           chrom="chrVII", win=(495374, 499965), fold=6),  # panel J
]

TREES = {
    "shorkie":     dict(sub="self_supervised_unet_small_bert_drop",                              nfeat=170, mtype="self_supervised"),
    "random_init": dict(sub="supervised_unet_small_bert_drop_variants/learning_rate_0.0005",     nfeat=4,   mtype="supervised"),
}


def parse_group(d):
    if "Chip-exo" in d or "_pos_logFE" in d: return "Chip-exo"
    if "Chip-MNase" in d: return "Chip-MNase"
    if "1000 strains RNAseq" in d: return "1000-RNA-seq"
    if "RNAseq" in d: return "RNA-seq"
    return "Other"


def paths():
    config.load()
    WORK = str(config.path("work_root"))
    base = f"{WORK}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
    out_dir = config.repo_root() / "reproduction" / "figure_03" / "reproduced" / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    return WORK, base, out_dir


def rna_targets(base):
    targets = pd.read_csv(f"{base}/cleaned_sheet.txt", sep="\t", index_col=0)
    targets["group"] = targets["description"].apply(parse_group)
    return targets[targets["group"] == "RNA-seq"]


def load_one(model_dir, nfeat, fold, target_index):
    pf = f"{model_dir}/train/f{fold}c0/train/params.json"
    mf = f"{model_dir}/train/f{fold}c0/train/model_best.h5"
    params = json.load(open(pf))["model"]
    params["num_features"] = nfeat
    params["seq_length"] = SEQLEN
    m = seqnn.SeqNN(params)
    m.restore(mf, trunk=False)
    m.build_slice(target_index)
    m.build_ensemble(True, [0])
    print(f"  loaded {model_dir.split('/')[-1]} fold {fold} (nfeat={nfeat})", flush=True)
    return m


def predict_tree_fold(tree, fold):
    WORK, base, out_dir = paths()
    spec = TREES[tree]
    model_dir = f"{base}/{spec['sub']}"
    rna = rna_targets(base)
    target_index = rna.index
    fasta = pysam.Fastafile(str(config.path("genome.fasta")))
    gene_pr = pr.read_gtf(str(config.path("genome.gtf")))
    gene_pr = gene_pr[gene_pr.Feature.isin(["gene", "exon", "five_prime_UTR", "three_prime_UTR"])]

    loci = [L for L in LOCI if L["fold"] == fold]
    if not loci:
        print(f"[{tree} f{fold}] no loci for this fold", flush=True); return
    m = load_one(model_dir, spec["nfeat"], fold, target_index)
    stride = m.model_strides[0]
    off = stride * m.target_crops[0]

    for L in loci:
        center = (L["win"][0] + L["win"][1]) // 2
        start, end = center - SEQLEN // 2, center + SEQLEN // 2
        x, _ = process_sequence(fasta, L["chrom"], start, end, gene_pr, model_type=spec["mtype"])
        y = predict_tracks([m], x).astype("float64")          # (1,1,bins,tracks)
        y = np.where(np.isfinite(y), y, np.nan)               # drop any overflow bins from the track-avg
        cov = np.nanmean(y, axis=(0, 1, 3)).astype("float32")  # (bins,)
        seq_out_start = int(start + off)
        np.savez(os.path.join(out_dir, f"_part_{L['name']}_{tree}.npz"),
                 cov=cov, seq_out_start=seq_out_start, stride=int(stride),
                 win=np.array(L["win"]), chrom=L["chrom"], fold=int(fold))
        print(f"[{tree} f{fold}] {L['name']} {L['chrom']}:{L['win'][0]}-{L['win'][1]} "
              f"bins={len(cov)} mean={np.nanmean(cov):.3f} std={np.nanstd(cov):.3f} max={np.nanmax(cov):.3f}",
              flush=True)
    fasta.close()


def merge():
    WORK, base, out_dir = paths()
    rna = rna_targets(base)
    step = max(1, len(rna) // OBS_N)
    obs_rows = rna.iloc[::step].head(OBS_N)

    for L in LOCI:
        ps = os.path.join(out_dir, f"_part_{L['name']}_shorkie.npz")
        pr_ = os.path.join(out_dir, f"_part_{L['name']}_random_init.npz")
        if not (os.path.exists(ps) and os.path.exists(pr_)):
            print(f"[merge] {L['name']}: missing partial(s); skip", file=sys.stderr, flush=True); continue
        ds, dr = np.load(ps, allow_pickle=True), np.load(pr_, allow_pickle=True)
        cov_self, cov_ri = ds["cov"], dr["cov"]
        seq_out_start, stride = int(ds["seq_out_start"]), int(ds["stride"])
        nb = len(cov_self)

        # observed: mean over subsampled RNA-seq bigwigs at the model output region
        region_start, region_end = seq_out_start, seq_out_start + nb * stride
        obs = []
        for _, row in obs_rows.iterrows():
            cv = read_coverage(row["file"], L["chrom"], region_start, region_end)
            obs.append(seq_norm(cv))                            # already binned to model stride (896 bins)
        obs = np.array(obs)
        if obs.shape[1] != nb:
            obs = obs[:, :nb]
        cov_obs = obs.mean(axis=0).astype("float32")

        np.savez(os.path.join(out_dir, f"{L['name']}.npz"),
                 cov_self=cov_self, cov_ri=cov_ri, cov_obs=cov_obs,
                 seq_out_start=seq_out_start, stride=stride,
                 win=np.array(L["win"]), chrom=L["chrom"],
                 fold=int(ds["fold"]), n_obs=len(obs_rows))
        rs = np.corrcoef(cov_self, cov_obs)[0, 1]
        rr = np.corrcoef(cov_ri, cov_obs)[0, 1]
        print(f"[merge] {L['name']} fold{int(ds['fold'])}: self_mean={cov_self.mean():.3f} "
              f"ri_mean={cov_ri.mean():.3f} obs_mean={cov_obs.mean():.3f} "
              f"R(self,obs)={rs:.3f} R(ri,obs)={rr:.3f}", flush=True)
    print("DONE_MERGE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", choices=list(TREES))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    elif args.tree is not None and args.fold is not None:
        predict_tree_fold(args.tree, args.fold)
    else:
        ap.error("need --tree T --fold F, or --merge")


if __name__ == "__main__":
    main()
