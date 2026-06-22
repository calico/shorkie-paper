#!/usr/bin/env python3
"""Figure 7 A/B (deep-recheck) — eQTL locus coverage in the published style.

Published A (OMA1, chrXI) / B (LAP3, chrXIV): a top **Signal** track with the model's
Ref (blue) and Alt (orange) predicted coverage filled and overlaid, a bottom **GT Signal**
track with the observed RNA-seq coverage (green, filled), and a **gene-annotation track**
(blue gene boxes + strand arrows + names — OMA1 & neighbor TVP38; NAR1 & LAP3). Markers:
Center (black dashed), gene Start/End (red dashed), eQTL SNP (star).

Predicted Ref/Alt come from the cached ISM run (``reproduced/ism/{oma1,lap3}.npz``,
``cov_ref``/``cov_alt`` = 8-fold ensemble mean over the RNA-seq T0 tracks, 896 output bins).
GT is the mean of a subsample of the released RNA-seq T0 bigwigs, binned with ``seq_norm``
onto the same 896-bin grid. Genes are read from the R64 GTF (bare-Roman seqids).

Outputs: reproduced/Figure_7A_reproduced.png, reproduced/Figure_7B_reproduced.png
         recheck/fig7AB_logsed.csv  (locus, snp, logSED, R_self_obs, R_alt_obs)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from shorkie import config
from shorkie.viz.load_cov import read_coverage, seq_norm

config.load()
REPRO = config.repo_root() / "reproduction" / "figure_07"
RECHECK = REPRO / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)
ROOT = str(config.path("work_root"))

OBS_N = 40                       # observed bigwig subsample for the GT track
SEQ_NT = 14336                   # 896 bins * 16 bp
PAD_BINS = 95                    # display flank around the gene body

LOCI = [("oma1", "A", "OMA1"), ("lap3", "B", "LAP3")]


def rna_t0_files():
    base = f"{ROOT}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
    df = pd.read_csv(f"{base}/cleaned_sheet_RNA-Seq_T0.txt", sep="\t", index_col=0)
    step = max(1, len(df) // OBS_N)
    return df.iloc[::step].head(OBS_N)["file"].tolist()


def observed_track(chrom, seq_out_start):
    files = rna_t0_files()
    obs = []
    for f in files:
        try:
            cv = read_coverage(f, chrom, seq_out_start, seq_out_start + SEQ_NT)
            if len(cv) < SEQ_NT:
                continue
            obs.append(seq_norm(np.asarray(cv, dtype="float32")))
        except Exception:
            continue
    return np.mean(obs, axis=0).astype("float32")


def genes_in_window(gtf_df, chrom_roman, g0, g1):
    sub = gtf_df[(gtf_df["seqid"] == chrom_roman) & (gtf_df["type"] == "gene") &
                 (gtf_df["end"] >= g0) & (gtf_df["start"] <= g1)].copy()
    out = []
    for _, r in sub.iterrows():
        attrs = dict(p.strip().split(" ", 1) for p in r["attributes"].strip(";").split(";") if p.strip())
        nm = attrs.get("gene_name", attrs.get("gene_id", "?")).strip('"')
        out.append((int(r["start"]), int(r["end"]), r["strand"], nm))
    return out


def load_gtf():
    cols = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
    return pd.read_csv(str(config.path("genome.gtf")), sep="\t", comment="#", names=cols, usecols=range(9))


def draw_genes(ax, genes, x0_bp, stride, seq_out_start):
    """Draw IGV-style gene boxes (blue), strand arrows, and names. x-axis = output bin."""
    def to_bin(bp):
        return (bp - seq_out_start) / stride
    for (gs, ge, strand, nm) in genes:
        b0, b1 = to_bin(gs), to_bin(ge)
        y = 0.0
        ax.add_patch(Rectangle((b0, y - 0.18), b1 - b0, 0.36, fc="#1f3fd6", ec="none", alpha=0.9))
        # strand arrowheads along the box
        n_arrows = max(1, int((b1 - b0) / 12))
        for k in range(n_arrows):
            xc = b0 + (k + 0.5) * (b1 - b0) / n_arrows
            dx = 2.2 if strand == "+" else -2.2
            ax.add_patch(Polygon([(xc - dx / 2, y - 0.12), (xc + dx / 2, y), (xc - dx / 2, y + 0.12)],
                                 closed=True, fc="white", ec="none"))
        ax.text((b0 + b1) / 2, y - 0.46, nm, ha="center", va="top", fontsize=9, color="#13207a")


def build_one(name, panel, label, gtf_df):
    z = np.load(REPRO / "reproduced" / "ism" / f"{name}.npz", allow_pickle=True)
    cov_ref = z["cov_ref"]; cov_alt = z["cov_alt"]
    gs = z["gene_slice_idx"]; stride = int(z["stride"]); seq_out_start = int(z["seq_out_start"])
    snp_pos = int(z["snp_pos"]); chrom = str(z["chrom"])
    snp_ref = str(z["snp_ref"]); snp_alt = str(z["snp_alt"]); logsed = float(z["snp_logsed"])
    nbins = len(cov_ref)

    obs = observed_track(chrom, seq_out_start)
    if len(obs) != nbins:
        obs = obs[:nbins]

    b_lo = max(0, int(gs.min()) - PAD_BINS)
    b_hi = min(nbins, int(gs.max()) + PAD_BINS)
    xb = np.arange(b_lo, b_hi)
    snp_bin = (snp_pos - seq_out_start) / stride
    gene_start_bin = (gs.min())
    gene_end_bin = (gs.max())
    center_bin = (gs.min() + gs.max()) / 2

    g0_bp = seq_out_start + b_lo * stride
    g1_bp = seq_out_start + b_hi * stride
    chrom_roman = chrom[3:] if chrom.startswith("chr") else chrom
    genes = genes_in_window(gtf_df, chrom_roman, g0_bp, g1_bp)

    fig, axes = plt.subplots(3, 1, figsize=(11, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.1]})
    # (1) Signal: Ref (blue) + Alt (orange) overlaid filled
    ax = axes[0]
    ax.fill_between(xb, 0, cov_ref[b_lo:b_hi], color="#1f77b4", alpha=0.65, lw=0, label="Ref")
    ax.fill_between(xb, 0, cov_alt[b_lo:b_hi], color="#ff7f0e", alpha=0.55, lw=0,
                    label=f"Alt ({snp_ref}>{snp_alt})")
    ax.set_ylabel("Signal", fontsize=10); ax.set_ylim(bottom=0)
    for xv, c, ls, lab in [(center_bin, "black", ":", "Center"),
                           (gene_start_bin, "red", "--", f"{str(z['gene'])} Start"),
                           (gene_end_bin, "red", "--", f"{str(z['gene'])} End")]:
        ax.axvline(xv, color=c, ls=ls, lw=1.0, alpha=0.8)
    ax.plot([snp_bin], [0], marker="*", ms=15, color="black", clip_on=False, label="SNP")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"{panel}  {label}  {chrom}:{snp_pos} {snp_ref}>{snp_alt}  (logSED={logsed:+.3f})",
                 fontsize=11)
    # (2) GT Signal (green)
    ax = axes[1]
    ax.fill_between(xb, 0, obs[b_lo:b_hi], color="#2ca02c", alpha=0.7, lw=0, label="Ground Truth")
    ax.set_ylabel("GT Signal", fontsize=10); ax.set_ylim(bottom=0)
    ax.axvline(snp_bin, color="black", ls=":", lw=0.8, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    # (3) gene track
    ax = axes[2]
    draw_genes(ax, genes, g0_bp, stride, seq_out_start)
    ax.axvline(snp_bin, color="black", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylim(-1.0, 0.6); ax.set_yticks([])
    ax.set_xlim(b_lo, b_hi)
    ax.set_xlabel("output bin (16 bp)", fontsize=10)

    fig.tight_layout(h_pad=0.4)
    out = REPRO / "reproduced" / f"Figure_7{panel}_reproduced.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}")

    r_ref = float(np.corrcoef(cov_ref[b_lo:b_hi], obs[b_lo:b_hi])[0, 1])
    r_alt = float(np.corrcoef(cov_alt[b_lo:b_hi], obs[b_lo:b_hi])[0, 1])
    print(f"  {label}: logSED={logsed:+.4f}  R(ref,obs)={r_ref:.3f}  R(alt,obs)={r_alt:.3f}  "
          f"genes={[g[3] for g in genes]}")
    return dict(locus=label, panel=panel, snp=f"{chrom}:{snp_pos}{snp_ref}>{snp_alt}",
                logSED=round(logsed, 4), R_ref_obs=round(r_ref, 3), R_alt_obs=round(r_alt, 3),
                genes=";".join(g[3] for g in genes))


def main():
    gtf_df = load_gtf()
    rows = [build_one(n, p, l, gtf_df) for (n, p, l) in LOCI]
    pd.DataFrame(rows).to_csv(RECHECK / "fig7AB_logsed.csv", index=False)
    print("saved", RECHECK / "fig7AB_logsed.csv")


if __name__ == "__main__":
    main()
