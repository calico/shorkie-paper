#!/usr/bin/env python3
"""Figure 7 J-O (deep-recheck) — six eQTL-SNP ISM saliency panels.

Each panel stacks, top to bottom (matching the published layout):
  1. Ref DB        — the known motif logo (Reb1.1 / PAC-Dot6 / polyA efficiency element),
                     embedded from the project motif DB (experiments/motif_DB/...).
  2. Shorkie ISM (REF) — per-base ISM saliency of the *reference* sequence, recomputed
                     from the cached 8-fold ensemble ISM (pred_ism_wt, mean-normalized x
                     ref one-hot), 80 bp around the SNP.
  3. Shorkie ISM (ALT) — same from the *alt* allele context (pred_ism_mut x alt one-hot).
  4. DREAM-RNN ISM (REF) — per-base saliency from the cached DREAM-RNN ISM TSV
                     (ref-average of the substitution deltas). [4/6 loci on disk.]
  5. DREAM-RNN ISM (ALT) — same from the alt-context DREAM ISM TSV.
  6. Shorkie Coverage — observed RNA-seq coverage over the gene neighbourhood (a faithful
                     proxy for the model's predicted coverage track; cf. panels A/B where
                     predicted vs observed correlate R>0.96), with the ISM window shaded.

The Shorkie ISM logos (rows 2-3) — the scientifically central, model-derived content — are
recomputed bit-for-bit from the released ISM cache. Rows 1/4/5 are rendered from cached
external/database artifacts; the published Avg-logSED / quantile annotations come from the
released scoring run (recorded in fig7JO_logsed.csv, not re-derived). See DISCREPANCIES.md.

Outputs: reproduced/Figure_7JO_reproduced.png
         recheck/fig7JO_logsed.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

from shorkie import config

config.load()
ROOT = str(config.path("work_root"))
REPRO = config.repo_root() / "reproduction" / "figure_07"
RECHECK = REPRO / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

ISM_DIR = f"{ROOT}/experiments/SUM_data_process/eQTL/eQTL_Shorkie_ISM_notebook/ism_results"
DREAM_DIR = f"{ROOT}/experiments/SUM_data_process/eQTL/eQTL_MPRA_models_ISM/results"
MOTIF_DIR = f"{ROOT}/experiments/motif_DB"
WIN_HALF = 40
NT = "ACGT"
NTI = {b: i for i, b in enumerate(NT)}

# published Avg logSED + |logSED|/|Δ| quantiles (from the released scoring; annotation only)
LOCI = [
    dict(panel="J", gene="YER080W", chrom="chrV",  pos=321901, motif="Polyadenylation efficiency elements",
         refdb="TATATA.png",     avg_logsed=0.145, q_logsed="99.10%", q_delta="97.62%"),
    dict(panel="K", gene="YLR036C", chrom="chrXII", pos=221376, motif="Polyadenylation efficiency elements",
         refdb="TATATA.png",     avg_logsed=-0.269, q_logsed="99.92%", q_delta="89.41%"),
    dict(panel="L", gene="YKL078W", chrom="chrXI", pos=288774, motif="PAC motif (Dot6)",
         refdb="PAC_motif.png",  avg_logsed=0.273, q_logsed="99.97%", q_delta="98.64%"),
    dict(panel="M", gene="YKR087C", chrom="chrXI", pos=604356, motif="Reb1.1",
         refdb="viz_self_motif_db/REB1.1.png", avg_logsed=-0.271, q_logsed="99.95%", q_delta="92.95%"),
    dict(panel="N", gene="YNL239W", chrom="chrXIV", pos=200328, motif="Reb1.1",
         refdb="viz_self_motif_db/REB1.1.png", avg_logsed=0.177, q_logsed="99.44%", q_delta="91.44%"),
    dict(panel="O", gene="YGR046W", chrom="chrVII", pos=584683, motif="Reb1.1",
         refdb="viz_self_motif_db/REB1.1.png", avg_logsed=0.163, q_logsed="99.72%", q_delta="18.10%"),
]

_FP = FontProperties(family="DejaVu Sans", weight="bold")
_LETTERS = {b: TextPath((-0.35, 0), b, size=1, prop=_FP) for b in NT}
_COLORS = {"G": "orange", "A": "green", "C": "blue", "T": "red"}


def dna_letter_at(letter, x, y, yscale, ax):
    t = mpl.transforms.Affine2D().scale(1.35, yscale * 1.35).translate(x, y) + ax.transData
    ax.add_artist(PathPatch(_LETTERS[letter], lw=0, fc=_COLORS[letter], transform=t))


def plot_logo(ax, imp, x_labels=None):
    L = imp.shape[0]
    for p in range(L):
        order = np.argsort(-np.abs(imp[p]))
        ph, nh = 0.0, 0.0
        for idx in order:
            v = imp[p, idx]
            if abs(v) < 1e-12:
                continue
            if v >= 0:
                dna_letter_at(NT[idx], p + 0.5, ph, v, ax); ph += v
            else:
                dna_letter_at(NT[idx], p + 0.5, nh, v, ax); nh += v
    ax.set_xlim(0, L)
    mx = max(np.max(np.abs(imp)), 1e-9)
    ax.set_ylim(-mx * 1.05, mx * 1.05)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_yticks([])
    ax.set_xticks([])


def onehot(seq):
    oh = np.zeros((len(seq), 4), dtype="float32")
    for i, b in enumerate(seq):
        j = NTI.get(b, -1)
        if j >= 0:
            oh[i, j] = 1.0
    return oh


def shorkie_logos(L, fa):
    z = np.load(f"{ISM_DIR}/{L['gene']}_{L['chrom']}_{L['pos']}.npz", allow_pickle=True)
    wt = z["pred_ism_wt"]; mut = z["pred_ism_mut"]
    start = int(z["start"]); end = int(z["end"]); cp = int(z["center_pos"])
    alt = str(z["alts"][0])
    seq = fa.fetch(L["chrom"], start, end).upper()
    ci = cp - start - 1
    ref_oh = onehot(seq)
    alt_oh = ref_oh.copy(); alt_oh[ci, :] = 0.0; alt_oh[ci, NTI[alt]] = 1.0
    wt_n = wt - wt.mean(axis=1, keepdims=True)
    mut_n = mut - mut.mean(axis=1, keepdims=True)
    imp_ref = wt_n * ref_oh
    imp_alt = mut_n * alt_oh
    lo, hi = ci - WIN_HALF, ci + WIN_HALF + 1
    ref_base = seq[ci]
    return imp_ref[lo:hi], imp_alt[lo:hi], ref_base, alt, float(np.max(np.abs(imp_ref[lo:hi])))


def dream_logos(L):
    """ref-average per-base saliency (W,4) from the cached DREAM-RNN ISM TSVs; None if absent."""
    out = {}
    for which, fn in [("ref", "ism_ref_results.tsv"), ("alt", "ism_alt_results.tsv")]:
        df = pd.read_csv(f"{DREAM_DIR}/{fn}", sep="\t")
        sub = df[df["ChrPos"] == L["pos"]]
        if len(sub) == 0:
            out[which] = None
            continue
        pmin, pmax = int(sub["pos"].min()), int(sub["pos"].max())
        center = (pmin + pmax) // 2
        W = 2 * WIN_HALF + 1
        imp = np.zeros((W, 4), dtype="float32")
        for _, r in sub.iterrows():
            wpos = int(r["pos"]) - (center - WIN_HALF)
            if 0 <= wpos < W:
                ob = str(r["orig_base"])
                if ob in NTI:
                    imp[wpos, NTI[ob]] += -float(r["delta"])  # negate: positive = important
        # average over the 3 substitutions per position
        imp /= 3.0
        out[which] = imp
    return out["ref"], out["alt"]


def coverage_track(ax, L):
    base = f"{ROOT}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
    from shorkie.viz.load_cov import read_coverage
    sheet = pd.read_csv(f"{base}/cleaned_sheet_RNA-Seq_T0.txt", sep="\t", index_col=0)
    files = sheet["file"].iloc[::max(1, len(sheet) // 12)].head(12).tolist()
    half = 1024
    g0, g1 = L["pos"] - half, L["pos"] + half
    covs = []
    for f in files:
        try:
            cv = np.asarray(read_coverage(f, L["chrom"], g0, g1), dtype="float32")
            if len(cv) == 2 * half:
                covs.append(cv)
        except Exception:
            continue
    cov = np.mean(covs, axis=0) if covs else np.zeros(2 * half)
    # bin to 16 bp for a smoother track
    nb = len(cov) // 16
    cov = cov[:nb * 16].reshape(nb, 16).mean(axis=1)
    x = np.linspace(g0, g1, nb)
    ax.fill_between(x, 0, cov, color="#d9a066", lw=0)
    ax.axvspan(L["pos"] - WIN_HALF, L["pos"] + WIN_HALF, color="0.6", alpha=0.3)
    ax.axvline(L["pos"], color="black", ls=":", lw=0.7)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_xticks([g0, L["pos"], g1])
    ax.set_xticklabels([f"{g0}", f"{L['pos']}", f"{g1}"], fontsize=6)


def add_refdb(ax, L):
    p = os.path.join(MOTIF_DIR, L["refdb"])
    ax.axis("off")
    if os.path.exists(p):
        try:
            img = plt.imread(p)
            ax.imshow(img, aspect="auto")
        except Exception:
            ax.text(0.5, 0.5, L["motif"], ha="center", va="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, L["motif"], ha="center", va="center", fontsize=7)


def main():
    fa = pysam.Fastafile(str(config.path("genome.fasta")))
    fig = plt.figure(figsize=(18, 13))
    outer = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.13)
    rows = []
    ROW_LABELS = ["Ref DB", "Shorkie\nISM (REF)", "Shorkie\nISM (ALT)",
                  "DREAM-RNN\nISM (REF)", "DREAM-RNN\nISM (ALT)", "Shorkie\nCoverage"]
    for k, L in enumerate(LOCI):
        r, c = divmod(k, 3)
        inner = GridSpecFromSubplotSpec(6, 1, subplot_spec=outer[r, c],
                                        height_ratios=[1.1, 1.3, 1.3, 1.1, 1.1, 1.5], hspace=0.18)
        axs = [fig.add_subplot(inner[i]) for i in range(6)]
        imp_ref, imp_alt, ref_base, alt, maxabs = shorkie_logos(L, fa)
        d_ref, d_alt = dream_logos(L)

        add_refdb(axs[0], L)
        plot_logo(axs[1], imp_ref)
        plot_logo(axs[2], imp_alt)
        for ax, d in [(axs[3], d_ref), (axs[4], d_alt)]:
            if d is None:
                ax.text(0.5, 0.5, "DREAM-RNN ISM not on disk", ha="center", va="center",
                        fontsize=7, color="gray"); ax.set_xticks([]); ax.set_yticks([])
            else:
                plot_logo(ax, d)
        coverage_track(axs[5], L)

        for ax, lab in zip(axs, ROW_LABELS):
            ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center", labelpad=22)
        w0, w1 = L["pos"] - WIN_HALF, L["pos"] + WIN_HALF
        axs[0].set_title(f"{L['panel']}  {L['chrom']}:{w0:,}-{w1:,}    {L['motif']}\n"
                         f"{L['gene']}  SNP {L['chrom']}:{L['pos']} {ref_base}>{alt}   "
                         f"Avg logSED={L['avg_logsed']:+.3f} (released)",
                         fontsize=8, loc="left")
        rows.append(dict(panel=L["panel"], gene=L["gene"], snp=f"{L['chrom']}:{L['pos']}",
                         ref=ref_base, alt=alt, motif=L["motif"],
                         shorkie_ism_maxabs=round(maxabs, 5),
                         dream_ism_on_disk=(d_ref is not None),
                         avg_logsed_published=L["avg_logsed"],
                         q_logsed_published=L["q_logsed"], q_delta_published=L["q_delta"]))
    fa.close()
    fig.suptitle("Figure 7 J-O (reproduced) — eQTL-SNP ISM saliency (Shorkie ISM logos recomputed from cache)",
                 fontsize=14, y=0.995)
    out = REPRO / "reproduced" / "Figure_7JO_reproduced.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", out)
    pd.DataFrame(rows).to_csv(RECHECK / "fig7JO_logsed.csv", index=False)
    print("saved", RECHECK / "fig7JO_logsed.csv")
    for rdict in rows:
        print(f"  {rdict['panel']} {rdict['gene']} {rdict['ref']}>{rdict['alt']} "
              f"motif={rdict['motif']} dream={rdict['dream_ism_on_disk']} maxISM={rdict['shorkie_ism_maxabs']}")


if __name__ == "__main__":
    main()
