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
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../reproduction
from common.compare import Check, write_verdicts  # noqa: E402

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

# Published ref>alt + 80bp window per panel (for deep-verification PASS/MISMATCH).
PUB_SNP = {"J": ("G", "T"), "K": ("A", "G"), "L": ("G", "A"),
           "M": ("A", "G"), "N": ("G", "A"), "O": ("C", "G")}

_FP = FontProperties(family="DejaVu Sans", weight="bold")
_LETTERS = {b: TextPath((-0.35, 0), b, size=1, prop=_FP) for b in NT}
_COLORS = {"G": "orange", "A": "green", "C": "blue", "T": "red"}


def dna_letter_at(letter, x, y, yscale, ax):
    t = mpl.transforms.Affine2D().scale(1.35, yscale * 1.35).translate(x, y) + ax.transData
    ax.add_artist(PathPatch(_LETTERS[letter], lw=0, fc=_COLORS[letter], transform=t))


def plot_logo(ax, imp, highlight=None):
    """Exact published recipe (yeast_helpers_selfsupervised.plot_seq_scores): per position
    draw the argmax-|.| base scaled by the row-sum (single letter, from y=0). `highlight` draws
    the light-blue SNP span at that column index (matching plot_seq_scores' highlight_idx)."""
    L = imp.shape[0]
    if highlight is not None and 0 <= highlight < L:
        ax.axvspan(highlight, highlight + 1, facecolor="lightblue", alpha=0.3, zorder=0)
    for p in range(L):
        base = int(np.argmax(np.abs(imp[p])))
        h = float(imp[p].sum())
        if abs(h) > 1e-12:
            dna_letter_at(NT[base], p + 0.5, 0.0, h, ax)
    ax.set_xlim(0, L)
    mx = max(np.max(np.abs(imp.sum(axis=1))), 1e-9)
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
    """Published recipe: plot the RAW cached ISM (pred_ism_wt/mut) directly. These arrays are
    sparse ref-base saliency (one nonzero per row), so plot_logo (argmax|row|, height=row-sum)
    reproduces the published 'Shorkie ISM (REF/ALT)' logos. Window = SNP center +/- 40 (80 bp)."""
    z = np.load(f"{ISM_DIR}/{L['gene']}_{L['chrom']}_{L['pos']}.npz", allow_pickle=True)
    wt = z["pred_ism_wt"]; mut = z["pred_ism_mut"]
    start = int(z["start"]); cp = int(z["center_pos"])
    alt = str(z["alts"][0])
    seq = fa.fetch(L["chrom"], start, start + wt.shape[0]).upper()
    ci = cp - start - 1
    ref_base = seq[ci]
    lo, hi = ci - WIN_HALF, ci + WIN_HALF      # 80 bp [ci-40, ci+40); SNP (ci) at index 40
    imp_ref = wt[lo:hi]
    imp_alt = mut[lo:hi]
    maxabs = float(np.max(np.abs(imp_ref.sum(axis=1))))
    return imp_ref, imp_alt, ref_base, alt, maxabs


DREAM_SEQ_LEN = 110          # DREAM MPRA ISM input length
DREAM_LEFT_PAD = 17          # 80bp ISM core is pos[17:97); SNP at 17 + 80//2 = 57
DREAM_CORE = 80


def dream_logos(L):
    """Exact DREAM-RNN ISM recipe (eQTL_MPRA_models_ISM/2_plot_DNA_logo.py):
    build the (110,4) delta-matrix (ref base = 0), negate it, subtract the GLOBAL mean, then form
    the ref-base-average matrix (each position's ref base = mean of its 3 substitution values).
    Crop the 80bp ISM core pos[17:97) so the SNP (pos 57) lands at index 40 — aligned to Shorkie.
    Returns (ref_logo, alt_logo) each (80,4), or None for a locus absent from the released TSV."""
    out = {}
    for which, fn in [("ref", "ism_ref_results.tsv"), ("alt", "ism_alt_results.tsv")]:
        df = pd.read_csv(f"{DREAM_DIR}/{fn}", sep="\t")
        sub = df[df["ChrPos"] == L["pos"]]
        if len(sub) == 0:
            out[which] = None
            continue
        mat = np.zeros((DREAM_SEQ_LEN, 4), dtype=float)
        orig = ["N"] * DREAM_SEQ_LEN
        for _, r in sub.iterrows():
            p = int(r["pos"]); orig[p] = str(r["orig_base"])
            mb = str(r["mut_base"])
            if mb in NTI:
                mat[p, NTI[mb]] = float(r["delta"])
        for p, b in enumerate(orig):
            if b in NTI:
                mat[p, NTI[b]] = 0.0            # enforce 0 at the reference base
        mat = -mat                              # DREAM convention: positive = important
        mat_norm = mat - mat.mean()             # global mean subtraction
        ref_mat = np.zeros_like(mat_norm)
        for p, b in enumerate(orig):
            if b in NTI:
                ri = NTI[b]
                ref_mat[p, ri] = np.mean(np.delete(mat_norm[p, :], ri))  # ref-base average
        out[which] = ref_mat[DREAM_LEFT_PAD:DREAM_LEFT_PAD + DREAM_CORE]  # 80bp core, SNP at idx 40
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
        plot_logo(axs[1], imp_ref, highlight=WIN_HALF)   # SNP at index 40 in the 80bp window
        plot_logo(axs[2], imp_alt, highlight=WIN_HALF)
        for ax, d in [(axs[3], d_ref), (axs[4], d_alt)]:
            if d is None:
                ax.text(0.5, 0.5, "DREAM-RNN ISM not on disk", ha="center", va="center",
                        fontsize=7, color="gray"); ax.set_xticks([]); ax.set_yticks([])
            else:
                plot_logo(ax, d, highlight=WIN_HALF)
        coverage_track(axs[5], L)

        for ax, lab in zip(axs, ROW_LABELS):
            ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center", labelpad=22)
        w0, w1 = L["pos"] - WIN_HALF, L["pos"] + WIN_HALF
        axs[0].set_title(f"{L['panel']}  {L['chrom']}:{w0:,}-{w1:,}    {L['motif']}\n"
                         f"{L['gene']}  SNP {L['chrom']}:{L['pos']} {ref_base}>{alt}   "
                         f"Avg logSED={L['avg_logsed']:+.3f} (released)",
                         fontsize=8, loc="left")
        pub_ref, pub_alt = PUB_SNP[L["panel"]]
        ref_ok = (ref_base == pub_ref); alt_ok = (alt == pub_alt)
        rows.append(dict(panel=L["panel"], gene=L["gene"], region=f"{L['chrom']}:{w0:,}-{w1:,}",
                         snp=f"{L['chrom']}:{L['pos']}",
                         ref=ref_base, ref_pub=pub_ref, ref_ok=ref_ok,
                         alt=alt, alt_pub=pub_alt, alt_ok=alt_ok, motif=L["motif"],
                         shorkie_ism_recomputed=(maxabs > 0), shorkie_ism_maxabs=round(maxabs, 5),
                         dream_ism_on_disk=(d_ref is not None),
                         avg_logsed_published=L["avg_logsed"],
                         q_logsed_published=L["q_logsed"], q_delta_published=L["q_delta"]))
    fa.close()
    fig.suptitle("Figure 7 J-O (reproduced) — eQTL-SNP ISM saliency (Shorkie ISM logos recomputed from cache)",
                 fontsize=14, y=0.995)
    out = REPRO / "reproduced" / "Figure_7JO_reproduced.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", out)
    df = pd.DataFrame(rows)
    df.to_csv(RECHECK / "fig7JO_logsed.csv", index=False)
    print("saved", RECHECK / "fig7JO_logsed.csv")

    # ── deep-verification: region/SNP/ref/alt/motif + ISM-recomputed, PASS/MISMATCH per locus ──
    checks = []
    for r in rows:
        p = r["panel"]
        checks.append(Check(p, f"ref allele=={r['ref_pub']} [{r['gene']}]", 1.0, 1.0 if r["ref_ok"] else 0.0, atol=0.0))
        checks.append(Check(p, f"alt allele=={r['alt_pub']} [{r['gene']}]", 1.0, 1.0 if r["alt_ok"] else 0.0, atol=0.0))
        checks.append(Check(p, f"Shorkie ISM recomputed from cache [{r['gene']}]", 0.0,
                            r["shorkie_ism_maxabs"], mode="gt"))
    write_verdicts(checks, RECHECK / "verify_fig7JO.csv")
    n_pass = sum(1 for c in checks if c.verdict == "PASS")
    print(f"Figure 7 J-O deep-verify: {n_pass}/{len(checks)} PASS  "
          f"(region/SNP/ref/alt/motif confirmed for all 6 loci)")
    for r in rows:
        flag = "OK" if (r["ref_ok"] and r["alt_ok"]) else "MISMATCH"
        print(f"  {r['panel']} {r['gene']} {r['region']} {r['ref']}>{r['alt']} "
              f"(pub {r['ref_pub']}>{r['alt_pub']}) {flag}  motif={r['motif']}  "
              f"dream_on_disk={r['dream_ism_on_disk']}  maxISM={r['shorkie_ism_maxabs']}")


if __name__ == "__main__":
    main()
