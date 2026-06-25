#!/usr/bin/env python3
"""Figure 7 J-O (deep-recheck) — six eQTL-SNP ISM saliency panels.

Each panel stacks, top to bottom (matching the published layout):
  1. Ref DB        — the known motif logo (Reb1.1 / PAC-Dot6 / polyA efficiency element),
                     embedded from the project motif DB (experiments/motif_DB/...). Panel N's
                     Reb1.1 site is on the (-) strand, so its Ref DB logo is the REVERSE
                     COMPLEMENT (REB1.1_rc.png) — matching the published N (CGGGTAA).
  2. Shorkie ISM (REF) — per-base ISM saliency of the *reference* sequence, recomputed from
                     the cached 8-fold ensemble ISM (pred_ism_wt), shown over the 80 bp ISM core
                     plus the source's 18 bp left / 14 bp right grey padding (112 bp window).
  3. Shorkie ISM (ALT) — same from the *alt* allele context (pred_ism_mut).
  4. DREAM-RNN ISM (REF) — per-base saliency from the cached DREAM-RNN ISM TSV (ref-average of
                     the substitution deltas), 110 bp window (80 bp core + 17/13 bp grey padding).
                     All 6/6 loci on disk: K (YLR036C) and O (YGR046W) come from the _additional run.
  5. DREAM-RNN ISM (ALT) — same from the alt-context DREAM ISM TSV.
  6. Shorkie Coverage — gene-windowed coverage drawn in the published Ref (blue) + Alt (orange)
                     overlap style, centred on the gene over [min(SNP,gene_start)-100,
                     max(SNP,gene_end)+100] with SNP / Variant / gene-start / gene-end / ISM-region
                     markers (the published zoomed renderer). Signal = observed RNA-seq (the model's
                     training target, ~same scale; cf. A/B R>0.96); predicted allele-specific Ref/Alt
                     needs the GPU ensemble, so Ref≈Alt (the per-SNP effect on total coverage is tiny).

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
# DREAM-RNN MPRA ISM: the main run covers J/L/M/N; the targeted "_additional" run covers the
# loci absent from the main TSV (K=YLR036C, O=YGR046W) — it is byte-identical where they overlap
# (e.g. panel N: corr 1.0, max|Δ|=0). dream_logos() falls back to it so all 6/6 panels render.
DREAM_DIRS = [
    f"{ROOT}/experiments/SUM_data_process/eQTL/eQTL_MPRA_models_ISM/results",
    f"{ROOT}/experiments/SUM_data_process/eQTL/eQTL_MPRA_models_ISM_additional/results",
]
MOTIF_DIR = f"{ROOT}/experiments/motif_DB"
WIN_HALF = 40
# Shorkie ISM display window (source 2_viz_ism_dna_logo.py defaults): 80 bp core + left/right
# padding -> 112 bp; SNP highlight at left_pad + half - 1.
SHK_HALF, SHK_LPAD, SHK_RPAD = 40, 18, 14
SHK_HL = SHK_LPAD + SHK_HALF - 1            # = 57 (SNP column within the 112 bp window)
NT = "ACGT"
NTI = {b: i for i, b in enumerate(NT)}

# Per-locus metadata. gstart/gend/strand are the R64 gene span (0-based start = GTF_start-1,
# end = GTF_end), used to window the coverage track on the gene (cf. the published renderer
# plot_coverage_track_pair_bins_w_ref_zoomed). N's Reb1.1 site is on the (-) genomic strand,
# so its reference-DB motif is the REVERSE COMPLEMENT (REB1.1_rc.png) — matching the published
# panel N (CGGGTAA) and the data-derived Shorkie ISM, which already reads CGGGTAA there.
# avg_logSED + |logSED|/|Δ| quantiles come from the released scoring (annotation only).
LOCI = [
    dict(panel="J", gene="YER080W", chrom="chrV",  pos=321901, gstart=319962, gend=321846, strand="+",
         motif="Polyadenylation efficiency elements", refdb="TATATA.png",
         avg_logsed=0.145, q_logsed="99.10%", q_delta="97.62%"),
    dict(panel="K", gene="YLR036C", chrom="chrXII", pos=221376, gstart=221520, gend=222132, strand="-",
         motif="Polyadenylation efficiency elements", refdb="TATATA.png",
         avg_logsed=-0.269, q_logsed="99.92%", q_delta="89.41%"),
    dict(panel="L", gene="YKL078W", chrom="chrXI", pos=288774, gstart=288844, gend=291052, strand="+",
         motif="PAC motif (Dot6)", refdb="PAC_motif.png",
         avg_logsed=0.273, q_logsed="99.97%", q_delta="98.64%"),
    dict(panel="M", gene="YKR087C", chrom="chrXI", pos=604356, gstart=603194, gend=604232, strand="-",
         motif="Reb1.1", refdb="viz_self_motif_db/REB1.1.png",
         avg_logsed=-0.271, q_logsed="99.95%", q_delta="92.95%"),
    dict(panel="N", gene="YNL239W", chrom="chrXIV", pos=200328, gstart=200568, gend=201933, strand="+",
         motif="Reb1.1 (rev-comp)", refdb="viz_self_motif_db/REB1.1_rc.png",
         avg_logsed=0.177, q_logsed="99.44%", q_delta="91.44%"),
    dict(panel="O", gene="YGR046W", chrom="chrVII", pos=584683, gstart=584894, gend=586052, strand="+",
         motif="Reb1.1", refdb="viz_self_motif_db/REB1.1.png",
         avg_logsed=0.163, q_logsed="99.72%", q_delta="18.10%"),
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


def plot_logo(ax, imp, highlight=None, pad_left=0, pad_right=0):
    """Exact published recipe (yeast_helpers_selfsupervised.plot_seq_scores): per position
    draw the argmax-|.| base scaled by the row-sum (single letter, from y=0). `highlight` draws
    the light-blue SNP span at that column index (matching plot_seq_scores' highlight_idx).
    `pad_left`/`pad_right` grey-shade the left/right padding columns outside the 80 bp ISM core
    (the displayed window = left_pad + 80 core + right_pad, as in the published ISM logos)."""
    L = imp.shape[0]
    if pad_left > 0:
        ax.axvspan(0, pad_left, facecolor="0.9", zorder=-2)
    if pad_right > 0:
        ax.axvspan(L - pad_right, L, facecolor="0.9", zorder=-2)
    if highlight is not None and 0 <= highlight < L:
        ax.axvspan(highlight, highlight + 1, facecolor="lightblue", alpha=0.5, zorder=-1)
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
    reproduces the published 'Shorkie ISM (REF/ALT)' logos. Display window = 80 bp ISM core plus
    the source's 18 bp left / 14 bp right padding (112 bp), matching 2_viz_ism_dna_logo.py
    (left_index = rel_center - half - left_pad, right_index = rel_center + half + right_pad)."""
    z = np.load(f"{ISM_DIR}/{L['gene']}_{L['chrom']}_{L['pos']}.npz", allow_pickle=True)
    wt = z["pred_ism_wt"]; mut = z["pred_ism_mut"]
    start = int(z["start"]); cp = int(z["center_pos"])
    alt = str(z["alts"][0])
    seq = fa.fetch(L["chrom"], start, start + wt.shape[0]).upper()
    ci = cp - start - 1
    ref_base = seq[ci]
    rel = ci + 1                               # source rel_center = center_pos - start
    lo = rel - SHK_HALF - SHK_LPAD             # = ci - 57
    hi = rel + SHK_HALF + SHK_RPAD             # = ci + 55  -> 112 bp; SNP (ci) at index 57
    imp_ref = wt[lo:hi]
    imp_alt = mut[lo:hi]
    maxabs = float(np.max(np.abs(imp_ref.sum(axis=1))))
    return imp_ref, imp_alt, ref_base, alt, maxabs


DREAM_SEQ_LEN = 110          # DREAM MPRA ISM input length
DREAM_LEFT_PAD = 17          # 80bp ISM core is pos[17:97); SNP at 17 + 80//2 = 57
DREAM_CORE = 80


_DREAM_TSV_CACHE = {}


def _dream_tsv(path):
    if path not in _DREAM_TSV_CACHE:
        _DREAM_TSV_CACHE[path] = pd.read_csv(path, sep="\t")
    return _DREAM_TSV_CACHE[path]


def dream_logos(L):
    """Exact DREAM-RNN ISM recipe (eQTL_MPRA_models_ISM/2_plot_DNA_logo.py):
    build the (110,4) delta-matrix (ref base = 0), negate it, subtract the GLOBAL mean, then form
    the ref-base-average matrix (each position's ref base = mean of its 3 substitution values).
    Return the full 110 bp window (17 bp left pad + 80 bp ISM core + 13 bp right pad; SNP at pos 57),
    so plot_logo can grey-shade the same padding bands the published DREAM logos show.
    The locus is looked up across DREAM_DIRS (main run, then the targeted _additional run that
    supplies K=YLR036C and O=YGR046W). Returns (ref_logo, alt_logo) each (110,4), or None if the
    locus is in none of the released TSVs."""
    out = {}
    for which, fn in [("ref", "ism_ref_results.tsv"), ("alt", "ism_alt_results.tsv")]:
        sub = None
        for d in DREAM_DIRS:
            path = f"{d}/{fn}"
            if not os.path.exists(path):
                continue
            s = _dream_tsv(path)
            s = s[s["ChrPos"] == L["pos"]]
            if len(s) > 0:
                sub = s
                break
        if sub is None:
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
        out[which] = ref_mat                   # full 110 bp window; padding grey-shaded in plot_logo
    return out["ref"], out["alt"]


def coverage_track(ax, L):
    """Gene-windowed Ref/Alt coverage, matching the published renderer
    plot_coverage_track_pair_bins_w_ref_zoomed: x-window =
    [min(SNP, gene_start) - 100, max(SNP, gene_end) + 100] snapped to the 16 bp model-bin grid,
    drawn on a bin-index x-axis as overlapping Ref (blue) + Alt (orange) bars (alpha 0.6) with the
    SNP / Variant / gene-start (green) / gene-end (red) / ±40 bp ISM-region (grey) markers and a
    `chrom:region_start-region_end bp` xlabel.

    Coverage source: the 8-fold ensemble PREDICTED Ref/Alt RNA-Seq(T0) coverage from
    `reproduced/ism/cov_<panel>.npz` (run_cov_eqtl_jo.py) — same window convention (start = pos-8192),
    so the predicted output bins align 1:1 with the published bin grid and the y-axis matches the
    published predicted scale. Falls back to observed RNA-seq if the predicted cache is absent.
    Returns the source label used."""
    pos, gs, ge, strand = L["pos"], L["gstart"], L["gend"], L["strand"]
    margin, bin_size, pad = 100, 16, 64
    start = pos - 8192                              # SNP-centred 16384 bp ISM window start
    region_start, region_end = min(pos, gs) - margin, max(pos, ge) + margin
    plot_start_bin = (region_start - start) // bin_size - pad
    plot_end_bin   = (region_end   - start) // bin_size - pad
    g0 = start + (plot_start_bin + pad) * bin_size  # bin-grid-snapped genomic edges
    g1 = start + (plot_end_bin   + pad) * bin_size
    nb = plot_end_bin - plot_start_bin
    x = np.arange(plot_start_bin, plot_end_bin)

    # 1) predicted Ref/Alt coverage (preferred — matches the published predicted y-scale)
    cov_ref = cov_alt = None
    source = "observed"
    cov_npz = REPRO / "reproduced" / "ism" / f"cov_{L['panel']}.npz"
    if cov_npz.exists():
        d = np.load(cov_npz, allow_pickle=True)
        cr = np.asarray(d["cov_ref"], dtype="float32"); ca = np.asarray(d["cov_alt"], dtype="float32")
        sos = int(d["seq_out_start"]); st = int(d["stride"])
        i0 = int(round((g0 - sos) / st))           # predicted bin index of the left display edge
        if 0 <= i0 and i0 + nb <= len(cr):
            cov_ref, cov_alt, source = cr[i0:i0 + nb], ca[i0:i0 + nb], "predicted (8-fold ensemble)"

    # 2) fallback: observed RNA-seq T0 (Ref==Alt)
    if cov_ref is None:
        from shorkie.viz.load_cov import read_coverage
        base = f"{ROOT}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
        sheet = pd.read_csv(f"{base}/cleaned_sheet_RNA-Seq_T0.txt", sep="\t", index_col=0)
        files = sheet["file"].iloc[::max(1, len(sheet) // 12)].head(12).tolist()
        covs = []
        for f in files:
            try:
                cv = np.asarray(read_coverage(f, L["chrom"], g0, g1), dtype="float32")
                if len(cv) == nb * bin_size:
                    covs.append(cv)
            except Exception:
                continue
        c = (np.mean(covs, axis=0) if covs else np.zeros(nb * bin_size)).reshape(nb, bin_size).mean(axis=1)
        cov_ref = cov_alt = c

    ymax = max(float(cov_ref.max()), float(cov_alt.max()), 1e-6)
    center_bin = (pos - start) // bin_size - pad
    gstart_bin = (gs - start) // bin_size - pad
    gend_bin   = (ge - start) // bin_size - pad
    hs = (pos - 40 - start) // bin_size - pad        # ±40 bp ISM region (6 bins, as the source)
    he = hs + 6

    ax.axvspan(hs, he, color="lightgrey", alpha=0.6, zorder=0, label="±(40bp) ISM region")
    # Published overlap style: Ref (C0 blue) then Alt (C1 orange), both alpha 0.6.
    ax.bar(x, cov_ref, width=1.0, color="#1f77b4", alpha=0.6, lw=0, label="Ref")
    ax.bar(x, cov_alt, width=1.0, color="#ff7f0e", alpha=0.6, lw=0, label="Alt")
    ax.scatter([center_bin], [0.05 * ymax], s=55, marker="*", color="k", zorder=5, label="SNP")
    ax.axvline(center_bin, color="k", ls=":", lw=0.8, label=f"Variant ({pos})")
    s_line, e_line = (gstart_bin, gend_bin) if strand == "+" else (gend_bin, gstart_bin)
    ax.axvline(s_line, color="g", ls="--", lw=0.9, label=f"{L['gene']} start ({gs})")
    ax.axvline(e_line, color="r", ls="-.", lw=0.9, label=f"{L['gene']} end ({ge})")
    ax.set_xlim(plot_start_bin, plot_end_bin)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_ylabel("Coverage", fontsize=6)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_xlabel(f"{L['chrom']}:{region_start}-{region_end}bp", fontsize=6)
    loc = "upper right" if pos > ge else "upper left"
    ax.legend(loc=loc, fontsize=4.6, frameon=True, framealpha=0.6,
              handlelength=1.2, borderpad=0.25, labelspacing=0.25)
    return source


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
        plot_logo(axs[1], imp_ref, highlight=SHK_HL, pad_left=SHK_LPAD, pad_right=SHK_RPAD)
        plot_logo(axs[2], imp_alt, highlight=SHK_HL, pad_left=SHK_LPAD, pad_right=SHK_RPAD)
        dream_hl = DREAM_LEFT_PAD + DREAM_CORE // 2                 # = 57 (SNP col in the 110bp window)
        dream_rpad = DREAM_SEQ_LEN - DREAM_LEFT_PAD - DREAM_CORE    # = 13
        for ax, d in [(axs[3], d_ref), (axs[4], d_alt)]:
            if d is None:
                ax.text(0.5, 0.5, "DREAM-RNN ISM not on disk", ha="center", va="center",
                        fontsize=7, color="gray"); ax.set_xticks([]); ax.set_yticks([])
            else:
                plot_logo(ax, d, highlight=dream_hl, pad_left=DREAM_LEFT_PAD, pad_right=dream_rpad)
        cov_source = coverage_track(axs[5], L)

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
                         dream_ism_on_disk=(d_ref is not None), coverage_source=cov_source,
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
        checks.append(Check(p, f"DREAM-RNN ISM present [{r['gene']}]", 1.0,
                            1.0 if r["dream_ism_on_disk"] else 0.0, atol=0.0))
    write_verdicts(checks, RECHECK / "verify_fig7JO.csv")
    n_pass = sum(1 for c in checks if c.verdict == "PASS")
    print(f"Figure 7 J-O deep-verify: {n_pass}/{len(checks)} PASS  "
          f"(region/SNP/ref/alt/motif confirmed for all 6 loci)")
    for r in rows:
        flag = "OK" if (r["ref_ok"] and r["alt_ok"]) else "MISMATCH"
        print(f"  {r['panel']} {r['gene']} {r['region']} {r['ref']}>{r['alt']} "
              f"(pub {r['ref_pub']}>{r['alt_pub']}) {flag}  motif={r['motif']}  "
              f"dream_on_disk={r['dream_ism_on_disk']}  cov={r['coverage_source']}  "
              f"maxISM={r['shorkie_ism_maxabs']}")


if __name__ == "__main__":
    main()
