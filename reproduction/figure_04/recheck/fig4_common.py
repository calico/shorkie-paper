#!/usr/bin/env python3
"""Shared helpers for the Figure 4 deep-recheck builders (recheck/build_4*.py).

Figure 4 = "Shorkie uses promoter and splicing motifs learned during pretraining."
All panels are CPU-reproducible from cached numeric data (no GPU). This module
centralises the exact recipes reverse-engineered from the source scripts:

  - Shorkie ISM saliency   : logSED in scores.h5, averaged over the 384 T0 RNA-seq
                             tracks (track_offset 1148), mean-centred over the 4
                             bases, projected on the reference one-hot.
                             (scripts/.../motif_shorkie__RP_TSS/1_plot_dna_logo_general.py)
  - Shorkie LM saliency    : LM masked-prediction probs x_pred from preds.npz,
                             transformed x_pred*log(x_pred/mean_over_window),
                             over a 450-up/50-down TSS window.
                             (scripts/.../shorkie_lm/.../motif_lm__RP_TSS/2_modisco_DNA_logo.py)
  - Database motif PWMs     : merged_meme_high_conf.meme -> IC-weighted (panels D,H).
  - MoDISco reconstruction  : contrib_scores CWM from modisco_results_10000_500.h5.

Logo glyphs use the project's custom DejaVu letters (A=green, C=blue, G=orange,
T=red) so the reproduced panels match the published rendering. Rows in panels A/B/C
are aligned on a shared **genomic** x-axis so the LM / ISM / Random / Reference-DB
tracks line up despite the ~120 bp LM-vs-ISM window offset for FUN12/KRE33.
"""
from __future__ import annotations
import os, re, json, shutil
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import h5py
import pysam
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle, Polygon

from shorkie import config
from shorkie.helpers.yeast_helpers import make_seq_1hot

config.load()
WORK = Path(config.path("work_root"))
EXP = WORK / "experiments"
REPRO = config.repo_root() / "reproduction" / "figure_04"
RECHECK = REPRO / "recheck"
RD = REPRO / "reproduced"
RD.mkdir(parents=True, exist_ok=True)

ISM_ROOT = Path(config.path("results.ism_scores"))            # .../SUM_data_process/motifs
MODISCO_ROOT = Path(config.path("results.modisco_ism"))       # .../motif_shorkie_RP_TSS/1_modisco_analysis
LM_ROOT = EXP / "motif_LM_RP_TSS"
MOTIF_DB_DIR = Path(config.path("motif_db_dir"))             # external yeast motif DB (config key: motif_db_dir)
MEME_HIGH_CONF = MOTIF_DB_DIR / "merged_meme_high_conf.meme"
MOTIF_DB_ASSETS = EXP / "motif_DB"                            # cached GTATGT/TACTAAC/PAC_motif/... PNG+pfm
GTF = str(config.path("genome.gtf"))
FASTA = str(config.path("genome.fasta"))
TARGETS = str(config.path("datasets.targets_sheet"))
LM_ARCH = "unet_small_bert_drop"                              # base LM replicate used by the figure
BED_DIR = WORK / "data" / "gene_exp_ism_window"

# ---- TomTom (MEME suite) binary, resolved like the upstream pipeline ----------------
# (scripts/.../1_map_modisco_pattern_to_meme_db.py uses shutil.which("tomtom")). Prefer
# the optional tools.tomtom_bin config key, else fall back to `tomtom` on PATH. Only
# run_tomtom.py needs this (to regenerate tomtom_RP_matches.tsv); the committed TSV +
# cached report/motifs.html let panel H reproduce without MEME installed.
def _resolve_tomtom():
    cfg = config.get("tools.tomtom_bin")
    if cfg:
        p = Path(os.path.expanduser(str(cfg)))
        if p.exists():
            return str(p)
        w = shutil.which(str(cfg))
        if w:
            return w
    return shutil.which("tomtom")                            # None if MEME not installed

TOMTOM_BIN = _resolve_tomtom()

# ---- fasta + chrom remap (scores.h5 use chrI..; GTF uses bare Roman I..) -----------
_fa = pysam.Fastafile(FASTA)
_FA_REFS = set(_fa.references)

def to_fa(chrom: str) -> str:
    if chrom in _FA_REFS:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    return alt if alt in _FA_REFS else chrom

# ---- T0 RNA-seq track subset the ISM was averaged over ----------------------------
@lru_cache(maxsize=1)
def t0_tracks():
    track_offset = 1148
    tgt = pd.read_csv(TARGETS, sep="\t")
    t0 = [int(i) - track_offset for i in tgt[tgt["identifier"].str.contains("_T0_")]["index"].astype(int)]
    return np.array([t for t in t0 if 0 <= t < 3053], dtype=int)

# ======================================================================================
# Panel-gene registry
# ======================================================================================
# display label for the Reference-DB / TF boxes (RRPE=Stb3, PAC=Dot6 per the manuscript)
TF_DISPLAY = {"FHL1": "Fhl1", "RAP1": "Rap1", "ABF1": "Abf1",
              "STB3": "RRPE (Stb3)", "DOT6": "PAC (Dot6)"}
# Splicing genes used by panel D (donor/branch reconstruction from the SS ISM PWM) — left
# as-is (panel D is out of scope). NOTE: this is *not* the ISM-logo panel registry; the
# clean per-gene ISM logos are driven by PANELS (below).
SPLICE = [
    dict(panel="E", gene="DTD1", part=22),
    dict(panel="F", gene="MMS2", part=57),
    dict(panel="G", gene="HOP2", part=80),
]
# Exact genomic window shown in each published Figure-4 panel header (PDF titles). Every
# row of a panel is drawn over this window. The correct on-disk ISM/Random entries (see
# PANELS) match these windows exactly (RPL26A/FUN12/MMS2), to the base (KRE33 −1 bp), or
# fully contain them (DTD1/HOP2 SS windows → cropped via slice_to_window).
PUB_WIN = {
    "RPL26A": ("chrXII", 818862, 819362),   # A — Shorkie-ISM _RP part4 idx10 (exact)
    "FUN12":  ("chrI",   75977,  76477),    # B — Shorkie-ISM _TSS part0 idx39 (exact)
    "KRE33":  ("chrXIV", 374871, 375371),   # C — Shorkie-ISM _RRB part11 idx3 (−1 bp)
    "DTD1":   ("chrIV",  65235,  65431),    # E — within SS data window (full crop)
    "MMS2":   ("chrVII", 346669, 347169),   # F — Shorkie-ISM _TSS part33 idx31 (exact; MMS2/MAD1 locus)
    "HOP2":   ("chrVII", 435625, 436401),   # G — within SS data window (full crop)
}

# ---------------------------------------------------------------------------------------
# PANELS — the clean per-gene ISM-logo registry (single source of truth for the saliency
# rows). Each gene declares the exact on-disk entry for every model row it has:
#   win    : published window (chrom, start, end) — drives every row's x-range
#   lm     : (lm_set, lm_row) for the Shorkie-LM masked-prediction row, or None
#   ism    : (sub, part, idx) Shorkie-ISM entry in tree motif_shorkie_RP_TSS
#   random : (tree, sub, part, idx) Random-Init (scratch) ISM entry, or None
# Published A,B,C,F = 3 model rows (LM+ISM+Random); E,G = ISM-only. (Boxes/TF text are
# NOT drawn — clean logos, mirroring the original 2_modisco_DNA_logo.py --no_motif_annotation.)
RANDOM_TREE = "motif_random_init_RP_TSS"
PANELS = [
    dict(panel="A", gene="RPL26A", win=PUB_WIN["RPL26A"],
         lm=("RP", 90),   ism=("gene_exp_motif_test_RP", 4, 10),
         random=(RANDOM_TREE, "gene_exp_motif_test_RP", 4, 10)),
    dict(panel="B", gene="FUN12",  win=PUB_WIN["FUN12"],
         lm=("TSS", 39),  ism=("gene_exp_motif_test_TSS", 0, 39),
         random=(RANDOM_TREE, "gene_exp_motif_test_TSS", 0, 39)),
    dict(panel="C", gene="KRE33",  win=PUB_WIN["KRE33"],
         lm=("TSS", 5134), ism=("gene_exp_motif_test_RRB_targets", 11, 3),
         random=(RANDOM_TREE, "gene_exp_motif_test_TSS_select", 0, 5)),
    dict(panel="E", gene="DTD1",   win=PUB_WIN["DTD1"],
         lm=None,         ism=("gene_exp_motif_test_SS", 22, 0), random=None),
    dict(panel="F", gene="MMS2",   win=PUB_WIN["MMS2"],
         lm=("TSS", 2175), ism=("gene_exp_motif_test_TSS", 33, 31),
         random=(RANDOM_TREE, "gene_exp_motif_test_MMS2_panel", 0, 0)),
    dict(panel="G", gene="HOP2",   win=PUB_WIN["HOP2"],
         lm=None,         ism=("gene_exp_motif_test_SS", 80, 0), random=None),
]
SHORKIE_TREE = "motif_shorkie_RP_TSS"
# Panel H curated TF order (figure label -> meme motif name / cached asset).
FIG4H_TFS = [
    ("Rap1", "RAP1"), ("Fhl1", "FHL1"), ("Sfp1.1", "SFP1"), ("TATA Box", "@TATATA"),
    ("Reb1", "REB1"), ("Abf1", "ABF1"), ("Tbf1.1", "TBF1"), ("Cbf1", "CBF1"),
    ("Ume6.2", "UME6"), ("Dot6p", "DOT6"), ("PAC motif (Dot6)", "@PAC_motif"),
    ("RRPE motif (Stb3)", "@TGAAAAATTTT"),
]

# ======================================================================================
# Logo rendering (custom DejaVu letters; A=green C=blue G=orange T=red)
# ======================================================================================
_FP = mpl.font_manager.FontProperties(family="DejaVu Sans", weight="bold")
_GLYPH = {
    "T": TextPath((-0.305, 0), "T", size=1, prop=_FP),
    "G": TextPath((-0.384, 0), "G", size=1, prop=_FP),
    "A": TextPath((-0.350, 0), "A", size=1, prop=_FP),
    "C": TextPath((-0.366, 0), "C", size=1, prop=_FP),
}
_COL = {"A": "green", "C": "blue", "G": "orange", "T": "red"}
_NT = ["A", "C", "G", "T"]


def _letter(ax, nt, x, y, yscale):
    t = (mpl.transforms.Affine2D().scale(1.0, yscale).translate(x, y) + ax.transData)
    ax.add_patch(PathPatch(_GLYPH[nt], lw=0, fc=_COL[nt], transform=t))


def draw_logo(ax, v, x0=0.0, ymin=None, ymax=None):
    """Draw a (L,4) attribution matrix as a stacked letter logo on `ax`, with the
    first column at genomic coordinate x0. Letters stacked by |magnitude|."""
    L = v.shape[0]
    for j in range(L):
        col = v[j]
        order = np.argsort(-np.abs(col))
        pos = neg = 0.0
        for i in order:
            s = float(col[i])
            if s >= 0:
                _letter(ax, _NT[i], x0 + j + 0.5, pos, s); pos += s
            else:
                _letter(ax, _NT[i], x0 + j + 0.5, neg, s); neg += s
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(x0, x0 + L)
    if ymin is None:
        a = float(np.max(np.abs(v))) if v.size else 1.0
        lo, hi = float(v.min()), float(v.max())
        ymin, ymax = lo - 0.08 * a, hi + 0.08 * a
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])
    return ymin, ymax


def draw_pwm_logo(ax, arr):
    """Draw a PWM/CWM (w,4) as a stacked letter logo (positive up, negative down),
    columns at 0..w. Used for DB motifs (IC-weighted) and modisco reconstruction."""
    return draw_logo(ax, np.asarray(arr, dtype=float), x0=0.0)


# ======================================================================================
# Shorkie ISM saliency  (logSED, T0-avg, mean-centre, project on reference)
# ======================================================================================
def ism_saliency(tree, sub, part, idx):
    """Return (v[L,4], chrom, start, end) or None if the part is absent on disk."""
    f = ISM_ROOT / tree / sub / "f0c0" / f"part{part}" / "scores.h5"
    if not f.exists():
        return None
    T0 = t0_tracks()
    with h5py.File(f, "r") as h:
        chrom = h["chr"][idx].decode(); start = int(h["start"][idx]); end = int(h["end"][idx])
        pwm = h["logSED"][idx, :, :, T0].mean(axis=-1)          # (L,4)
    pwm = pwm - pwm.mean(axis=-1, keepdims=True)
    s1 = make_seq_1hot(_fa, to_fa(chrom), start, end, end - start).astype("float32")
    return pwm * s1, chrom, start, end


def localization(v):
    """Scale-invariant peak/median of per-site |saliency| (a localized motif > diffuse noise)."""
    persite = np.abs(v).sum(axis=1)
    med = float(np.median(persite))
    return float(persite.max() / med) if med > 0 else float("inf")


def slice_to_window(v, data_start, pub_start, pub_end):
    """Slice a per-base saliency array `v` (L,4) — which covers genomic [data_start,
    data_start+L) — down to the published window [pub_start, pub_end).

    Returns (v_sub, x0, covered_frac):
      v_sub        the rows of v inside the window (may be empty / partial),
      x0           genomic coordinate of v_sub[0] (draw the logo at this x),
      covered_frac fraction of the published window the data actually covers
                   (1.0 = fully covered; <1.0 for the offset FUN12/MMS2 windows; 0.0 = no overlap).
    The caller sets ax.set_xlim(pub_start, pub_end) so a partial logo sits in the covered
    sub-range with the uncovered flank left blank — rows still align on genomic coordinate."""
    L = int(v.shape[0])
    data_end = data_start + L
    lo = max(int(pub_start), int(data_start))
    hi = min(int(pub_end), int(data_end))
    pub_len = int(pub_end) - int(pub_start)
    if hi <= lo or pub_len <= 0:
        return v[:0], int(pub_start), 0.0
    return v[lo - data_start: hi - data_start], lo, (hi - lo) / pub_len


# ======================================================================================
# Shorkie LM saliency  (preds.npz x_pred transform, 450up/50dn TSS window)
# ======================================================================================
@lru_cache(maxsize=4)
def _lm_bed(lm_set):
    bed = BED_DIR / ("RP_windows.bed" if lm_set == "RP" else "TSS_windows_ex_chrmt.bed")
    return pd.read_csv(bed, sep="\t", names=["contig", "start", "end", "gene", "species", "strand"])


def _lm_row_cache(lm_set, row):
    """Per-(set,row) cache of the single LM x_true/x_pred row (avoids re-loading the 740MB
    TSS preds.npz). First call loads the full preds.npz (run on a compute node); later
    calls read the tiny cache file."""
    cache = RECHECK / f"lm_cache_{lm_set}_{row}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["x_true_row"], z["x_pred_row"]
    d = LM_ROOT / f"lm_saccharomycetales_gtf_{LM_ARCH}" / f"eval_{lm_set}"
    z = np.load(d / "preds.npz", allow_pickle=True)
    xt = np.asarray(z["x_true"][row]); xp = np.asarray(z["x_pred"][row])
    np.savez(cache, x_true_row=xt, x_pred_row=xp)
    return xt, xp


def lm_saliency(lm_set, row, up=450, dn=50, win=None):
    """Return (v[Lwin,4], chrom, region_start, region_end) for the LM masked-prediction
    IC logo, matching 2_modisco_DNA_logo.py (xp*log(xp/mean) over the full 16,384 window).

    If `win=(pub_start, pub_end)` is given, the logo is sliced to that exact genomic
    window (by coordinate within the 16,384 bp prediction). Otherwise it falls back to the
    TSS-anchored 450up/50dn window. The explicit-window form is required for genes whose
    published window is not the gene-midpoint 450/50 (e.g. MMS2 chrVII:346,669-347,169)."""
    x_true_row, x_pred_row = _lm_row_cache(lm_set, row)
    bed = _lm_bed(lm_set).iloc[row]
    contig, bstart, bend, strand = bed["contig"], int(bed["start"]), int(bed["end"]), bed["strand"]
    xp = x_pred_row.astype("float64") + 1e-4                    # (16384,4)
    mean_pred = xp.mean(axis=0, keepdims=True)                  # genome-window base comp
    xp = xp * np.log(xp / mean_pred)                            # IC-like transform
    if win is not None:
        rs, re_ = int(win[1]), int(win[2])
    elif strand == "+":
        rs, re_ = (bstart + bend) // 2 - up, (bstart + bend) // 2 + dn
    else:
        rs, re_ = (bstart + bend) // 2 - dn, (bstart + bend) // 2 + up
    ls, le = rs - bstart, re_ - bstart
    return xp[ls:le].astype("float32"), contig, rs, re_


# ======================================================================================
# Database motif PWMs  (merged_meme_high_conf.meme  ->  IC-weighted)
# ======================================================================================
def read_meme(filename):
    motifs, motif, width, i = {}, None, None, 0
    pwm = None
    with open(filename) as fh:
        for line in fh:
            if motif is None:
                if line[:5] == "MOTIF":
                    motif = line.split()[1]
            elif width is None:
                if line[:6] == "letter":
                    width = int(line.split()[5]); pwm = np.zeros((width, 4)); i = 0
            elif i < width:
                pwm[i] = list(map(float, line.split()[:4])); i += 1
            else:
                motifs[motif] = pwm; motif, width, i = None, None, 0
        if motif is not None and pwm is not None:
            motifs[motif] = pwm
    return motifs


def per_position_ic(ppm, bg=np.array([.25, .25, .25, .25]), ps=0.001):
    return np.sum((np.log((ppm + ps) / (1 + ps * 4)) / np.log(2)) * ppm
                  - (np.log(bg) * bg / np.log(2))[None, :], axis=1)


@lru_cache(maxsize=1)
def _meme_db():
    return read_meme(str(MEME_HIGH_CONF))


def _trim_ic(arr, frac=0.15, pad=1):
    """Trim low-information flanks of an IC-weighted logo array (w,4) to its core."""
    ic = arr.sum(axis=1)
    if ic.max() <= 0:
        return arr
    keep = np.where(ic >= frac * ic.max())[0]
    if len(keep) == 0:
        return arr
    s, e = max(keep.min() - pad, 0), min(keep.max() + pad + 1, len(ic))
    return arr[s:e]


def db_motif_ic(name, trim=True):
    """IC-weighted DB motif logo array (w,4). `name` is a meme motif id, or '@asset'
    to use a synthesized consensus PWM (PAC/RRPE/TATA fallbacks)."""
    if name.startswith("@"):
        return _cached_asset_pwm(name[1:])
    db = _meme_db()
    if name not in db:
        # prefer the shortest informative variant (e.g. SFP1 -> SFP1.1)
        cand = [k for k in db if k.upper().startswith(name.upper())]
        if not cand:
            return None
        name = sorted(cand)[0]
    ppm = db[name]
    arr = ppm * per_position_ic(ppm)[:, None]
    return _trim_ic(arr) if trim else arr


# canonical promoter TFs scanned for the "Reference DB" row of panels A/B/C
REF_DB_TFS = ["RAP1", "FHL1", "SFP1", "REB1", "ABF1", "TBF1", "CBF1", "UME6", "DOT6", "STB3"]


def scan_db_motifs(chrom, win_start, win_end, tf_names=None, frac=0.80, max_hits=4, force=False):
    """FIMO-like scan: place the best log-odds match of each candidate DB motif in the
    window. Returns [{tf, gstart, gend, score, strand}] for the strongest hits — this is
    the published 'Reference DB' track (database motif matches in the promoter). With
    force=True, returns the best hit for every named TF regardless of `frac` (used to
    place the published author-curated TF annotations)."""
    if tf_names is None:
        tf_names = REF_DB_TFS
    if force:
        frac = -1e9
    seq = _fa.fetch(to_fa(chrom), win_start, win_end).upper()
    n = len(seq)
    oh = np.zeros((n, 4))
    for i, ch in enumerate(seq):
        if ch in _NT:
            oh[i, _NT.index(ch)] = 1.0
    db = _meme_db()
    out = []
    for tf in tf_names:
        name = tf if tf in db else next((k for k in sorted(db) if k.upper().startswith(tf.upper())), None)
        if name is None:
            continue
        lo = np.log2((db[name] + 1e-3) / 0.25)
        w = lo.shape[0]
        if w > n:
            continue
        perfect = float(lo.max(axis=1).sum())
        best = (-1e18, 0, "+")
        for strand, mat in (("+", lo), ("-", lo[::-1, ::-1])):
            for s in range(0, n - w + 1):
                sc = float((oh[s:s + w] * mat).sum())
                if sc > best[0]:
                    best = (sc, s, strand)
        sc, s, strand = best
        if perfect > 0 and sc >= frac * perfect:
            out.append(dict(tf=name.split(".")[0], gstart=win_start + s, gend=win_start + s + w,
                            score=sc / perfect, strand=strand))
    out.sort(key=lambda r: -r["score"])
    return out[:max_hits]


def _cached_asset_pwm(token):
    """Synthesize a simple IC logo for TATA / PAC / RRPE fallbacks from a consensus string."""
    consensus = {"TATATA": "TATATAA", "PAC_motif": "GCGATGAG", "TGAAAAATTTT": "TGAAAAATTTT"}
    s = consensus.get(token, token)
    arr = np.zeros((len(s), 4))
    for j, ch in enumerate(s):
        if ch in _NT:
            arr[j, _NT.index(ch)] = 2.0   # ~2 bits, fully conserved
    return arr


# ======================================================================================
# MoDISco reconstruction patterns (RP)
# ======================================================================================
def modisco_h5(sub="gene_exp_motif_test_RP"):
    return MODISCO_ROOT / sub / "f0c0" / "logSED" / "modisco_results_10000_500.h5"


def load_modisco_patterns(sub="gene_exp_motif_test_RP"):
    """Return dict tag -> contrib_scores CWM (w,4) for pos+neg patterns."""
    out = {}
    with h5py.File(modisco_h5(sub), "r") as f:
        for grp in ["pos_patterns", "neg_patterns"]:
            if grp not in f:
                continue
            for pn in sorted(f[grp].keys(), key=lambda x: int(x.split("_")[-1])):
                out[f"{grp}.{pn}"] = np.array(f[grp][pn]["contrib_scores"][:])
    return out


def trim_cwm(cwm, thr=0.3, pad=4):
    sc = np.sum(np.abs(cwm), axis=1)
    if sc.max() == 0:
        return cwm
    w = np.where(sc >= sc.max() * thr)[0]
    if len(w) == 0:
        return cwm
    return cwm[max(w.min() - pad, 0): min(w.max() + pad + 1, len(sc))]


# ======================================================================================
# Gene model from GTF  (strand-aware exons + intron arrows over a genomic window)
# ======================================================================================
@lru_cache(maxsize=1)
def _gtf_df():
    cols = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attr"]
    return pd.read_csv(GTF, sep="\t", comment="#", names=cols, usecols=range(9),
                       dtype={"start": int, "end": int})


def gene_features(gene):
    """Return dict with gene/exons/CDS/introns (genomic coords, 1-based GTF) for `gene`."""
    g = _gtf_df()
    sub = g[g["attr"].str.contains(f'gene_name "{gene}"', regex=False, na=False)]
    if sub.empty:
        return None
    gr = sub[sub["type"] == "gene"].iloc[0]
    exons = sorted((int(r.start), int(r.end)) for _, r in sub[sub["type"] == "exon"].iterrows())
    cds = sorted((int(r.start), int(r.end)) for _, r in sub[sub["type"] == "CDS"].iterrows())
    introns = [(exons[i][1] + 1, exons[i + 1][0] - 1) for i in range(len(exons) - 1)]
    return dict(chrom="chr" + str(gr.seqid), start=int(gr.start), end=int(gr.end),
                strand=gr.strand, exons=exons, cds=cds, introns=introns)


def draw_gene_model(ax, gene, win_start, win_end, color="#4169E1"):
    """Draw a strand-aware IGV-style gene model on `ax` over [win_start, win_end] (genomic)."""
    gf = gene_features(gene)
    ax.set_xlim(win_start, win_end); ax.set_ylim(-1, 1); ax.set_yticks([])
    if gf is None:
        return gf
    y = 0.0
    # intron line across the visible gene span
    gs, ge = max(gf["start"], win_start), min(gf["end"], win_end)
    if ge > gs:
        ax.plot([gs, ge], [y, y], color=color, lw=1.0, zorder=1)
        # strand arrows along intron
        n = max(1, int((ge - gs) / 60))
        for k in range(1, n):
            xa = gs + (ge - gs) * k / n
            ax.annotate("", xy=(xa + (8 if gf["strand"] == "+" else -8), y), xytext=(xa, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))
    for (es, ee) in gf["exons"]:
        ax.add_patch(Rectangle((es, y - 0.45), ee - es, 0.9, fc=color, ec=color, zorder=2))
    ax.text((gs + ge) / 2, y + 0.6, gene, ha="center", va="bottom", fontsize=8, style="italic")
    return gf


def splice_annotations(gene):
    """Return list of (label, genomic_pos) for Start Codon / 5' donor / Branch / 3' acceptor / Stop."""
    gf = gene_features(gene)
    if gf is None:
        return []
    ann = []
    plus = gf["strand"] == "+"
    cds = gf["cds"]
    if cds:
        start_codon = cds[0][0] if plus else cds[-1][1]
        stop_codon = cds[-1][1] if plus else cds[0][0]
        ann.append(("Start Codon", start_codon))
        ann.append(("Stop Codon (TGA)", stop_codon))
    for (i_s, i_e) in gf["introns"]:
        donor = i_s if plus else i_e
        acceptor = i_e if plus else i_s
        branch = (acceptor - 30) if plus else (acceptor + 30)
        ann.append(("5' splice site\n(donor site)", donor))
        ann.append(("Branch point", branch))
        ann.append(("3' Splice site\n(acceptor site)", acceptor))
    return ann


# ======================================================================================
# Clean per-gene ISM panel renderer (PANELS-driven; no boxes/text — matches the original
# 2_modisco_DNA_logo.py --no_motif_annotation, at the published ~33:1 per-row aspect)
# ======================================================================================
# Each logo row is an identical physical box (≈33:1 wide/thin, the original figsize=(100,3)
# ratio, scaled); the gene-model track is a short row beneath. bp/inch varies per gene
# (windows are 197/500/776 bp) — same convention as the published.
BOX_W, BOX_H = 20.0, 0.60                 # logo row (≈33:1)
GENE_H = 0.34                             # gene-model track
PANEL_LEFT, PANEL_RIGHT, PANEL_TOP, PANEL_BOT = 2.6, 0.35, 0.55, 0.5
ROW_GAP, GENE_GAP = 0.14, 0.22


def panel_rows(spec):
    """Return the present model rows for a gene as dicts, each saliency vector sliced to the
    published window. Rows: Shorkie LM / Shorkie ISM / Shorkie Random-Init ISM (where on disk)."""
    chrom, ps, pe = spec["win"]
    out = []
    if spec.get("lm"):
        v, _, rs, _ = lm_saliency(*spec["lm"], win=spec["win"])
        out.append(dict(label="Shorkie LM", v=v, x0=rs, cov=1.0, off=0, loc=localization(v)))
    sub, part, idx = spec["ism"]
    ism = ism_saliency(SHORKIE_TREE, sub, part, idx)
    if ism is not None:
        v, _, ds, _ = ism
        vs, x0, cov = slice_to_window(v, ds, ps, pe)
        out.append(dict(label="Shorkie ISM", v=vs, x0=x0, cov=cov, off=ds - ps, loc=localization(v)))
    if spec.get("random"):
        rnd = ism_saliency(*spec["random"])
        if rnd is not None:
            v, _, ds, _ = rnd
            vs, x0, cov = slice_to_window(v, ds, ps, pe)
            out.append(dict(label="Shorkie Random-Init ISM", v=vs, x0=x0, cov=cov,
                            off=ds - ps, loc=localization(v)))
    return out


def render_ism_panel(spec, dpi=150):
    """Render one clean per-gene ISM panel (stacked logo rows + gene-model track) to
    reproduced/Figure_4{panel}_reproduced.png. No red/purple boxes, TF/splice text, TSS
    dividers, '450/50' labels or Reference-DB insets. Returns a metrics dict."""
    panel, gene = spec["panel"], spec["gene"]
    chrom, ps, pe = spec["win"]
    rows = panel_rows(spec)
    n = len(rows)
    fig_w = PANEL_LEFT + BOX_W + PANEL_RIGHT
    fig_h = PANEL_TOP + n * BOX_H + (n - 1) * ROW_GAP + GENE_GAP + GENE_H + PANEL_BOT
    fig = plt.figure(figsize=(fig_w, fig_h))

    y = fig_h - PANEL_TOP
    for r in rows:
        y -= BOX_H
        ax = fig.add_axes([PANEL_LEFT / fig_w, y / fig_h, BOX_W / fig_w, BOX_H / fig_h])
        if r["v"].shape[0]:
            draw_logo(ax, r["v"], x0=r["x0"])
        else:
            ax.axhline(0, color="black", lw=0.8); ax.set_ylim(-1, 1)
        ax.set_xlim(ps, pe); ax.set_yticks([]); ax.set_xticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        lbl = r["label"] + ("" if r["cov"] >= 0.995 else f"\n(covered {r['cov']:.0%})")
        ax.text(-0.010, 0.5, lbl, transform=ax.transAxes, ha="right", va="center", fontsize=8)
        y -= ROW_GAP
    # gene-model track (kept — strand-aware exons/introns + gene name; not a "box/text annotation")
    y -= (GENE_GAP - ROW_GAP) + GENE_H
    axg = fig.add_axes([PANEL_LEFT / fig_w, y / fig_h, BOX_W / fig_w, GENE_H / fig_h])
    draw_gene_model(axg, gene, ps, pe)
    for s in ("top", "right", "left"):
        axg.spines[s].set_visible(False)
    ticks = np.linspace(ps, pe, 4)
    axg.set_xticks(ticks); axg.set_xticklabels([f"{int(t):,}" for t in ticks], fontsize=7)
    axg.text(-0.010, 0.5, "gene model", transform=axg.transAxes, ha="right", va="center", fontsize=8)
    fig.text((PANEL_LEFT + BOX_W / 2) / fig_w, 1 - 0.16 / fig_h,
             f"4{panel}  —  {gene}  ({chrom}:{ps:,}-{pe:,})", ha="center", va="top", fontsize=11)

    out = RD / f"Figure_4{panel}_reproduced.png"
    fig.savefig(out, dpi=dpi)                                  # no bbox_inches='tight' -> exact box
    plt.close(fig)
    by = {r["label"]: r for r in rows}
    return dict(panel=f"4{panel}", gene=gene, chrom=chrom, win_start=ps, win_end=pe,
                n_rows=n, has_LM=int("Shorkie LM" in by), has_ISM=int("Shorkie ISM" in by),
                has_Random=int("Shorkie Random-Init ISM" in by),
                ism_offset=by.get("Shorkie ISM", {}).get("off", "n/a"),
                ism_covered=round(by.get("Shorkie ISM", {}).get("cov", float("nan")), 4),
                random_covered=(round(by["Shorkie Random-Init ISM"]["cov"], 4)
                                if "Shorkie Random-Init ISM" in by else "n/a"),
                loc_LM=round(by["Shorkie LM"]["loc"], 2) if "Shorkie LM" in by else "n/a",
                loc_ISM=round(by["Shorkie ISM"]["loc"], 2) if "Shorkie ISM" in by else "n/a",
                loc_Random=(round(by["Shorkie Random-Init ISM"]["loc"], 2)
                            if "Shorkie Random-Init ISM" in by else "n/a"),
                aspect_w_over_h=round(BOX_W / BOX_H, 2), out=str(out))
