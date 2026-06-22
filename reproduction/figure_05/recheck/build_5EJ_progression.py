#!/usr/bin/env python3
"""Figure 5 panels E (MSN2) and J (MSN4): the TF-Modisco motif PROGRESSION.

The PUBLISHED E/J are a curated row — a YeTFaSCo reference motif on the left, then one
small logo per ΔT(Tn-T0) showing the TF-Modisco-detected binding-site motif (the MSN2/MSN4
STRE element) emerging/strengthening with induction. (The prior notebook instead rendered a
generic "top-5 modisco patterns from the single T90-T0 diff" grid and mislabeled it 5D/5I.)

For each timepoint we load …/MSN{2,4}/T{n}/modisco_results_10000_500_diff.h5 and select the
pattern whose trimmed consensus best matches the STRE core (a run of G/C) — the analogue of
the authors' curation of the binding-site motif. We orient MSN2 to the G-strand (AGGGG, as
published E) and MSN4 to the C-strand (CCCC, as published J), reverse-complementing the logo
when the selected pattern is on the opposite strand. Where no pattern reaches the STRE
threshold the cell is marked "Not clustered" (as the published MSN2 T5 cell is).

RESIDUAL (documented): the published MSN2 panel E uses the EXTENDED timepoint series
T5,T10,T20,T40,T70,T120,T180 (the series the released code assigns to SWI4); the released MSN2
modisco only has the standard 8 timepoints (T0,5,10,15,30,45,60,90). We reproduce over the
available MSN2 timepoints. MSN4's published series (5,10,15,30,45,60,90) matches the released
data, so panel J reproduces directly.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig05_lib import MODISCO_DIFF, RD, ic, trim, consensus, max_run

DT = [5, 10, 15, 30, 45, 60, 90]          # ΔT(Tn-T0) timepoints available in the released modisco
STRE_MIN = 3                               # min G/C-run to call a STRE binding-site motif
REF = {"MSN2": ("AGGGGG", "G"), "MSN4": ("CCCCTT", "C")}  # YeTFaSCo-style reference + published orientation


def rc_df(df):
    """Reverse-complement an ACGT logo DataFrame."""
    return (df.iloc[::-1].rename(columns={"A": "T", "T": "A", "C": "G", "G": "C"})[list("ACGT")]
            .reset_index(drop=True))


def consensus_pwm(seq, p=0.90):
    """Near-deterministic PPM for a consensus string (for the reference logo)."""
    rows = []
    for ch in seq:
        r = np.full(4, (1 - p) / 3); r["ACGT".index(ch)] = p
        rows.append(r)
    return np.array(rows)


def select_motif(h5path, orient):
    """Return (ic_weighted_ppm_trimmed_oriented, consensus_str, run_len) for the STRE-matching
    pattern in a ΔT modisco h5, oriented to `orient` ('G' for MSN2, 'C' for MSN4); or None."""
    if not Path(h5path).exists():
        return None
    best = None
    with h5py.File(h5path, "r") as f:
        for grp in ("pos_patterns", "neg_patterns"):
            if grp not in f:
                continue
            for pn in f[grp].keys():
                cwm = np.array(f[grp][pn]["contrib_scores"][:])
                ppm = np.array(f[grp][pn]["sequence"][:])
                s, e = trim(cwm)
                cs = consensus(ppm[s:e])
                run = max(max_run(cs, "G"), max_run(cs, "C"))
                if best is None or run > best[0]:
                    best = (run, ppm[s:e], cs)
    if best is None or best[0] < STRE_MIN:
        return None
    run, ppm_t, cs = best
    logo = pd.DataFrame(ppm_t * ic(ppm_t)[:, None], columns=list("ACGT"))
    # orient: published MSN2 = G-strand, MSN4 = C-strand
    if max_run(cs, orient) < max_run(cs, {"G": "C", "C": "G"}[orient]):
        logo = rc_df(logo); cs = consensus(rc_df(pd.DataFrame(ppm_t, columns=list("ACGT"))).values)
    return logo, cs, run


def build(tf):
    ref_seq, orient = REF[tf]
    cells = []  # (label, logo_df_or_None, note)
    ref_logo = pd.DataFrame(consensus_pwm(ref_seq) * ic(consensus_pwm(ref_seq))[:, None], columns=list("ACGT"))
    cells.append(("YeTFaSCo\nDB motif", ref_logo, ref_seq))
    for tn in DT:
        h5 = MODISCO_DIFF / f"gene_exp_motif_test_{tf}_targets" / "f0c0" / tf / f"T{tn}" / "modisco_results_10000_500_diff.h5"
        sel = select_motif(h5, orient)
        if sel is None:
            cells.append((f"T{tn}-T0", None, "Not clustered"))
        else:
            logo, cs, run = sel
            cells.append((f"T{tn}-T0", logo, cs))
    ncol = len(cells)
    fig, axes = plt.subplots(1, ncol, figsize=(2.05 * ncol, 1.9))
    for ax, (lbl, logo, note) in zip(np.atleast_1d(axes), cells):
        if logo is None:
            ax.text(0.5, 0.5, "Not\nclustered", ha="center", va="center", fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            logomaker.Logo(logo, ax=ax, color_scheme={"A": "green", "C": "blue", "G": "orange", "T": "red"})
            ax.set_xticks([]); ax.set_yticks([])
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(lbl, fontsize=9)
    panel = "5E" if tf == "MSN2" else "5J"
    fig.suptitle(f"Figure {panel} (reproduced) — {tf} Induction: TF-Modisco-detected binding site over ΔT", fontsize=11, y=1.06)
    fig.tight_layout()
    out = RD / (f"Figure_5E_MSN2_motif_progression.png" if tf == "MSN2" else "Figure_5J_MSN4_motif_progression.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    found = [c[0] for c in cells[1:] if c[1] is not None]
    print(f"[{tf}] {panel}: STRE motif recovered at {found}  -> {out}")
    return cells


if __name__ == "__main__":
    for tf in ("MSN2", "MSN4"):
        build(tf)
