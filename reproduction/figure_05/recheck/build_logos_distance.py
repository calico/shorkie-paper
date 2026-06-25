#!/usr/bin/env python3
"""Figure 5 panels A/C (MSN2 @ ATG42) and F/H (MSN4 @ TSL1): full-window ISM logo stacks
+ pairwise Euclidean-distance heatmaps.

Fixes vs the prior notebook reproduction:
  - A (MSN2) now uses the REAL ATG42 promoter (recomputed ISM at reproduced/ism_atg42/),
    not the representative surrogate; C is its distance heatmap.
  - A and F render the FULL 500 bp promoter window (the published view), not the 90 bp zoom.

Usage:  python build_logos_distance.py [--locus tsl1|atg42|both]
TSL1 uses the released MSN4 ISM (always available); ATG42 needs the GPU recompute
(panels/run_atg42_ism.sbatch) to have produced reproduced/ism_atg42/scores.h5.
"""
import sys, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig05_lib import (ISM, RD, tp_tracks, load_locus, plot_full_logos, distance_heatmap)

LOCI = {
    "tsl1": dict(
        tf="MSN4", h5=lambda: ISM/"gene_exp_motif_test_MSN4_targets"/"f0c0"/"part2"/"scores.h5", idx=7,
        win=("chrXIII", 70173, 70673),
        logo_png="Figure_5F_TSL1_logos.png", dist_png="Figure_5H_TSL1_distance.png",
        logo_title="Figure 5F (reproduced) — MSN4 @ TSL1 ISM logos over time",
        dist_title="Figure 5H (reproduced) — MSN4 @ TSL1 ISM distance",
    ),
    "atg42": dict(
        tf="MSN2", h5=lambda: RD/"ism_atg42"/"scores.h5", idx=0,
        win=("chrII", 515214, 515714),
        logo_png="Figure_5A_ATG42_logos.png", dist_png="Figure_5C_ATG42_distance.png",
        logo_title="Figure 5A (reproduced) — MSN2 @ ATG42 ISM logos over time",
        dist_title="Figure 5C (reproduced) — MSN2 @ ATG42 ISM distance",
    ),
}


def build(key):
    L = LOCI[key]
    h5 = L["h5"]()
    if not Path(h5).exists():
        print(f"[{key}] {h5} MISSING — skip "
              f"({'run panels/run_atg42_ism.sbatch first' if key=='atg42' else 'released ISM not found'})")
        return None
    print(f"[{key}] {L['tf']} @ {h5}")
    norm, proj, win, _ = load_locus(h5, L["idx"], tp_tracks(L["tf"]))
    ok = (win[0] == L["win"][0] and win[1] == L["win"][1] and win[2] == L["win"][2])
    print(f"  window {win} | matches published {L['win']}: {ok}")
    plot_full_logos(proj, win, L["logo_title"], L["logo_png"])
    D, ts = distance_heatmap(norm, L["dist_title"], L["dist_png"])
    mono = bool(all(D[0, i] <= D[0, i + 1] + 1e-6 for i in range(len(ts) - 1)))
    print(f"  T0-row distances {np.round(D[0], 3)} | monotone-from-T0 {mono}")
    return dict(window=win, window_ok=ok, t0_row=D[0].tolist(), monotone=mono, n_tp=len(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locus", choices=["tsl1", "atg42", "both"], default="both")
    a = ap.parse_args()
    keys = ["tsl1", "atg42"] if a.locus == "both" else [a.locus]
    for k in keys:
        build(k)


if __name__ == "__main__":
    main()
