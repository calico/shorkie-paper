#!/usr/bin/env python3
"""Figure 4 verification (recheck) — rebuild reproduced/verify_fig04.csv.

Keeps the rigorous correctness checks of the original reproduction and adds the
panel-completeness checks for the recheck upgrade:
  - 6x window-overlap (each panel's ISM window covers the claimed R64 gene)
  - 6x localization >= 5x (Shorkie ISM peak/median per-site saliency)
  - Shorkie > Random_Init localization at panel A
  - n TF-MoDISco patterns >= 6
  - rows rendered for A/B/C (Shorkie LM / ISM / Random / Reference DB)  [from fig4ABC_metrics]
  - panel-H TomTom-matched reconstruction TFs >= 8 of 12               [from fig4H pairs]
  - panel-D donor + branch reconstruction rendered

Reads the per-panel metric CSVs the builders wrote, recomputes the modisco count.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import fig4_common as F
from compare import Check, write_verdicts, summary

LOC_MIN = 5.0


def main():
    checks = []
    abc = pd.read_csv(F.RECHECK / "fig4ABC_metrics.csv") if (F.RECHECK / "fig4ABC_metrics.csv").exists() else None
    efg = pd.read_csv(F.RECHECK / "fig4EFG_metrics.csv") if (F.RECHECK / "fig4EFG_metrics.csv").exists() else None
    h = pd.read_csv(F.RECHECK / "fig4H_tomtom_pairs.csv") if (F.RECHECK / "fig4H_tomtom_pairs.csv").exists() else None

    # ---- promoter panels A/B/C ----
    if abc is not None:
        for _, r in abc.iterrows():
            p = r["panel"]
            checks.append(Check(p, f"window_overlaps_{r['gene']}", 1.0, float(r["window_overlaps_gene"]), rtol=0, atol=0))
            checks.append(Check(p, f"Shorkie_ISM_localization(>={LOC_MIN}x)", LOC_MIN, float(r["loc_ISM"]), mode="ge"))
            checks.append(Check(p, "ShorkieLM_row_rendered", 1.0, 1.0 if r["loc_LM"] == r["loc_LM"] else 0.0, rtol=0, atol=0))
        # Shorkie > Random at panel A
        a = abc[abc["panel"] == "4A"].iloc[0]
        if str(a["loc_Random"]) not in ("n/a", "nan"):
            checks.append(Check("4A", "Shorkie_loc>Random_loc", float(a["loc_Random"]), float(a["loc_ISM"]), mode="gt"))

    # ---- splicing panels E/F/G ----
    if efg is not None:
        for _, r in efg.iterrows():
            checks.append(Check(r["panel"], f"window_overlaps_{r['gene']}", 1.0, float(r["window_overlaps_gene"]), rtol=0, atol=0))
            checks.append(Check(r["panel"], f"splicing_ISM_localization(>={LOC_MIN}x)", LOC_MIN, float(r["localization"]), mode="ge"))

    # ---- panel H ----
    nmod = 0
    with h5py.File(F.modisco_h5("gene_exp_motif_test_RP"), "r") as f:
        for g in ["pos_patterns", "neg_patterns"]:
            if g in f:
                nmod += len(f[g].keys())
    checks.append(Check("4H", "n_TFMoDISco_patterns(>=6)", 6.0, float(nmod), mode="ge"))
    if h is not None:
        nmatch = int(h["modisco_pattern"].notna().sum())
        checks.append(Check("4H", "panelH_TomTom_matched_TFs(>=8of12)", 8.0, float(nmatch), mode="ge"))

    # ---- panel D ----
    checks.append(Check("4D", "panelD_reconstruction_rendered", 1.0,
                        1.0 if (F.RD / "Figure_4D_reproduced.png").exists() else 0.0, rtol=0, atol=0))

    print(summary(checks))
    write_verdicts(checks, F.RD / "verify_fig04.csv")
    npass = sum(c.verdict == "PASS" for c in checks)
    print(f"\n{npass}/{len(checks)} checks PASS")


if __name__ == "__main__":
    main()
