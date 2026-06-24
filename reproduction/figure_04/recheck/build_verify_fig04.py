#!/usr/bin/env python3
"""Figure 4 verification (recheck) — rebuild reproduced/verify_fig04.csv.

Validates the clean exact-window 3-model ISM panels:
  - per panel: Shorkie-ISM window matches the published window (exact for
    RPL26A/FUN12/MMS2, |off|<=1 for KRE33, full crop for DTD1/HOP2)
  - per panel: Shorkie-ISM localization >= 5x (peak/median per-site saliency)
  - 3-model panels A/B/C/F: LM + ISM + Random-Init rows all rendered, and
    Shorkie-ISM localization > Random-Init localization
  - panel F (MMS2) is a 3-row panel (LM+ISM+Random), not ISM-only
  - panel H: TF-MoDISco pattern count + TomTom-matched reconstruction TFs
  - panel D reconstruction rendered
  - clean ISM grid: png rendered, uniform box size, all rows full-coverage, localization

Reads the per-panel metric CSVs the builders wrote.
"""
import sys
from pathlib import Path
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import fig4_common as F
from compare import Check, write_verdicts, summary

LOC_MIN = 5.0
EXACT_GENES = {"RPL26A", "FUN12", "MMS2"}          # Shorkie-ISM window == published, to the base
THREE_MODEL = {"4A", "4B", "4C", "4F"}             # published 3-row panels


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    checks = []
    frames = [F.RECHECK / "fig4ABC_metrics.csv", F.RECHECK / "fig4EFG_metrics.csv"]
    metrics = pd.concat([pd.read_csv(p) for p in frames if p.exists()], ignore_index=True)

    for _, r in metrics.iterrows():
        p, gene = r["panel"], r["gene"]
        # window exactness / coverage
        if gene in EXACT_GENES:
            checks.append(Check(p, f"ism_window_exact_{gene}(off==0)", 0.0, _num(r["ism_offset"]), atol=0))
        elif gene == "KRE33":
            checks.append(Check(p, f"ism_window_{gene}(|off|<=1)", 0.0, _num(r["ism_offset"]), atol=1))
        checks.append(Check(p, f"ism_coverage_{gene}(>=0.99)", 0.99, _num(r["ism_covered"]), mode="ge"))
        # localization
        checks.append(Check(p, f"Shorkie_ISM_localization_{gene}(>={LOC_MIN}x)", LOC_MIN, _num(r["loc_ISM"]), mode="ge"))
        # 3-model panels: all rows + Shorkie>Random
        if p in THREE_MODEL:
            checks.append(Check(p, "three_model_rows(LM+ISM+Random)", 3.0, _num(r["n_rows"]), atol=0))
            checks.append(Check(p, "Random_Init_row_rendered", 1.0, _num(r["has_Random"]), atol=0))
            checks.append(Check(p, "ShorkieLM_row_rendered", 1.0, _num(r["has_LM"]), atol=0))
            checks.append(Check(p, "Shorkie_loc>Random_loc", _num(r["loc_Random"]), _num(r["loc_ISM"]), mode="gt"))

    # panel F is now a 3-model panel (was ISM-only in the prior reproduction)
    f_row = metrics[metrics["panel"] == "4F"]
    if len(f_row):
        checks.append(Check("4F", "MMS2_is_3model_panel", 3.0, _num(f_row.iloc[0]["n_rows"]), atol=0))

    # ---- panel H ----
    nmod = 0
    with h5py.File(F.modisco_h5("gene_exp_motif_test_RP"), "r") as f:
        for g in ["pos_patterns", "neg_patterns"]:
            if g in f:
                nmod += len(f[g].keys())
    checks.append(Check("4H", "n_TFMoDISco_patterns(>=6)", 6.0, float(nmod), mode="ge"))
    hp = F.RECHECK / "fig4H_tomtom_pairs.csv"
    if hp.exists():
        h = pd.read_csv(hp)
        checks.append(Check("4H", "panelH_TomTom_matched_TFs(>=8of12)", 8.0,
                            float(int(h["modisco_pattern"].notna().sum())), mode="ge"))

    # ---- panel D ----
    checks.append(Check("4D", "panelD_reconstruction_rendered", 1.0,
                        1.0 if (F.RD / "Figure_4D_reproduced.png").exists() else 0.0, atol=0))

    # ---- clean uniform ISM grid ----
    grid_csv = F.RECHECK / "fig4_ism_grid_metrics.csv"
    if grid_csv.exists():
        g = pd.read_csv(grid_csv)
        checks.append(Check("4grid", "ISM_grid_png_rendered", 1.0,
                            1.0 if (F.RD / "Figure_4_ISM_grid_reproduced.png").exists() else 0.0, atol=0))
        # rows: A=3 B=3 C=3 E=1 F=3 G=1 = 14 (3-model promoter/MMS2 panels + ISM-only DTD1/HOP2)
        checks.append(Check("4grid", "n_saliency_logos(==14)", 14.0, float(len(g)), atol=0))
        uniform = 1.0 if (g["box_w_in"].nunique() == 1 and g["box_h_in"].nunique() == 1) else 0.0
        checks.append(Check("4grid", "uniform_box_size(all_equal)", 1.0, uniform, atol=0))
        checks.append(Check("4grid", "all_rows_full_coverage(>=0.99)", 0.99, float(g["covered_frac"].min()), mode="ge"))
        ism = g[g["source"] == "Shorkie ISM"]
        checks.append(Check("4grid", f"all_ISM_localization(>={LOC_MIN}x)", LOC_MIN, float(ism["localization"].min()), mode="ge"))

    print(summary(checks))
    write_verdicts(checks, F.RD / "verify_fig04.csv")
    npass = sum(c.verdict == "PASS" for c in checks)
    print(f"\n{npass}/{len(checks)} checks PASS")
    assert npass == len(checks), "some Figure-4 checks FAILED"


if __name__ == "__main__":
    main()
