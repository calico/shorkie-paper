#!/usr/bin/env python3
"""Figure 2 deep-recheck — Step 6: assemble the tightened verify_fig02.csv from the
per-panel recheck artifacts, replacing the original thin 3-qualitative-check CSV.

Reads:
  recheck/fig2C_presence_grid.csv   (2C conservation grid)
  recheck/fig2D_enrichment.csv      (2D per-TF TSS enrichment)
  recheck/fig2E_separation.csv      (2E t-SNE silhouette)
  recheck/fig2A_consistency.csv     (2A Shorkie consistency + motifs)
  reproduced/iterative_smt3/preds_smt3_iterative.npz (2B coverage)

Writes reproduced/verify_fig02.csv (canonical) + recheck/recheck_checks_fig02.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shorkie import config

F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa: E402

TIERS = ["S. cerevisiae R64", "4 S. cerevisiae strains", "5 Saccharomycetales",
         "4 Ascomycota", "4 Orbiliales", "4 Schizosaccharomycetales"]


def main():
    checks = []

    # ---- 2A ----
    a = dict(zip(*[pd.read_csv(RECHECK / "fig2A_consistency.csv")[c] for c in ("metric", "value")]))
    checks.append(Check("2A", "Shorkie_unmasked_vs_iterative_pwm_corr(>=0.4)", 0.4,
                        float(a["Shorkie_unmasked_vs_iter_pwm_corr"]), mode="ge"))
    checks.append(Check("2A", "Cbf1_Ebox_in_promoter", 1.0,
                        float(a["Cbf1_Ebox(CACGTG)_in_Shorkie_promoter"]), mode="ge"))
    checks.append(Check("2A", "polydAdT_in_promoter", 1.0,
                        float(a["polydAdT_run>=8_in_Shorkie_promoter"]), mode="ge"))

    # ---- 2B ----
    iz = np.load(RD / "iterative_smt3" / "preds_smt3_iterative.npz", allow_pickle=True)
    assign = np.asarray(iz["iter_assignment"])
    covered = float((assign >= 0).all())
    n_iters = int(assign[0].max()) + 1
    checks.append(Check("2B", "iterative_all_positions_covered", 1.0, covered, mode="ge"))
    checks.append(Check("2B", "n_iterations(==ceil(1/0.15)=7)", 7.0, float(n_iters), atol=0.0, rtol=0.0))

    # ---- 2C ----
    pres = pd.read_csv(RECHECK / "fig2C_presence_grid.csv", index_col=0)
    counts = pres.sum(axis=0)
    checks.append(Check("2C", "TF_motifs_recovered_in_R64(>=8/9)", 8.0,
                        float(counts[TIERS[0]]), mode="ge"))
    checks.append(Check("2C", "conservation_decline_count(R64>Schizo)", 1.0,
                        1.0 if counts[TIERS[0]] > counts[TIERS[5]] else 0.0, mode="ge"))
    checks.append(Check("2C", "Mcm1.1_present_through_Orbiliales", 1.0,
                        float(pres.loc["Mcm1.1", TIERS[4]]), mode="ge"))
    checks.append(Check("2C", "Mcm1.1_absent_in_Schizosacc(==0)", 0.0,
                        float(pres.loc["Mcm1.1", TIERS[5]]), mode="le"))
    prom_lost = all(pres.loc[m, TIERS[k]] == 0 for m in ("Rap1.1", "Abf1.1", "Dot6") for k in (3, 4, 5))
    checks.append(Check("2C", "promoterTFs(Rap1/Abf1/Dot6)_lost_beyond_Sacc", 1.0,
                        1.0 if prom_lost else 0.0, mode="ge"))

    # ---- 2D ----
    d = pd.read_csv(RECHECK / "fig2D_enrichment.csv")
    for _, r in d[d["enriched_near_tss"].isin([True, False])].iterrows():
        checks.append(Check("2D", f"{r['panel']}_enriched_near_TSS", 1.0,
                            1.0 if bool(r["enriched_near_tss"]) else 0.0, mode="ge"))

    # ---- 2E ----
    e = dict(zip(*[pd.read_csv(RECHECK / "fig2E_separation.csv")[c] for c in ("metric", "value")]))
    checks.append(Check("2E", "feature_classes_present(==5)", 5.0, float(e["n_classes"]), atol=0.0, rtol=0.0))
    checks.append(Check("2E", "tSNE_silhouette_2d(>0)", 0.0, float(e["silhouette_2d"]), mode="gt"))
    checks.append(Check("2E", "n_points_all_16_chr(>=10000)", 10000.0, float(e["n_points"]), mode="ge"))

    print(summary(checks))
    n_pass = sum(c.verdict == "PASS" for c in checks)
    print(f"\n{n_pass}/{len(checks)} checks PASS")
    write_verdicts(checks, RD / "verify_fig02.csv")
    write_verdicts(checks, RECHECK / "recheck_checks_fig02.csv")


if __name__ == "__main__":
    main()
