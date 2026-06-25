#!/usr/bin/env python3
"""Figure 2 — assemble verify_fig02.csv from the per-panel recheck artifacts.

Scope after the panel-2B/2C revision: 2A (SMT3 logos), 2D (TSS enrichment, the
published 6-panel grid), 2E (t-SNE silhouette). 2B (GPU iterative masking) was
removed; 2C (motif-conservation grid) is provided as upstream scripts and skipped in
the reproduction, so neither is verified here.

Reads:
  recheck/fig2A_consistency.csv     (2A Shorkie consistency + motifs)
  recheck/fig2D_enrichment.csv      (2D per-motif TSS enrichment, 6 panels)
  recheck/fig2E_separation.csv      (2E t-SNE silhouette)

Writes reproduced/verify_fig02.csv (canonical) + recheck/recheck_checks_fig02.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

from shorkie import config

F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa: E402


def main():
    checks = []

    # ---- 2A ---- (three rows registered to the same SMT3_seq[690:800] window)
    a = dict(zip(*[pd.read_csv(RECHECK / "fig2A_consistency.csv")[c] for c in ("metric", "value")]))
    checks.append(Check("2A", "three_rows_same_width", 1.0,
                        float(a["three_rows_same_width"]), mode="ge"))
    checks.append(Check("2A", "SpeciesLM_argmax_vs_genome_agree(>=0.8)", 0.8,
                        float(a["SpeciesLM_argmax_vs_genome_agree"]), mode="ge"))
    checks.append(Check("2A", "SpeciesLM_vs_Shorkie_registered_pwm_corr(>=0.5)", 0.5,
                        float(a["SpeciesLM_vs_Shorkie_registered_pwm_corr"]), mode="ge"))
    checks.append(Check("2A", "Shorkie_unmasked_vs_iterative_pwm_corr(>=0.4)", 0.4,
                        float(a["Shorkie_unmasked_vs_iter_pwm_corr"]), mode="ge"))
    checks.append(Check("2A", "Cbf1_Ebox_in_window", 1.0,
                        float(a["Cbf1_Ebox(CACGTG)_in_window"]), mode="ge"))
    checks.append(Check("2A", "polydAdT_in_window", 1.0,
                        float(a["polydAdT_run>=6_in_window"]), mode="ge"))

    # ---- 2D ---- (published 6-panel grid: each enriched near the TSS + count matches)
    d = pd.read_csv(RECHECK / "fig2D_enrichment.csv")
    for _, r in d.iterrows():
        checks.append(Check("2D", f"{r['panel']}_enriched_near_TSS", 1.0,
                            1.0 if bool(r["enriched_near_tss"]) else 0.0, mode="ge"))
    for _, r in d.iterrows():
        # count vs published within 1% (MIG3.4 is 2216 vs 2218 in the released CSV — a
        # 2-row / 0.09% residual; the other five match the published n exactly).
        tol = 0.01 * float(r["published_n"])
        ok = abs(float(r["n_true"]) - float(r["published_n"])) <= tol
        checks.append(Check("2D", f"{r['panel']}_n≈published({int(r['published_n'])},±1%)",
                            1.0, 1.0 if ok else 0.0, mode="ge"))

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
