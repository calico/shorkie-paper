#!/usr/bin/env python3
"""Rebuild reproduced/verify_fig06.csv against the PUBLISHED FIGURE targets.

Recomputes every Figure-6 panel from cached data (no GPU) and checks the reproduced
Pearson/Spearman for both the Shorkie (blue) and DREAM-RNN (green) subpanels of D-I,
plus the 6B/6C AUROC/AUPRC > 0.95 aggregate claim, against the values printed on the
published figure. Single source of truth for the verification table.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpra_common as mc
import build_panel_I as PI
import build_panels_BC as BC
from shorkie import config

sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa

config.load()
REPRO = config.repo_root() / "reproduction" / "figure_06" / "reproduced"

# Published printed values (Pearson, Spearman) — read directly off Figure_6.pdf.
PUB = {
    "6D": dict(shk=(0.695, 0.718), dream=(0.891, 0.891)),
    "6E": dict(shk=(0.744, 0.758), dream=(0.981, 0.983)),
    "6F": dict(shk=(0.539, 0.420), dream=(0.866, 0.709)),
    "6G": dict(shk=(0.819, 0.745), dream=(0.983, 0.970)),
    "6H": dict(shk=(0.561, 0.546), dream=(0.943, 0.944)),
}
SINGLE = {"6D": "yeast_seqs", "6E": "all_random_seqs"}
DUAL = {"6F": "all_SNVs_seqs", "6G": "motif_perturbation", "6H": "motif_tiling_seqs"}


def main():
    gt = mc.load_ground_truth()
    dream = mc.load_dream()
    checks = []

    # 6B/6C — AUROC/AUPRC > 0.95 (quantile aggregates)
    metrics = {g: BC.gene_site_metrics(g) for grp in BC.QUANTILES.values() for g in grp}
    metrics = {g: m for g, m in metrics.items() if m}
    auroc = float(np.mean([metrics[g][c][0] for g in metrics for c in metrics[g]]))
    auprc = float(np.mean([metrics[g][c][1] for g in metrics for c in metrics[g]]))
    checks.append(Check("6B", "mean_AUROC_high_vs_low(>0.95)", 0.95, round(auroc, 4), mode="ge"))
    checks.append(Check("6C", "mean_AUPRC_high_vs_low(>0.95)", 0.95, round(auprc, 4), mode="ge"))

    # 6D/6E single-sequence: Shorkie blue + DREAM green, Pearson & Spearman
    for panel, seq in SINGLE.items():
        _, _, sp, ss, _, _ = mc.shorkie_single(seq, gt)
        _, _, dp, ds, _ = mc.dream_single(seq, gt, dream)
        checks.append(Check(panel, "Shorkie_Pearson", PUB[panel]["shk"][0], round(sp, 3), atol=0.01, rtol=0.0))
        checks.append(Check(panel, "Shorkie_Spearman", PUB[panel]["shk"][1], round(ss, 3), atol=0.01, rtol=0.0))
        checks.append(Check(panel, "DREAM_Pearson", PUB[panel]["dream"][0], round(dp, 3), atol=0.01, rtol=0.0))
        checks.append(Check(panel, "DREAM_Spearman", PUB[panel]["dream"][1], round(ds, 3), atol=0.01, rtol=0.0))

    # 6F/6G/6H dual-sequence (Alt-Ref): Shorkie blue + DREAM green
    for panel, seq in DUAL.items():
        _, _, sp, ss, _, _ = mc.shorkie_dual(seq, gt)
        _, _, dp, ds, _ = mc.dream_dual(seq, gt, dream)
        checks.append(Check(panel, "Shorkie_Pearson", PUB[panel]["shk"][0], round(sp, 3), atol=0.01, rtol=0.0))
        checks.append(Check(panel, "Shorkie_Spearman", PUB[panel]["shk"][1], round(ss, 3), atol=0.015, rtol=0.0))
        checks.append(Check(panel, "DREAM_Pearson", PUB[panel]["dream"][0], round(dp, 3), atol=0.01, rtol=0.0))
        checks.append(Check(panel, "DREAM_Spearman", PUB[panel]["dream"][1], round(ds, 3), atol=0.015, rtol=0.0))

    # 6I density correlations
    sh = PI.process(PI.load_shorkie())
    dr = PI.process(PI.load_dream())
    checks.append(Check("6I", "Shorkie_density_Pearson_r", 0.895, round(sh["r"], 3), atol=0.01, rtol=0.0))
    checks.append(Check("6I", "Shorkie_density_Spearman_rho", 0.837, round(sh["rho"], 3), atol=0.01, rtol=0.0))
    checks.append(Check("6I", "DREAM_density_Pearson_r", 0.249, round(dr["r"], 3), atol=0.01, rtol=0.0))
    checks.append(Check("6I", "DREAM_density_Spearman_rho", 0.261, round(dr["rho"], 3), atol=0.01, rtol=0.0))

    print(summary(checks))
    write_verdicts(checks, REPRO / "verify_fig06.csv")
    npass = sum(c.verdict == "PASS" for c in checks)
    print(f"\n{npass}/{len(checks)} checks PASS")


if __name__ == "__main__":
    main()
