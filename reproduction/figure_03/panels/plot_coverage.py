#!/usr/bin/env python3
"""Render Figure 3 panels H/I/J from reproduced coverage npz (run_coverage.py output).

Layout mirrors the published panels: one column per locus (RPL7A, RPS16B-RPL13A,
EFM5), three stacked filled coverage tracks per column — Experimental (observed
bigwig, purple), Fine-tuned model = Shorkie (red), Scratch-trained = Random_Init
(blue). Per-track Pearson R vs observed is annotated. Saves:
  reproduced/Figure_3HIJ_coverage.png
and appends coverage correlation rows to recheck/recheck_checks_coverage.csv.
"""
import sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../reproduction
from common.compare import Check, write_verdicts  # noqa
from shorkie import config
config.load()
REPRO = config.repo_root() / "reproduction" / "figure_03"
COV = REPRO / "reproduced" / "coverage"

LOCI = [
    ("rpl7a",         "RPL7A\nchrVII:362,180-366,023 (fold 3)"),
    ("rps16b_rpl13a", "RPS16B-RPL13A\nchrIV:305,657-310,505 (fold 3)"),
    ("efm5",          "EFM5\nchrVII:495,374-499,965 (fold 6)"),
]
ROWS = [("cov_obs", "Experimental (observed)", "tab:purple"),
        ("cov_self", "Fine-tuned (Shorkie)", "tab:red"),
        ("cov_ri", "Scratch-trained (Random_Init)", "tab:blue")]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(15, 7.5), sharex="col")
    checks = []
    for j, (name, title) in enumerate(LOCI):
        d = np.load(COV / f"{name}.npz", allow_pickle=True)
        stride = int(d["stride"]); start = int(d["seq_out_start"])
        n = len(d["cov_obs"]); x = (start + np.arange(n) * stride) / 1000.0  # kb
        obs = d["cov_obs"]
        for i, (key, lbl, col) in enumerate(ROWS):
            ax = axes[i, j]
            y = d[key]
            ax.fill_between(x, 0, y, color=col, alpha=0.75, linewidth=0)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if key != "cov_obs":
                r = float(np.corrcoef(y, obs)[0, 1])
                ax.text(0.98, 0.92, f"R={r:.3f}", ha="right", va="top",
                        transform=ax.transAxes, fontsize=9,
                        bbox=dict(boxstyle="round", fc="white", ec=col, alpha=0.8))
                panel = {"rpl7a": "3H", "rps16b_rpl13a": "3I", "efm5": "3J"}[name]
                model = "Shorkie" if key == "cov_self" else "Random_Init"
                checks.append(Check(panel, f"coverage_Pearson_R[{model},obs](>0.5)", 0.5, round(r, 4), mode="ge"))
            if j == 0:
                ax.set_ylabel(lbl, fontsize=8)
            if i == 2:
                ax.set_xlabel("genomic position (kb)", fontsize=9)
    fig.suptitle("Figure 3 H/I/J (reproduced) — RNA-seq coverage: observed vs Shorkie vs Random_Init", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPRO / "reproduced" / "Figure_3HIJ_coverage.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("saved", out)
    write_verdicts(checks, REPRO.parent / "recheck" / "recheck_checks_coverage.csv")
    for c in checks:
        print(f"  {c.panel} {c.metric} repro={c.reproduced} {c.verdict}")


if __name__ == "__main__":
    main()
