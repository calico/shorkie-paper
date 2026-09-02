#!/usr/bin/env python3
"""Revision experiment 11, step 3 — score every re-initialisation arm with cross-fold
intervals.

Reuses experiment 01's machinery rather than recomputing metrics, so the arms arrive on the
same scale, with the same panel recipes and the same paired-fold statistics as the published
Shorkie and Shorkie_Random_Init numbers. Anything else would make the comparison
uninterpretable.

Five arms are scored: the three re-initialisation arms trained by step 2, plus the two
released anchors (Shorkie = all LM weights kept, Shorkie_Random_Init = none kept), which
bracket the range and are read from their existing evaluation trees.

The result is read as a decomposition. If resetting the transformer costs much more than
resetting the convolutional tower, the transferred information is concentrated in the
transformer blocks; if the two cost similarly, it is distributed -- which is the outcome the
is a real possibility, and should be reported as such rather than forced into a
story.

**Parameter counts must be reported alongside.** The three groups are not the same size:
the transformer holds ~60% of the trunk parameters against ~16% for the convolutional tower
(see ``results/layer_partition.csv``). A larger drop from resetting the transformer is
therefore expected on capacity grounds alone, so the interesting quantity is the drop
*relative* to how much of the model was discarded.

Writes ``results/arm_metrics.csv`` and ``results/arm_comparison.png``. CPU only.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shorkie import config

ARMS = ["conv_reset", "transformer_reset", "decoder_reset"]
EXP01 = "scripts/05_revision/01_headline_uncertainty"

def parse_args():
    parser = argparse.ArgumentParser(description="Score the re-initialisation arms.")
    parser.add_argument("--panel", default="3C", help="Figure-3 panel statistic to report")
    parser.add_argument("--track_type", default="RNA-Seq")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--n_boot", type=int, default=2000)
    return parser.parse_args()

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    collect = repo / EXP01 / "1_collect_fold_metrics.py"
    boot = repo / EXP01 / "2_bootstrap_ci.py"
    for p in (collect, boot):
        if not p.exists():
            sys.exit(f"error: experiment 01 script not found: {p}")

    frames = []
    for arm in ARMS:
        arm_root = out_dir / "arms" / arm
        if not arm_root.exists():
            print(f"SKIPPED: no trained arm at {arm_root} -- run 2_train_arms.sh {arm}",
                  file=sys.stderr)
            continue
        work = out_dir / "metrics" / arm
        work.mkdir(parents=True, exist_ok=True)
        # Experiment 01 expects a supervised_root holding model subtrees; point it at the
        # arm's output directory so the same loaders apply unchanged.
        subprocess.run([sys.executable, str(collect), "--supervised_root", str(arm_root),
                        "--out_dir", str(work)], check=True)
        # --no_verify: the verify CSV compares against the PUBLISHED 0.78/0.67 anchors,
        # which an ablation arm is not expected to hit; leaving it on prints a misleading
        # "0/4 PASS" per arm.
        subprocess.run([sys.executable, str(boot), "--out_dir", str(work),
                        "--n_boot", str(args.n_boot), "--no_verify"], check=True)
        df = pd.read_csv(work / "headline_ci.csv")
        df["arm"] = arm
        frames.append(df)

    # The two released anchors, from experiment 01's own run over the published trees.
    anchor = repo / EXP01 / "results" / "headline_ci.csv"
    if anchor.exists():
        a = pd.read_csv(anchor)
        a = a[a.model.isin(["Shorkie", "Shorkie_Random_Init"])].copy()
        a["arm"] = a.model.map({"Shorkie": "full LM (Shorkie)",
                                "Shorkie_Random_Init": "all random (Random_Init)"})
        frames.append(a)
    else:
        print(f"SKIPPED: anchors not found at {anchor} -- run experiment 01 first",
              file=sys.stderr)

    if not frames:
        sys.exit("error: nothing to score")
    allm = pd.concat(frames, ignore_index=True)
    sel = allm[(allm.panel == args.panel) & (allm.track_type == args.track_type)]
    sel = sel[~sel.model.astype(str).str.startswith("Shorkie_minus")]
    sel.to_csv(out_dir / "arm_metrics.csv", index=False)
    print(sel[["arm", "model", "point", "fold_ci95_lo", "fold_ci95_hi"]].to_string(index=False))

    part_csv = out_dir / "layer_partition.csv"
    if part_csv.exists():
        print("\nparameter share of each group (from 1_make_reinit_checkpoints.py):")
        print(pd.read_csv(part_csv).to_string(index=False))
        print("Interpret each arm's drop relative to the share of the model it discarded.")

    if len(sel):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        order = sel.sort_values("point")
        y = range(len(order))
        ax.errorbar(order.point, list(y),
                    xerr=[order.point - order.fold_ci95_lo,
                          order.fold_ci95_hi - order.point],
                    fmt="o", capsize=3, color="#377eb8")
        ax.set_yticks(list(y)); ax.set_yticklabels(order.arm, fontsize=9)
        ax.set_xlabel(f"{args.panel} {args.track_type}: Pearson's R "
                      "(point estimate, 95% CI over the 8 test folds)")
        ax.set_title("Which pretrained layer group carries the transfer?", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "arm_comparison.png", dpi=160)
        plt.close(fig)
        print(f"\nwrote {out_dir/'arm_comparison.png'}")
    print(f"wrote {out_dir/'arm_metrics.csv'}")

if __name__ == "__main__":
    main()
