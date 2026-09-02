#!/usr/bin/env python3
"""Revision experiment 10, step 1 (companion) — compare LM training dynamics across the four
pretraining corpora.

The "sweet spot" is confounded with corpus size, annotation quality, and -- the one this step
addresses -- OPTIMISATION DIFFICULTY.

Three confounds are named above. ``1_characterize_corpora.py`` quantifies the first two
from the species lists and the manifest. The third -- optimisation -- needs the training
logs, which are on disk for all four tiers, so it costs no GPU either and there is no reason
to leave it as an acknowledgement in prose.

The question is whether a tier's ranking could be an artifact of how its run trained rather
than of what it learned. For each tier this reports the best validation loss and when it
occurred, how many epochs ran, the loss slope over the final 10% of epochs -- and, crucially,
the EPOCH BUDGET AND PATIENCE the run was actually given, read from that tier's own
params.json.

That last part matters more than it might seem. A tier that stopped improving long before it
stopped training was not optimisation-limited within its own budget; a tier still descending
when its budget ran out was. But if the tiers were given DIFFERENT budgets, then "converged
within its own budget" is not the same as "trained comparably", and the comparison between
them carries an optimisation confound regardless of each run's individual convergence. The
script checks for that explicitly rather than assuming the schedules matched.

Parses ``train.out`` with the same ``valid_loss:`` regex as
``reproduction/figure_01/recheck/recompute_fig01.py`` -- deliberately, because the legacy
``split(':', 1)`` parser mis-reads the ``steps:`` field as the loss.

Writes ``results/training_dynamics.csv`` and ``results/training_dynamics.png``.
CPU only, seconds.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shorkie import config

# tier label -> train.out sub-path under results.lm_eval_logs, matching the Figure 1F
# sub-paths in reproduction/figure_01/recheck/recompute_fig01.py (unet_small architecture).
SUBS = {
    "R64": "lm_r64_gtf/lm_r64_gtf_unet_small",
    "80_strains": "lm_strains_gtf/lm_strains_gtf_unet_small",
    "165_Saccharomycetales": "lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small",
    "1341_Fungus": "lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small",
}
VLRE = re.compile(r"valid_loss:\s*([0-9.]+)")
TAIL_FRACTION = 0.10          # slope is measured over the final 10% of epochs
# Training-schedule keys that must match for a cross-tier comparison to be schedule-matched.
SCHEDULE_KEYS = ["train_epochs_max", "train_epochs_min", "patience", "steps_per_epoch_max"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LM training dynamics across the pretraining corpora.")
    parser.add_argument("--log_root", default=None,
                        help="[default: config results.lm_eval_logs]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def read_schedule(train_out):
    """The tier's own training schedule, from the params.json beside its train/ directory."""
    import json
    params = train_out.parent.parent / "params.json"
    if not params.exists():
        return {k: None for k in SCHEDULE_KEYS}
    try:
        train = json.loads(params.read_text()).get("train", {})
    except (OSError, ValueError):
        return {k: None for k in SCHEDULE_KEYS}
    return {k: train.get(k) for k in SCHEDULE_KEYS}

def read_curve(path):
    """Validation-loss trajectory from a train.out."""
    if not path.exists():
        return None
    vals = [float(m.group(1)) for line in open(path, errors="ignore")
            for m in [VLRE.search(line)] if m]
    return np.array(vals, dtype=float) if vals else None

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.log_root) if args.log_root else config.path("results.lm_eval_logs")
    if root is None or not root.exists():
        sys.exit(f"error: LM training logs not found at {root}\n"
                 "       set --log_root or results.lm_eval_logs in config/paths.yaml")
    print(f"log_root : {root}", flush=True)

    rows, curves = [], {}
    for tier, sub in SUBS.items():
        path = root / sub / "train" / "train.out"
        curve = read_curve(path)
        if curve is None:
            # some tiers live one level up from the LM_Johannes sub-root
            alt = root.parent / sub / "train" / "train.out"
            curve = read_curve(alt)
            path = alt if curve is not None else path
        if curve is None:
            print(f"SKIPPED: no validation-loss trace for {tier} ({path})", file=sys.stderr)
            continue
        curves[tier] = curve
        n = len(curve)
        argmin = int(np.argmin(curve))
        tail = max(2, int(n * TAIL_FRACTION))
        x = np.arange(tail, dtype=float)
        slope = float(np.polyfit(x, curve[-tail:], 1)[0]) if tail > 1 else np.nan
        sched = read_schedule(path)
        rows.append(dict(
            tier=tier, epochs_logged=n, **sched,
            best_valid_loss=round(float(curve.min()), 5),
            best_epoch=argmin,
            best_epoch_fraction=round(argmin / max(n - 1, 1), 3),
            final_valid_loss=round(float(curve[-1]), 5),
            # >0 means the loss was still falling when the run ended
            improvement_after_best=round(float(curve[argmin:].min() - curve.min()), 6),
            tail_slope_per_epoch=round(slope, 8),
            still_descending_at_end=bool(slope < -1e-6),
            log=str(path)))

    if not rows:
        sys.exit("error: no training logs parsed")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "training_dynamics.csv", index=False)
    print(df[["tier", "epochs_logged", "train_epochs_max", "patience", "best_valid_loss",
              "best_epoch", "best_epoch_fraction", "still_descending_at_end"]]
.to_string(index=False))

    stuck = df[df.still_descending_at_end]
    if len(stuck):
        print(f"\nStill descending at the end of its own budget: {', '.join(stuck.tier)} — "
              "those tiers' perplexities understate them.")
    else:
        print("\nWithin its own budget, no tier was still improving materially at the end.")

    # The comparison is only optimisation-clean if the tiers shared a schedule.
    varied = {k: sorted({r for r in df[k] if r is not None}) for k in SCHEDULE_KEYS}
    mismatched = {k: v for k, v in varied.items() if len(v) > 1}
    df.attrs["schedule_matched"] = not mismatched
    if mismatched:
        print("\n*** The tiers were NOT trained on a matched schedule. ***")
        for k, v in mismatched.items():
            print(f"    {k}: {v}")
        groups = (df.groupby([k for k in SCHEDULE_KEYS if k in mismatched])["tier"]
.apply(list).to_dict())
        for sched, tiers in groups.items():
            print(f"    {sched} -> {tiers}")
        print("    So 'converged within its own budget' is NOT 'trained comparably'. This is")
        print("    the 'optimization difficulties' confound, and it is real: it must")
        print("    be disclosed, and any tier comparison across schedule groups carries it.")
    else:
        print("\nAll tiers shared a training schedule, so 'optimization difficulties' does "
              "not explain the corpus ranking.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = {"R64": "#999999", "80_strains": "#4daf4a",
               "165_Saccharomycetales": "#377eb8", "1341_Fungus": "#e41a1c"}
    for tier, curve in curves.items():
        ax1.plot(curve, lw=1.2, color=colours.get(tier), label=tier, alpha=0.9)
        ax1.scatter([int(np.argmin(curve))], [curve.min()], s=36,
                    color=colours.get(tier), zorder=3, edgecolor="white")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation loss")
    ax1.set_title("A · validation-loss trajectory (dot = best epoch)", fontsize=11, loc="left")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    y = np.arange(len(df))
    ax2.barh(y, df.best_epoch_fraction, color=[colours.get(t, "#777") for t in df.tier])
    ax2.set_yticks(y); ax2.set_yticklabels(df.tier, fontsize=9)
    ax2.set_xlim(0, 1.05)
    ax2.axvline(1.0, color="0.4", ls="--", lw=1)
    ax2.set_xlabel("best epoch as a fraction of the run\n(1.0 = still improving when the budget ended)")
    ax2.set_title("B · was the run optimisation-limited?", fontsize=11, loc="left")
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_dynamics.png", dpi=160)
    plt.close(fig)
    print(f"\nwrote {out_dir/'training_dynamics.csv'}")
    print(f"wrote {out_dir/'training_dynamics.png'}")

if __name__ == "__main__":
    main()
