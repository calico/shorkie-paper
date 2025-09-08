#!/usr/bin/env python3
"""
model_arch_comparison.py

Refactored from your script:
- Centralized config per dataset (labels, epochs, file templates).
- Robust, reusable log parsing (tolerates minor format variance).
- Single plotting routine for train/valid to avoid code duplication.
- Safer smoothing/trim + graceful handling of missing logs.
- Dynamic x-limits based on configured epochs (with small margin).

Behavior preserved:
- Colors, line styles, labels, smoothing window, trim_end.
- Vertical line at the smoothed-curve minimum.
- File outputs and naming.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# =========================
# User parameters
# =========================
FIGSIZE: Tuple[float, float] = (5, 2.85)
DPI: int = 300

steps: int = 64
trim_end: int = 5
window_size: int = 21

input_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI"
out_dir   = "viz/model_arch_comparison/all_datasets"
os.makedirs(out_dir, exist_ok=True)

# Datasets + epochs + labels (used for x-limit and pretty titles)
DATASET_SPECS = {
    "r64": {
        "label": "R64",
        "epochs": 23,
        "base": f"{input_dir}/lm_r64_gtf/lm_r64_gtf",
        "models": {
            "Conv_Small":   "{base}_small/train/train.out",
            "Conv_Big":     "{base}_big/train/train.out",
            "U-Net_Small":  "{base}_unet_small/train/train.out",
            "U-Net_Big":    "{base}_unet_big/train/train.out",
        },
    },
    "strains": {
        "label": "80_strains",
        "epochs": 41,
        "base": f"{input_dir}/lm_strains_gtf/lm_strains_gtf",
        "models": {
            "Conv_Small":   "{base}_small/train/train.out",
            "Conv_Big":     "{base}_big/train/train.out",
            "U-Net_Small":  "{base}_unet_small/train/train.out",
            "U-Net_Big":    "{base}_unet_big/train/train.out",
        },
    },
    "saccharomycetales": {
        "label": "165_Saccharomycetales",
        "epochs": 450,
        "base": f"{input_dir}/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf",
        "models": {
            "Conv_Small":   "{base}_small/train/train.out",
            "Conv_Big":     "{base}_big/train/train.out",
            "U-Net_Small":  "{base}_unet_small_bert_drop/train/train.out",
            "U-Net_Big":    "{base}_unet_big_bert_drop/train_bk/train.out",
        },
    },
    "fungi_1385": {
        "label": "1342_fungus",
        "epochs": 350,
        "base": f"{input_dir}/LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf",
        "models": {
            "Conv_Small":   "{base}_small_bert/train/train.out",
            "Conv_Big":     "{base}_big_bert/train/train.out",
            "U-Net_Small":  "{base}_unet_small_bert_drop/train/train.out",
            "U-Net_Big":    "{base}_unet_big_bert_drop/train/train.out",
        },
    },
}

# Plotting aesthetics
MODEL_NAMES    = ["Conv_Small", "Conv_Big", "U-Net_Small", "U-Net_Big"]
DATASETS_ORDER = ["r64", "strains", "saccharomycetales", "fungi_1385"]

dataset_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
model_colors   = ["tab:purple", "tab:olive", "tab:cyan", "tab:pink"]
model_styles   = ["-", "--"]  # [train, valid]


# =========================
# Utilities
# =========================

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_epoch_line(line: str, epoch_prefix: str = "Epoch") -> Optional[Tuple[float, float]]:
    """Extract (train_loss, valid_loss) after the first colon on an 'Epoch' line."""
    s = line.strip()
    if not s.startswith(epoch_prefix) or ":" not in s:
        return None
    payload = s.split(":", 1)[1]
    nums = FLOAT_RE.findall(payload)
    if len(nums) < 2:
        return None
    try:
        return float(nums[0]), float(nums[1])
    except ValueError:
        return None

def load_losses(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a train.out and return (train_losses, valid_losses). Missing → ([], [])."""
    if not os.path.isfile(filepath):
        print(f"[WARN] Missing log file: {filepath}")
        return np.array([]), np.array([])
    tr, va = [], []
    with open(filepath, "rt") as f:
        for line in f:
            parsed = parse_epoch_line(line, epoch_prefix="Epoch")
            if parsed:
                a, b = parsed
                tr.append(a); va.append(b)
    return np.asarray(tr, float), np.asarray(va, float)

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size == 0:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    half = w // 2
    n = len(x)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        out[i] = x[lo:hi+1].mean()
    return out

def trim_tail(x: np.ndarray, t: int) -> np.ndarray:
    if t <= 0:
        return x
    if t >= len(x):
        return x[:1]
    return x[:-t]

def steps_axis(n_points: int, step: float) -> np.ndarray:
    if n_points <= 0:
        return np.array([])
    return np.arange(1, n_points + 1, dtype=float) * step


# =========================
# Data assembly
# =========================

@dataclass
class Series:
    xs: np.ndarray
    ys: np.ndarray
    label: str
    color: str
    style: str
    vline_x: Optional[float] = None  # where to draw a vertical line

def build_metrics() -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Returns nested dict:
      metrics[dataset][model] = {"train": np.ndarray, "valid": np.ndarray, "steps": np.ndarray}
    """
    metrics: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for ds_key in DATASETS_ORDER:
        spec = DATASET_SPECS[ds_key]
        base = spec["base"]
        ds_metrics: Dict[str, Dict[str, np.ndarray]] = {}
        for model_name in MODEL_NAMES:
            pattern = spec["models"][model_name]
            filepath = pattern.format(base=base)
            tr, va = load_losses(filepath)
            st = steps_axis(len(tr), steps)  # steps from training length
            ds_metrics[model_name] = {"train": tr, "valid": va, "steps": st}
        metrics[ds_key] = ds_metrics
    return metrics


# =========================
# Plotting
# =========================

def plot_group(
    series_list: List[Series],
    title: str,
    ylabel: str,
    outfile: str,
    xmin: float,
    xmax: float,
):
    plt.figure(figsize=FIGSIZE, dpi=DPI)

    max_seen = xmin
    for s in series_list:
        if s.xs.size == 0 or s.ys.size == 0:
            continue
        plt.plot(s.xs, s.ys, label=s.label, color=s.color, linestyle=s.style, linewidth=1.5)
        if s.vline_x is not None:
            plt.axvline(x=s.vline_x, color=s.color, linestyle=s.style, alpha=0.25)
        if s.xs[-1] > max_seen:
            max_seen = s.xs[-1]

    # dynamic right bound: min(max of data, configured limit) with a small margin
    right = min(max_seen + 1000, xmax) if xmax > 0 else max_seen + 1000

    plt.xlim(xmin, right)
    plt.xlabel("# Training Batches")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"[OK] Wrote {outfile}")


def prepare_smoothed_series(
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
    style: str,
    label_prefix: str,
) -> Optional[Series]:
    if ys.size == 0 or xs.size == 0:
        return None
    sm = moving_average(ys, window_size)
    sm = trim_tail(sm, trim_end)
    xs = xs[:len(sm)]
    mn = float(sm.min())
    arg = int(sm.argmin())
    vline = float(xs[arg]) if xs.size else None
    return Series(xs=xs, ys=sm, label=f"{label_prefix}; min={mn:.4f}", color=color, style=style, vline_x=vline)


# =========================
# Main orchestration
# =========================

def main():
    metrics = build_metrics()

    # ---------- Per-model plots (each model across datasets) ----------
    for m_idx, model_name in enumerate(MODEL_NAMES):
        # Training
        series: List[Series] = []
        # x-limits: start at one step; end at max epochs across these datasets
        xmin = float(steps)
        xmax = max(DATASET_SPECS[ds]["epochs"] * steps for ds in DATASETS_ORDER)
        for ds_i, ds_key in enumerate(DATASETS_ORDER):
            ds = metrics[ds_key][model_name]
            s = prepare_smoothed_series(
                xs=ds["steps"],
                ys=ds["train"],
                color=dataset_colors[ds_i],
                style=model_styles[0],
                label_prefix=ds_key,
            )
            if s: series.append(s)

        plot_group(
            series_list=series,
            title=f"{model_name}: Training Loss Across Datasets",
            ylabel="Training Loss",
            outfile=os.path.join(out_dir, f"{model_name}_train_losses.png"),
            xmin=xmin,
            xmax=xmax,
        )

        # Validation
        series = []
        for ds_i, ds_key in enumerate(DATASETS_ORDER):
            ds = metrics[ds_key][model_name]
            s = prepare_smoothed_series(
                xs=ds["steps"],
                ys=ds["valid"],
                color=dataset_colors[ds_i],
                style=model_styles[1],
                label_prefix=ds_key,
            )
            if s: series.append(s)

        plot_group(
            series_list=series,
            title=f"{model_name}: Validation Loss Across Datasets",
            ylabel="Validation Loss",
            outfile=os.path.join(out_dir, f"{model_name}_valid_losses.png"),
            xmin=xmin,
            xmax=xmax,
        )

    # ---------- Per-dataset plots (each dataset across models) ----------
    for ds_i, ds_key in enumerate(DATASETS_ORDER):
        spec = DATASET_SPECS[ds_key]
        label = spec["label"]
        epochs = spec["epochs"]
        xmin = float(steps)
        xmax = float(epochs * steps)

        # Training
        series: List[Series] = []
        for m_idx, model_name in enumerate(MODEL_NAMES):
            ds = metrics[ds_key][model_name]
            s = prepare_smoothed_series(
                xs=ds["steps"],
                ys=ds["train"],
                color=model_colors[m_idx],
                style=model_styles[0],
                label_prefix=model_name,
            )
            if s: series.append(s)

        plot_group(
            series_list=series,
            title=f"{label}: Training Loss Across Models",
            ylabel="Training Loss",
            outfile=os.path.join(out_dir, f"{label}_train_losses.png"),
            xmin=xmin,
            xmax=xmax,
        )

        # Validation
        series = []
        for m_idx, model_name in enumerate(MODEL_NAMES):
            ds = metrics[ds_key][model_name]
            s = prepare_smoothed_series(
                xs=ds["steps"],
                ys=ds["valid"],
                color=model_colors[m_idx],
                style=model_styles[1],
                label_prefix=model_name,
            )
            if s: series.append(s)

        plot_group(
            series_list=series,
            title=f"{label}: Validation Loss Across Models",
            ylabel="Validation Loss",
            outfile=os.path.join(out_dir, f"{label}_valid_losses.png"),
            xmin=xmin,
            xmax=xmax,
        )


if __name__ == "__main__":
    main()
