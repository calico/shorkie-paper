#!/usr/bin/env python3
"""
plot_unet_small_comparison.py

Refactored plotting utility for training/validation losses across multiple models.

Highlights
- Robust log parsing (tolerates various "Epoch …" line formats)
- Per-model steps-per-epoch (handles rescaling like 500/150 easily)
- Clean config object for names, files, colors, styles
- Moving-average smoothing with safe edge handling + optional trim_end
- Consistent legend labeling and minima markers
- CLI for overrides (window size, figsize, dpi, etc.)

Usage (defaults match your original):
    python plot_unet_small_comparison.py \
        --outdir viz/unet_small_comparison \
        --window-size 21 \
        --trim-end 5 \
        --steps 64 \
        --figsize 6 4 \
        --dpi 600
"""

from __future__ import annotations
import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------ Config ------------------------------

@dataclass
class ModelConfig:
    name: str
    log_file: str
    color: str = "tab:blue"
    style_train: str = "-"     # line style for training curve
    style_valid: str = "--"    # line style for validation curve
    steps_per_epoch: float = 64.0  # allows rescaling per model


def default_model_configs(root_dir: str, base_steps: float) -> List[ModelConfig]:
    """
    Builds the default model list matching your original setup.
    If you need the Saccharomycetales model to use a different step scale,
    just change `steps_per_epoch` for that entry.
    """
    return [
        ModelConfig(
            name="R64_Yeast",
            log_file=f"{root_dir}lm_r64_gtf/lm_r64_gtf_unet_small/train/train.out",
            color="tab:blue",
            steps_per_epoch=base_steps,
        ),
        ModelConfig(
            name="80_Strains",
            log_file=f"{root_dir}lm_strains_gtf/lm_strains_gtf_unet_small/train/train.out",
            color="tab:orange",
            steps_per_epoch=base_steps,
        ),
        ModelConfig(
            name="165_Saccharomycetales",
            log_file=f"{root_dir}LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small/train/train.out",
            color="tab:green",
            # Example: rescale if this model logs fewer "Epoch" lines per same wall time
            # steps_per_epoch=base_steps * (500.0 / 150.0),
            steps_per_epoch=base_steps,
        ),
        ModelConfig(
            name="1342_Fungus",
            log_file=f"{root_dir}LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small/train/train.out",
            color="tab:red",
            steps_per_epoch=base_steps,
        ),
    ]


# ------------------------------ IO & Parsing ------------------------------

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_epoch_line(line: str, epoch_prefix: str = "Epoch") -> Optional[Tuple[float, float]]:
    """
    Extract (train_loss, valid_loss) from a log line that begins with 'Epoch'.
    We avoid numbers before the first colon (i.e., the epoch index).
    Then we find the first two floats afterwards as train and valid.

    Works with lines like:
        'Epoch 1: train 0.123  val 0.456  R 0.88  R2 0.75'
        'Epoch 2: 0.1234, 0.4567 ...'
        'Epoch 3: train: 0.1, valid: 0.2'
    """
    s = line.strip()
    if not s.startswith(epoch_prefix):
        return None
    if ":" not in s:
        return None
    payload = s.split(":", 1)[1]  # everything after the first colon
    nums = FLOAT_RE.findall(payload)
    if len(nums) < 2:
        return None
    try:
        train = float(nums[0])
        valid = float(nums[1])
        return train, valid
    except ValueError:
        return None


def load_metrics(log_path: str, epoch_prefix: str = "Epoch") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_losses, valid_losses) as 1D arrays.
    Missing file -> returns empty arrays.
    """
    if not os.path.isfile(log_path):
        print(f"[WARN] Missing log file: {log_path}")
        return np.array([]), np.array([])

    train_vals, valid_vals = [], []
    with open(log_path, "rt") as f:
        for raw in f:
            parsed = parse_epoch_line(raw, epoch_prefix=epoch_prefix)
            if parsed is not None:
                tr, va = parsed
                train_vals.append(tr)
                valid_vals.append(va)

    return np.asarray(train_vals, dtype=float), np.asarray(valid_vals, dtype=float)


# ------------------------------ Smoothing ------------------------------

def moving_average(x: np.ndarray, window_size: int = 1) -> np.ndarray:
    """
    Safe moving average with adaptive edges (mirroring behavior of your original loop).
    For each j, averages a symmetric window truncated at boundaries.
    """
    if window_size <= 1 or x.size == 0:
        return x.copy()

    out = np.empty_like(x, dtype=float)
    half = window_size // 2
    n = len(x)
    for j in range(n):
        lo = max(0, j - half)
        hi = min(n - 1, j + half)
        out[j] = x[lo:hi + 1].mean()
    return out


def maybe_trim_end(x: np.ndarray, trim_end: int) -> np.ndarray:
    if trim_end <= 0:
        return x
    if trim_end >= len(x):
        # Avoid empty arrays; keep at least one point
        return x[:1]
    return x[:-trim_end]


# ------------------------------ Plotting ------------------------------

def compute_steps_axis(length: int, steps_per_epoch: float) -> np.ndarray:
    """Return x-values as '# training batches', starting at 1*steps_per_epoch."""
    if length <= 0:
        return np.array([])
    return np.arange(1, length + 1, dtype=float) * steps_per_epoch


def plot_losses(
    models: List[ModelConfig],
    all_train: List[np.ndarray],
    all_valid: List[np.ndarray],
    outdir: str,
    window_size: int = 21,
    trim_end: int = 5,
    figsize: Tuple[float, float] = (6.0, 4.0),
    dpi: int = 600,
    epoch_prefix: str = "Epoch",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    title_suffix: str = "",
):
    os.makedirs(outdir, exist_ok=True)

    # --- Training plot ---
    fig = plt.figure(figsize=figsize, dpi=dpi)
    xmins, xmaxs = [], []

    for cfg, tr in zip(models, all_train):
        if tr.size == 0:
            print(f"[WARN] No training metrics for {cfg.name}")
            continue

        tr_smooth = moving_average(tr, window_size=window_size)
        tr_smooth = maybe_trim_end(tr_smooth, trim_end)
        tr_orig   = maybe_trim_end(tr, trim_end)

        xs = compute_steps_axis(len(tr_smooth), cfg.steps_per_epoch)
        if xs.size:
            xmins.append(xs[0])
            xmaxs.append(xs[-1])

        # Legend shows min of ORIGINAL (unsmoothed)
        min_idx = int(np.argmin(tr_orig))
        min_val = float(np.min(tr_orig))

        plt.plot(xs, tr_smooth, label=f"{cfg.name}; loss = {min_val:.4f}",
                 linewidth=1.2, linestyle=cfg.style_train, color=cfg.color, zorder=-100)

        # Mark the (unsmoothed) minimum with a vertical line
        xs_full = compute_steps_axis(len(tr_orig), cfg.steps_per_epoch)
        if xs_full.size:
            plt.axvline(x=xs_full[min_idx], linestyle='-', linewidth=1, alpha=0.25,
                        color=cfg.color, zorder=-200)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if xmin is None and xmins:
        xmin = min(xmins)
    if xmax is None and xmaxs:
        xmax = max(xmaxs)
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    plt.xlabel('# Training Batches', fontsize=10)
    plt.ylabel('Training Loss', fontsize=10)
    plt.title(f'Training Losses{title_suffix}')
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(outdir, "train_losses.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")

    # --- Validation plot ---
    fig = plt.figure(figsize=figsize, dpi=dpi)
    xmins, xmaxs = [], []

    for cfg, va in zip(models, all_valid):
        if va.size == 0:
            print(f"[WARN] No validation metrics for {cfg.name}")
            continue

        va_smooth = moving_average(va, window_size=window_size)
        va_smooth = maybe_trim_end(va_smooth, trim_end)
        va_orig   = maybe_trim_end(va, trim_end)

        xs = compute_steps_axis(len(va_smooth), cfg.steps_per_epoch)
        if xs.size:
            xmins.append(xs[0])
            xmaxs.append(xs[-1])

        # Legend shows min of ORIGINAL (unsmoothed)
        min_idx = int(np.argmin(va_orig))
        min_val = float(np.min(va_orig))

        plt.plot(xs, va_smooth, label=f"{cfg.name}; loss = {min_val:.4f}",
                 linewidth=1.2, linestyle=cfg.style_valid, color=cfg.color, zorder=-100)

        xs_full = compute_steps_axis(len(va_orig), cfg.steps_per_epoch)
        if xs_full.size:
            plt.axvline(x=xs_full[min_idx], linestyle='--', linewidth=1, alpha=0.25,
                        color=cfg.color, zorder=-200)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if xmin is None and xmins:
        xmin = min(xmins)
    if xmax is None and xmaxs:
        xmax = max(xmaxs)
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    plt.xlabel('# Training Batches', fontsize=10)
    plt.ylabel('Validation Loss', fontsize=10)
    plt.title(f'Validation Losses{title_suffix}')
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(outdir, "valid_losses.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


# ------------------------------ Main ------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training/validation losses for multiple models.")
    p.add_argument("--root-dir", type=str,
                   default="/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/",
                   help="Base directory containing model subfolders.")
    p.add_argument("--outdir", type=str, default="viz/unet_small_comparison/",
                   help="Where to write PNGs.")
    p.add_argument("--steps", type=float, default=64.0,
                   help="Default steps-per-epoch; can be overridden per model in code.")
    p.add_argument("--window-size", type=int, default=21, help="Moving-average window.")
    p.add_argument("--trim-end", type=int, default=5, help="Drop this many points from end.")
    p.add_argument("--figsize", type=float, nargs=2, default=(6.0, 4.0),
                   help="Figure size (inches): WIDTH HEIGHT")
    p.add_argument("--dpi", type=int, default=600, help="Figure DPI.")
    p.add_argument("--epoch-prefix", type=str, default="Epoch", help="Prefix marking epoch lines.")
    p.add_argument("--xmin", type=float, default=None, help="x-axis lower bound; default auto.")
    p.add_argument("--xmax", type=float, default=None, help="x-axis upper bound; default auto.")
    return p


def main():
    args = build_argparser().parse_args()

    # Build model list (edit here for per-model overrides if needed)
    models = default_model_configs(root_dir=args.root_dir, base_steps=args.steps)

    # Load metrics
    trains, valids = [], []
    for cfg in models:
        tr, va = load_metrics(cfg.log_file, epoch_prefix=args.epoch_prefix)
        trains.append(tr)
        valids.append(va)

    # Plot
    plot_losses(
        models=models,
        all_train=trains,
        all_valid=valids,
        outdir=args.outdir,
        window_size=args.window_size,
        trim_end=args.trim_end,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        epoch_prefix=args.epoch_prefix,
        xmin=args.xmin,
        xmax=args.xmax,
        title_suffix="",
    )


if __name__ == "__main__":
    main()
