#!/usr/bin/env python3
"""Revision experiment 06, step 3 — add Shorkie_Random_Init as a third series to the
Figure 6 MPRA panels.

The published panels compare Shorkie against DREAM-RNN. This step recomputes every one of
them with a third series, so the question is answered on exactly the figure that raises it.

Reuse rather than reimplementation is deliberate: the correlation recipes are imported from
``reproduction/figure_06/recheck/mpra_common.py`` and the Random_Init NPZ tree is swapped in
by repointing that module's ``STRANDED`` root. Every filtering rule the published analysis
applies -- the 180 bp insertion context, the finite mask, the exact-zero drop for the dual
categories, the float16 clip for DREAM -- therefore applies identically to both models. If
these were recomputed independently, any difference could be a difference in recipe.

Panels covered:
  6B/6C  AUROC / AUPRC discriminating high- from low-expression sequences, per insertion site
  6D/6E  Pearson / Spearman vs measured expression, single-sequence categories
  6F/6G/6H  the same for the ref/alt paired categories

Writes ``results/fig6_with_random_init.csv`` and ``results/fig6_three_way.png``.
CPU only, a few minutes.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from shorkie import config

config.load()
_F6 = Path(config.repo_root()) / "reproduction" / "figure_06" / "recheck"
sys.path.insert(0, str(_F6))
import mpra_common as mc                                       # noqa: E402
from build_panels_BC import SITES, QUANTILES                   # noqa: E402

SINGLE_PANELS = {"6D": "yeast_seqs", "6E": "all_random_seqs", "6E2": "challenging_seqs"}
DUAL_PANELS = {"6F": "all_SNVs_seqs", "6G": "motif_perturbation", "6H": "motif_tiling_seqs"}
COLORS = {"Shorkie": "#377eb8", "Shorkie_Random_Init": "#ff7f00", "DREAM-RNN": "green"}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Add Shorkie_Random_Init to the Figure 6 MPRA comparisons.")
    parser.add_argument("--random_init_npz", default=None,
                        help="Random_Init NPZ tree [default:./results/npz]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def with_root(root):
    """Context-manager-free swap of mpra_common's NPZ root, so the published loaders read
    a different model's scores without any other behaviour changing."""
    previous = mc.STRANDED
    mc.STRANDED = str(root)
    return previous

def classifier_metrics():
    """6B/6C per-gene AUROC / AUPRC at each insertion site, for the current NPZ root."""
    out = {}
    for genes in QUANTILES.values():
        for sym in genes:
            orf, strand = mc.ORF[sym], mc.GENE_STRAND[sym]
            hi = mc.gene_site_single("high_exp_seqs", sym, orf, strand)
            lo = mc.gene_site_single("low_exp_seqs", sym, orf, strand)
            per_site = {}
            for c in SITES:
                if c not in hi or c not in lo:
                    continue
                score = np.concatenate([hi[c], lo[c]])
                label = np.concatenate([np.ones(len(hi[c])), np.zeros(len(lo[c]))])
                per_site[c] = (roc_auc_score(label, score),
                               average_precision_score(label, score))
            if per_site:
                out[sym] = per_site
    return out

def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rnd_root = Path(args.random_init_npz) if args.random_init_npz else out_dir / "npz"
    if not rnd_root.exists():
        sys.exit(f"error: {rnd_root} not found -- run steps 1 and 2 first")

    gt = mc.load_ground_truth()
    dream = mc.load_dream()
    rows = []

    for label, root in (("Shorkie", mc.STRANDED), ("Shorkie_Random_Init", str(rnd_root))):
        with_root(root)
        # --- 6B/6C ---------------------------------------------------------------
        metrics = classifier_metrics()
        if metrics:
            auroc = [metrics[g][c][0] for g in metrics for c in metrics[g]]
            auprc = [metrics[g][c][1] for g in metrics for c in metrics[g]]
            rows.append(dict(panel="6B", category="high vs low expression", model=label,
                             metric="AUROC", value=round(float(np.mean(auroc)), 4),
                             n=len(auroc)))
            rows.append(dict(panel="6C", category="high vs low expression", model=label,
                             metric="AUPRC", value=round(float(np.mean(auprc)), 4),
                             n=len(auprc)))
        else:
            print(f"SKIPPED: no high/low NPZ for {label}", file=sys.stderr)

        # --- 6D/6E and 6F/6G/6H --------------------------------------------------
        for panel, seq_type in {**SINGLE_PANELS, **DUAL_PANELS}.items():
            dual = seq_type in DUAL_PANELS.values()
            try:
                _, _, r, rho, n, _ = (mc.shorkie_dual(seq_type, gt) if dual
                                      else mc.shorkie_single(seq_type, gt))
            except Exception as e:
                print(f"SKIPPED: {label} {panel} {seq_type} ({e})", file=sys.stderr)
                continue
            rows.append(dict(panel=panel, category=seq_type, model=label,
                             metric="pearson", value=round(float(r), 4), n=n))
            rows.append(dict(panel=panel, category=seq_type, model=label,
                             metric="spearman", value=round(float(rho), 4), n=n))

    # --- DREAM-RNN reference series (independent of the NPZ root) -----------------
    for panel, seq_type in {**SINGLE_PANELS, **DUAL_PANELS}.items():
        dual = seq_type in DUAL_PANELS.values()
        try:
            _, _, r, rho, n = (mc.dream_dual(seq_type, gt, dream) if dual
                               else mc.dream_single(seq_type, gt, dream))
        except Exception as e:
            print(f"SKIPPED: DREAM-RNN {panel} ({e})", file=sys.stderr)
            continue
        rows.append(dict(panel=panel, category=seq_type, model="DREAM-RNN",
                         metric="pearson", value=round(float(r), 4), n=n))
        rows.append(dict(panel=panel, category=seq_type, model="DREAM-RNN",
                         metric="spearman", value=round(float(rho), 4), n=n))

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("error: nothing computed")
    df.to_csv(out_dir / "fig6_with_random_init.csv", index=False)

    pear = df[df.metric.isin(["pearson", "AUROC", "AUPRC"])]
    piv = pear.pivot_table(index=["panel", "category"], columns="model", values="value")
    print(piv.to_string())

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    panels = list(dict.fromkeys(pear.panel))
    x = np.arange(len(panels))
    width = 0.26
    for i, model in enumerate(["Shorkie", "Shorkie_Random_Init", "DREAM-RNN"]):
        vals = [pear[(pear.panel == p) & (pear.model == model)].value.mean() for p in panels]
        ax.bar(x + (i - 1) * width, vals, width, color=COLORS[model], label=model)
    ax.set_xticks(x); ax.set_xticklabels(panels)
    ax.set_ylabel("Pearson R (6D-6H) / AUROC-AUPRC (6B-6C)")
    ax.set_title("Figure 6 with Shorkie_Random_Init added as a third series", fontsize=12)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_three_way.png", dpi=160)
    plt.close(fig)
    print(f"\nwrote {out_dir/'fig6_with_random_init.csv'}")
    print(f"wrote {out_dir/'fig6_three_way.png'}")

if __name__ == "__main__":
    main()
