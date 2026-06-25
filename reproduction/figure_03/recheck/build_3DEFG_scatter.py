#!/usr/bin/env python3
"""Figure 3 D/E/F/G — bin- and gene-level scatter panels, reproduced with the
EXACT source-of-truth recipes (the previous reproduction pooled the wrong eval
files and got 0.9178 for gene-level).

  * 3D  Pearson's R (Bin level)            — 1_bin_level_freq_viz.py::scatter_all_groups_scatter
        top-level bin acc.txt, RNA-Seq + 1000-strain, mean-per-identifier, group
        means over ALL valid points.
  * 3E  Pearson's R (Gene Level)           — 3_gene_level_score_dist_viz.py, level="track"
        per-data-type acc.txt in gene_level_eval_rc/f*/{dt}/, group
        [identifier,description,group], mean across folds, metric=pearsonr.
  * 3F  Pearson's R Norm (Gene Level)      — same as 3E, metric=pearsonr_norm.
  * 3G  Pearson's R within-gene (Track Lvl)— level="gene": per-data-type gene_acc.txt,
        group [gene_id], mean across folds, DROP bottom-10% coverage_norm_self,
        metric=pearsonr_gene.
  (The level<->title inversion is in the released script: panels E/F titled
  "(Gene Level)" come from level="track"; G titled "(Track Level)" from level="gene".)
  Gene-level group means are over positive values only (matches the script).

Outputs reproduced/Figure_3DEFG_reproduced.png and recheck/fig3DEFG_means.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shorkie import config
config.load()
WORK = str(config.path("work_root"))
REPRO = config.repo_root() / "reproduction" / "figure_03"
BASE = f"{WORK}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
SHK = f"{BASE}/self_supervised_unet_small_bert_drop"
RND = f"{BASE}/supervised_unet_small_bert_drop_variants/learning_rate_0.0005"

COLORS = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3"]
MEAN_COLORS = ["#2b6291", "#cc6600", "#3d8c3b", "#773d80"]


# ---------------------------------------------------------------- 3D (bin) ----
def categorize(desc):
    d = str(desc).lower()
    if "pos_logfe" in d or "chip-exo" in d:
        return "ChIP-exo"
    if "chip-mnase" in d or "mnase" in d:
        return "ChIP-MNase"
    if "1000 strains rnaseq" in d:
        return "1000 strains RNA-Seq"
    if "rnaseq" in d or "rna_seq" in d:
        return "RNA-Seq"
    return "Other"


def groups_bin():
    rows = []
    for ttype, root in [("supervised", RND), ("self_supervised", SHK)]:
        for fold in range(8):
            p = f"{root}/train/f{fold}c0/eval/acc.txt"
            try:
                d = pd.read_csv(p, sep="\t")
            except FileNotFoundError:
                continue
            d["track_type"] = d["description"].apply(categorize)
            d["training_type"] = ttype
            rows.append(d)
    big = pd.concat(rows, ignore_index=True)
    groups = []
    for trk in ["RNA-Seq", "1000 strains RNA-Seq"]:
        dft = big[big.track_type == trk]
        g = dft.groupby(["identifier", "training_type"])["pearsonr"].mean().reset_index()
        piv = g.pivot(index="identifier", columns="training_type", values="pearsonr")
        if "supervised" not in piv or "self_supervised" not in piv:
            continue
        piv = piv.dropna(subset=["supervised", "self_supervised"]).reset_index()
        piv = piv.rename(columns={"supervised": "pearsonr_sup", "self_supervised": "pearsonr_self"})
        groups.append((piv, trk))
    return groups


# ----------------------------------------------------------- 3E/3F/3G (gene) --
def load_avg(fname, on_cols, dt, do_filter):
    dfs = []
    for f in range(8):
        sp = f"{SHK}/gene_level_eval_rc/f{f}c0/{dt}/{fname}"
        up = f"{RND}/gene_level_eval_rc/f{f}c0/{dt}/{fname}"
        if not (Path(sp).exists() and Path(up).exists()):
            continue
        s = pd.read_csv(sp, sep="\t")
        u = pd.read_csv(up, sep="\t")
        on = [c for c in on_cols if c in s.columns]
        dfs.append(pd.merge(s, u, on=on, suffixes=("_self", "_sup")))
    if not dfs:
        return pd.DataFrame()
    big = pd.concat(dfs, ignore_index=True)
    gc = [c for c in on_cols if c in big.columns]
    avg = big.groupby(gc, as_index=False).mean(numeric_only=True)
    if do_filter and "coverage_norm_self" in avg.columns:
        thr = np.percentile(avg["coverage_norm_self"].dropna(), 10)
        avg = avg[avg["coverage_norm_self"] > thr]
    return avg


def groups_gene(fname, on_cols, do_filter):
    groups = []
    for dt in ["RNA-Seq", "1000-RNA-seq"]:
        avg = load_avg(fname, on_cols, dt, do_filter)
        if not avg.empty:
            groups.append((avg, dt))
    return groups


# ------------------------------------------------------------------ plotting --
def plot_panel(ax, groups, metric, title, pos_only):
    """Replicates plot_all_groups_scatter / scatter_all_groups_scatter for one metric."""
    all_x, all_y = [], []
    recs = []
    for gi, (gdf, label) in enumerate(groups):
        cs, cu = f"{metric}_self", f"{metric}_sup"
        if cs not in gdf.columns or cu not in gdf.columns:
            continue
        sub = gdf[[cs, cu]].dropna()
        x, y = sub[cu].values, sub[cs].values
        all_x.extend(x); all_y.extend(y)
        ax.scatter(x, y, s=15, color=COLORS[gi], alpha=0.45, label=label)
    for gi, (gdf, label) in enumerate(groups):
        cs, cu = f"{metric}_self", f"{metric}_sup"
        if cs not in gdf.columns or cu not in gdf.columns:
            continue
        vs, vf = gdf[cu].dropna(), gdf[cs].dropna()
        if pos_only:
            xm = vs[vs > 0].mean() if (vs > 0).any() else 0.0
            ym = vf[vf > 0].mean() if (vf > 0).any() else 0.0
        else:
            xm, ym = vs.mean(), vf.mean()
        # fraction above diagonal (y>x) over non-negative points
        sub = gdf[[cs, cu]].dropna()
        xx, yy = sub[cu].values, sub[cs].values
        nn = (xx >= 0) & (yy >= 0)
        pct = 100.0 * np.mean(yy[nn] > xx[nn]) if nn.any() else np.nan
        recs.append(dict(panel=title, group=label, n=int(nn.sum()),
                         Random_mean_x=round(float(xm), 4), Shorkie_mean_y=round(float(ym), 4),
                         pct_above_diag=round(float(pct), 1)))
        lbl = f"Shorkie_Random_Init Mean (x): {xm:.2f}, \nShorkie Mean (y): {ym:.2f}"
        ax.scatter(xm, ym, s=100, edgecolor="black", color=MEAN_COLORS[gi], label=lbl)

    amin = min(min(all_x), min(all_y)); amax = max(max(all_x), max(all_y))
    if amin < 0:
        amin = 0.05
    pad = 0.05
    ax.set_xlim(amin - pad, amax + pad); ax.set_ylim(amin - pad, amax + pad)
    ax.plot([amin - pad, amax + pad], [amin - pad, amax + pad], "k--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Shorkie_Random_Init (x)")
    ax.set_ylabel("Shorkie (y)")
    ax.legend(fontsize=8, loc="lower right")
    return recs


def main():
    panels = [
        ("3D", groups_bin(),                                          "pearsonr",      "Pearson's R (Bin level)",                 False),
        ("3E", groups_gene("acc.txt", ["identifier", "description", "group"], False), "pearsonr",      "Pearson's R (Gene Level)",                True),
        ("3F", groups_gene("acc.txt", ["identifier", "description", "group"], False), "pearsonr_norm", "Pearson's R Norm (Gene Level)",           True),
        ("3G", groups_gene("gene_acc.txt", ["gene_id"], True),        "pearsonr_gene", "Pearson's R within-gene (Track Level)",   True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.9))
    all_recs = []
    for ax, (pid, groups, metric, title, pos_only) in zip(axes, panels):
        recs = plot_panel(ax, groups, metric, title, pos_only)
        for r in recs:
            r["panel_id"] = pid
        all_recs += recs
    fig.suptitle("Figure 3 D/E/F/G (reproduced) — bin & gene-level Shorkie vs Shorkie_Random_Init", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = REPRO / "reproduced" / "Figure_3DEFG_reproduced.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("saved", out)

    df = pd.DataFrame(all_recs)[["panel_id", "panel", "group", "n", "Random_mean_x", "Shorkie_mean_y", "pct_above_diag"]]
    csv = REPRO / "recheck" / "fig3DEFG_means.csv"
    df.to_csv(csv, index=False)
    print("wrote", csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
