#!/usr/bin/env python3
"""Figure 7 H/I (deep-recheck) — AUPRC stratified by variant-to-TSS distance bin.

Faithful re-implementation of
``scripts/04_analysis/shorkie/eqtl/3_visualization/2_AUROC_AUPRC_by_dsitance.py``,
rendering only the two panels the published Figure 7 shows: **H = Caudal AUPRC**,
**I = Kita AUPRC** (both AUPRC; 4 models = Shorkie + DREAM-Atten/CNN/RNN). Per negative
set, the 6-model inner-join is binned by TSS distance (Caudal 5 bins, Kita 4 bins); each
(model, neg_set, bin) gives an AUPRC point; a regression line + 95% CI band is drawn per
model. The x-axis tick labels carry per-bin Pos/Neg counts (Shorkie, neg_set 1).

Outputs: reproduced/Figure_7HI_reproduced.png
         recheck/fig7HI_auprc.csv   (panel, dataset, model, distance_bin, mean_pr_auc, n_points)
         + frac-bins (Shorkie >= DREAM-RNN) summary printed/recorded.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, average_precision_score

from shorkie import config

config.load()
ROOT = str(config.path("work_root"))
REPRO = config.repo_root() / "reproduction" / "figure_07"
RECHECK = REPRO / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

MPRA_MODELS = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP = {"DREAM_Atten": "DREAM-Atten", "DREAM_CNN": "DREAM-CNN", "DREAM_RNN": "DREAM-RNN"}
MODEL_COLORS = {"Shorkie": "#2196F3", "DREAM-Atten": "#4CAF50", "DREAM-CNN": "#9C27B0", "DREAM-RNN": "#795548"}
PANELS = [("caudal_etal", "H", "Caudal et al."), ("kita_etal", "I", "Kita et al.")]


def _vizdir(neg):
    return os.path.join(ROOT, "revision_experiments", "eQTL", "viz_new", "results", f"negset_{neg}")


def process_shorkie(exp, neg):
    d = _vizdir(neg)
    o = pd.read_csv(os.path.join(d, f"{exp}_Shorkie_scores.tsv"), sep="\t",
                    usecols=["Position_Gene", "logSED_agg", "label", "distance"])
    o["Position_Gene"] = o["Position_Gene"].astype(str).str.strip()
    o = o.drop_duplicates(["Position_Gene"]); o["Shorkie"] = o["logSED_agg"].abs(); o = o.drop(columns=["logSED_agg"])
    lm = pd.read_csv(os.path.join(d, f"{exp}_Shorkie_LM_scores.tsv"), sep="\t", usecols=["Position_Gene", "LLR", "label"])
    lm["Position_Gene"] = lm["Position_Gene"].astype(str).str.strip()
    lm = lm.drop_duplicates(["Position_Gene"]); lm["Shorkie_LM"] = lm["LLR"].abs(); lm = lm.drop(columns=["LLR", "label"])
    ri = pd.read_csv(os.path.join(d, f"{exp}_Shorkie_Random_Init_scores.tsv"), sep="\t", usecols=["Position_Gene", "logSED_agg"])
    ri["Position_Gene"] = ri["Position_Gene"].astype(str).str.strip()
    ri = ri.drop_duplicates(["Position_Gene"]); ri["Shorkie_Random_Init"] = ri["logSED_agg"].abs(); ri = ri.drop(columns=["logSED_agg"])
    m = o.merge(lm, on="Position_Gene", how="inner").merge(ri, on="Position_Gene", how="inner")
    m["label"] = m["label"].astype(int)
    return m[["Position_Gene", "Shorkie_LM", "Shorkie", "Shorkie_Random_Init", "label", "distance"]]


def process_mpra(model, neg, base):
    md = os.path.join(base, model)
    pos = pd.read_csv(os.path.join(md, "final_pos_predictions.tsv"), sep="\t", usecols=["Position_Gene", "logSED"])
    ng = pd.read_csv(os.path.join(md, f"final_neg_predictions_{neg}.tsv"), sep="\t", usecols=["Position_Gene", "logSED"])
    pos["label"], ng["label"] = 1, 0
    df = pd.concat([pos, ng], ignore_index=True)
    df["Position_Gene"] = df["Position_Gene"].astype(str).str.strip()
    df["score"] = df["logSED"].abs()
    return df[["Position_Gene", "score", "label"]]


def mpra_base(exp):
    if exp == "caudal_etal":
        return os.path.join(ROOT, "experiments", "SUM_data_process", "eQTL", "eqtl_MPRA_modeals_eval",
                            "eQTL_MPRA_models_eval_caudal_etal", "results")
    return os.path.join(ROOT, "experiments", "SUM_data_process", "eQTL", "eqtl_MPRA_modeals_eval",
                        "eQTL_MPRA_models_eval_kita_etal_select", "results")


def metrics_by_bin(df, model, bins, labels, bmap):
    out = []
    df = df.dropna(subset=["score", "distance"]).copy()
    df["distance_bin"] = pd.cut(df["distance"], bins=bins, labels=labels)
    for b in labels:
        seg = df[(df.distance_bin == b) & (df.score > 0)]
        if len(seg) < 10 or seg["label"].nunique() < 2:
            continue
        y, s = seg["label"], seg["score"]
        fpr, tpr, _ = roc_curve(y, s)
        out.append(dict(model=model, distance_bin=b, distance=bmap[b],
                        n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
                        roc_auc=auc(fpr, tpr), pr_auc=average_precision_score(y, s)))
    return out


def collect(exp):
    if exp == "caudal_etal":
        bins = [0, 1000, 2000, 3000, 4000, 5000]; labels = ["0-1kb", "1kb-2kb", "2kb-3kb", "3k-4kb", "4k-5kb"]
    else:
        bins = [0, 500, 1200, 2000, 3000]; labels = ["0-0.5kb", "0.5-1.2kb", "1.2kb-2kb", "2kb-3kb"]
    bmap = dict(zip(labels, bins[1:]))
    base = mpra_base(exp)
    model_cols = ["Shorkie"] + [MPRA_NAME_MAP[m] for m in MPRA_MODELS]
    rows = []
    for ns in range(1, 5):
        cdf = process_shorkie(exp, ns)
        for m in MPRA_MODELS:
            md = process_mpra(m, ns, base).rename(columns={"score": MPRA_NAME_MAP[m]})
            cdf = cdf.merge(md[["Position_Gene", MPRA_NAME_MAP[m]]], on="Position_Gene", how="inner")
        cdf = cdf.dropna(subset=model_cols)
        for mc in model_cols:
            t = cdf[["Position_Gene", "label", "distance"]].copy(); t["score"] = cdf[mc]
            rows += metrics_by_bin(t, mc, bins, labels, bmap)
    return pd.DataFrame(rows), labels, bmap


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    csv_rows = []
    for ax, (exp, panel, nm) in zip(axes, PANELS):
        df, labels, bmap = collect(exp)
        # x-tick labels with Pos/Neg counts (first occurrence per distance = Shorkie neg1)
        counts = df.groupby("distance")[["n_pos", "n_neg"]].first()
        xticks = []
        for lbl in labels:
            dv = bmap[lbl]
            np_, nn = (int(counts.loc[dv, "n_pos"]), int(counts.loc[dv, "n_neg"])) if dv in counts.index else (0, 0)
            xticks.append(f"{lbl}\nPos: {np_}\nNeg: {nn}")
        order = ["Shorkie", "DREAM-Atten", "DREAM-CNN", "DREAM-RNN"]
        sub = df[df.model.isin(order)]
        sns.scatterplot(data=sub, x="distance", y="pr_auc", hue="model", style="model",
                        marker="o", s=55, edgecolor="w",
                        palette=MODEL_COLORS, hue_order=order, ax=ax, legend=False)
        for m in order:
            sns.regplot(data=sub[sub.model == m], x="distance", y="pr_auc",
                        scatter=False, label=m, color=MODEL_COLORS[m], ax=ax)
        ax.set_xticks(list(bmap.values())); ax.set_xticklabels(xticks, fontsize=10)
        ax.set_xlabel("TSS Distance Bin", fontsize=13); ax.set_ylabel("AUPRC", fontsize=13)
        ax.set_title(f"{panel}  {nm}: AUPRC by Distance Bin", fontsize=14)
        ax.legend(ncol=2, loc="upper center", fontsize=10, frameon=False)
        # CSV + frac-bins (Shorkie >= DREAM-RNN, mean per bin)
        mean_bin = sub.groupby(["model", "distance_bin"], observed=True)["pr_auc"].mean().unstack("model")
        common = mean_bin.dropna(subset=["Shorkie", "DREAM-RNN"])
        frac = float((common["Shorkie"] >= common["DREAM-RNN"]).mean()) if len(common) else float("nan")
        print(f"[{panel} {exp}] frac bins Shorkie>=DREAM-RNN AUPRC = {frac:.3f} (n_bins={len(common)})")
        for (mdl, b), g in sub.groupby(["model", "distance_bin"], observed=True):
            csv_rows.append(dict(panel=panel, dataset=exp, model=mdl, distance_bin=str(b),
                                 mean_pr_auc=round(g["pr_auc"].mean(), 4), n_points=len(g)))
    fig.suptitle("Figure 7 H/I (reproduced) — AUPRC by TSS-distance bin (Shorkie vs DREAM)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPRO / "reproduced" / "Figure_7HI_reproduced.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)
    pd.DataFrame(csv_rows).to_csv(RECHECK / "fig7HI_auprc.csv", index=False)
    print("saved", RECHECK / "fig7HI_auprc.csv")


if __name__ == "__main__":
    main()
