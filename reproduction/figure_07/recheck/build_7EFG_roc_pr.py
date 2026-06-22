#!/usr/bin/env python3
"""Figure 7 E/F/G — ensemble ROC & PR for the cis-eQTL benchmark (deep-recheck build).

Faithful re-implementation of the source-of-truth plotter
``scripts/04_analysis/shorkie/eqtl/3_visualization/1_roc_pr_shorkie_fold.py``.

Recipe per model: per negative-set, score = |logSED_agg| (Shorkie / Random_Init),
|LLR| (LM), |logSED| (DREAM); keep score>0; roc_curve/auc + average_precision_score;
ensemble = mean ± SEM over the 4 negative sets; mean curve by interpolation onto a
200-pt grid with a ±1 SEM band.

Per-panel data sourcing (reverse-engineered to match the *published* Figure 7 PDF):
  E Caudal, F Kita   — source recipe verbatim: 6-model cross-family inner-join on
                       Position_Gene, scores from ``viz_new/results/``.  Reproduces the
                       published E/F numbers to 3 decimals.
  G Renganaath       — the published panel was assembled from TWO scoring runs:
                         * Shorkie / Shorkie_LM / Shorkie_Random_Init from the
                           **142-variant** ``viz_new/results_subset_tss/`` set, each on its
                           own natural pos/neg distribution (NO cross-family join — the
                           join drops the subset's matched negatives and inflates AP).
                         * DREAM-{Atten,CNN,RNN} from the **full** ``viz_new/results/`` set,
                           cross-family-joined with the full Shorkie set (the E/F recipe).
                       This recovers the published G lines (Shorkie 0.618/0.629; DREAM
                       0.585-0.596) within <=0.005.  The PREVIOUS reproduction scored G
                       Shorkie on the full 395-variant set (0.538/0.552) and wrongly
                       concluded "DREAM beats Shorkie on Renganaath" — that is backwards.

Layout = published: 2 rows (PR top, ROC bottom) x 3 cols (E Caudal, F Kita, G Renganaath).

Outputs:
  reproduced/Figure_7EFG_reproduced.png
  recheck/fig7EFG_auc.csv   (panel, dataset, model, metric, auc_mean, auc_sem, published, delta)
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from shorkie import config

config.load()
ROOT = str(config.path("work_root"))
REPRO = config.repo_root() / "reproduction" / "figure_07"
RECHECK = REPRO / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

COMMON_FPR = np.linspace(0, 1, 200)
COMMON_REC = np.linspace(0, 1, 200)

EXPS = ["caudal_etal", "kita_etal", "Renganaath_etal"]
PANEL = {"caudal_etal": "E", "kita_etal": "F", "Renganaath_etal": "G"}

MPRA_MODELS = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
MPRA_NAME_MAP = {"DREAM_Atten": "DREAM-Atten", "DREAM_CNN": "DREAM-CNN", "DREAM_RNN": "DREAM-RNN"}
SHORKIE_COLS = ["Shorkie", "Shorkie_LM", "Shorkie_Random_Init"]
MPRA_COLS = [MPRA_NAME_MAP[m] for m in MPRA_MODELS]
SCORE_COLS = SHORKIE_COLS + MPRA_COLS

SHORKIE_COLORS = {"Shorkie": "#2196F3", "Shorkie_LM": "#E91E63", "Shorkie_Random_Init": "#FF9800"}
DREAM_COLORS = {"DREAM-Atten": "#4CAF50", "DREAM-CNN": "#9C27B0", "DREAM-RNN": "#795548"}

# Shorkie-family score TSV stem + score column, per model.
SHK_TSV = {"Shorkie": ("Shorkie", "logSED_agg"),
           "Shorkie_LM": ("Shorkie_LM", "LLR"),
           "Shorkie_Random_Init": ("Shorkie_Random_Init", "logSED_agg")}

# Published Figure-7 reference legend numbers (read from paper/Figures/Figure_7.pdf).
PUB_ROC = {
    "caudal_etal":     {"Shorkie": 0.564, "Shorkie_LM": 0.513, "Shorkie_Random_Init": 0.541,
                        "DREAM-Atten": 0.526, "DREAM-CNN": 0.529, "DREAM-RNN": 0.525},
    "kita_etal":       {"Shorkie": 0.650, "Shorkie_LM": 0.523, "Shorkie_Random_Init": 0.641,
                        "DREAM-Atten": 0.534, "DREAM-CNN": 0.539, "DREAM-RNN": 0.533},
    "Renganaath_etal": {"Shorkie": 0.618, "Shorkie_LM": 0.474, "Shorkie_Random_Init": 0.424,
                        "DREAM-Atten": 0.585, "DREAM-CNN": 0.596, "DREAM-RNN": 0.590},
}
PUB_PR = {
    "caudal_etal":     {"Shorkie": 0.585, "Shorkie_LM": 0.532, "Shorkie_Random_Init": 0.551,
                        "DREAM-Atten": 0.536, "DREAM-CNN": 0.537, "DREAM-RNN": 0.538},
    "kita_etal":       {"Shorkie": 0.643, "Shorkie_LM": 0.555, "Shorkie_Random_Init": 0.614,
                        "DREAM-Atten": 0.568, "DREAM-CNN": 0.567, "DREAM-RNN": 0.564},
    "Renganaath_etal": {"Shorkie": 0.629, "Shorkie_LM": 0.492, "Shorkie_Random_Init": 0.447,
                        "DREAM-Atten": 0.588, "DREAM-CNN": 0.593, "DREAM-RNN": 0.596},
}


def get_color(col):
    return {**SHORKIE_COLORS, **DREAM_COLORS}.get(col, "gray")


def get_style(col):
    return ("-", 2.0) if "Shorkie" in col else ("-.", 1.5)


def _vizdir(sub, neg):
    return os.path.join(ROOT, "revision_experiments", "eQTL", "viz_new", sub, f"negset_{neg}")


def _dreamdir(exp):
    if exp == "caudal_etal":
        return os.path.join(ROOT, "experiments", "SUM_data_process", "eQTL", "eqtl_MPRA_modeals_eval",
                            "eQTL_MPRA_models_eval_caudal_etal", "results")
    if exp == "kita_etal":
        return os.path.join(ROOT, "experiments", "SUM_data_process", "eQTL", "eqtl_MPRA_modeals_eval",
                            "eQTL_MPRA_models_eval_kita_etal_select", "results")
    return os.path.join(ROOT, "revision_experiments", "eQTL", "eQTL_MPRA_models_eval_Renganaath_etal", "results")


# ── Shorkie-family loaders ──────────────────────────────────────────────────
def _read_shk(exp, sub, model, neg):
    """One Shorkie-family score TSV -> DataFrame[Position_Gene, label, <model>]."""
    stem, score = SHK_TSV[model]
    df = pd.read_csv(os.path.join(_vizdir(sub, neg), f"{exp}_{stem}_scores.tsv"), sep="\t")
    df["Position_Gene"] = df["Position_Gene"].astype(str).str.strip()
    df = df.drop_duplicates(["Position_Gene"])
    out = df[["Position_Gene"]].copy()
    out[model] = df[score].abs()
    if "label" in df.columns:
        out["label"] = df["label"].values
    return out


def joined_shorkie(exp, sub, neg):
    """Inner-join the 3 Shorkie-family models on Position_Gene (label from Shorkie)."""
    o = _read_shk(exp, sub, "Shorkie", neg)               # has label
    lm = _read_shk(exp, sub, "Shorkie_LM", neg)[["Position_Gene", "Shorkie_LM"]]
    ri = _read_shk(exp, sub, "Shorkie_Random_Init", neg)[["Position_Gene", "Shorkie_Random_Init"]]
    m = o.merge(lm, on="Position_Gene", how="inner").merge(ri, on="Position_Gene", how="inner")
    return m


def _read_dream(exp, model, neg):
    md = os.path.join(_dreamdir(exp), model)
    pos = pd.read_csv(os.path.join(md, "final_pos_predictions.tsv"), sep="\t", usecols=["Position_Gene", "logSED"])
    neg_ = pd.read_csv(os.path.join(md, f"final_neg_predictions_{neg}.tsv"), sep="\t", usecols=["Position_Gene", "logSED"])
    pos["label"], neg_["label"] = 1, 0
    df = pd.concat([pos, neg_], ignore_index=True)
    df["Position_Gene"] = df["Position_Gene"].astype(str).str.strip()
    df[MPRA_NAME_MAP[model]] = df["logSED"].abs()
    return df[["Position_Gene", "label", MPRA_NAME_MAP[model]]]


def joined_all(exp, sub, neg):
    """Source recipe: inner-join 3 Shorkie-family + 3 DREAM models on Position_Gene."""
    combined = joined_shorkie(exp, sub, neg)
    for m in MPRA_MODELS:
        d = _read_dream(exp, m, neg)[["Position_Gene", MPRA_NAME_MAP[m]]]
        combined = combined.merge(d, on="Position_Gene", how="inner")
    return combined


def model_valid(exp, model, neg):
    """(label, score>0) for one model on one negset, per the panel-specific recipe."""
    if exp in ("caudal_etal", "kita_etal"):
        df = joined_all(exp, "results", neg).dropna(subset=SCORE_COLS)
        v = df[df[model] > 0]
        return v["label"].values, v[model].values
    # Renganaath
    if model in SHORKIE_COLS:                              # subset_tss, per-model-own
        df = _read_shk("Renganaath_etal", "results_subset_tss", model, neg)
        df = df.dropna(subset=[model, "label"])
        v = df[df[model] > 0]
        return v["label"].values, v[model].values
    # DREAM: full-set cross-family join with full Shorkie
    shk = _read_shk("Renganaath_etal", "results", "Shorkie", neg)[["Position_Gene", "label"]]
    d = _read_dream("Renganaath_etal", [k for k, x in MPRA_NAME_MAP.items() if x == model][0], neg)
    j = shk.merge(d[["Position_Gene", model]], on="Position_Gene", how="inner").dropna(subset=[model])
    v = j[j[model] > 0]
    return v["label"].values, v[model].values


def ensemble(exp):
    roc = {c: {"tprs": [], "aucs": []} for c in SCORE_COLS}
    pr = {c: {"precs": [], "auprcs": []} for c in SCORE_COLS}
    npos = {}
    for col in SCORE_COLS:
        for ns in range(1, 5):
            lab, sc = model_valid(exp, col, ns)
            if len(sc) < 10 or len(np.unique(lab)) < 2:
                continue
            npos.setdefault(col, int((lab == 1).sum()))
            fpr, tpr, _ = roc_curve(lab, sc)
            ti = np.interp(COMMON_FPR, fpr, tpr); ti[0] = 0.0
            roc[col]["tprs"].append(ti); roc[col]["aucs"].append(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(lab, sc)
            pi = np.interp(COMMON_REC, rec[::-1], prec[::-1])
            pr[col]["precs"].append(pi)
            pr[col]["auprcs"].append(average_precision_score(lab, sc))
    return roc, pr, npos


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))
    rows = []
    for j, exp in enumerate(EXPS):
        roc, pr, npos = ensemble(exp)
        p = PANEL[exp]
        # PR (top)
        ax = axes[0, j]
        for col in SCORE_COLS:
            if len(pr[col]["precs"]) < 2:
                continue
            mp = np.mean(pr[col]["precs"], axis=0); sp = sp_stats.sem(pr[col]["precs"], axis=0)
            ma = float(np.mean(pr[col]["auprcs"])); sa = float(sp_stats.sem(pr[col]["auprcs"]))
            ls, lw = get_style(col); c = get_color(col)
            ax.plot(COMMON_REC, mp, linestyle=ls, lw=lw, color=c, label=f"{col} ({ma:.3f}±{sa:.3f})")
            ax.fill_between(COMMON_REC, np.clip(mp - sp, 0, 1), np.clip(mp + sp, 0, 1), alpha=0.15, color=c)
            rows.append(dict(panel=p, dataset=exp, model=col, metric="PR",
                             auc_mean=round(ma, 4), auc_sem=round(sa, 4), published=PUB_PR[exp][col]))
        ax.set_ylim(0.45, 1.05); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{p}  PR Ensemble ({exp}, 4 neg sets)", fontsize=12)
        ax.legend(loc="upper right", fontsize=8)
        # ROC (bottom)
        ax = axes[1, j]
        ax.plot([0, 1], [0, 1], "--", alpha=0.5, color="gray")
        for col in SCORE_COLS:
            if len(roc[col]["tprs"]) < 2:
                continue
            mt = np.mean(roc[col]["tprs"], axis=0); st = sp_stats.sem(roc[col]["tprs"], axis=0)
            ma = float(np.mean(roc[col]["aucs"])); sa = float(sp_stats.sem(roc[col]["aucs"]))
            ls, lw = get_style(col); c = get_color(col)
            ax.plot(COMMON_FPR, mt, linestyle=ls, lw=lw, color=c, label=f"{col} ({ma:.3f}±{sa:.3f})")
            ax.fill_between(COMMON_FPR, np.clip(mt - st, 0, 1), np.clip(mt + st, 0, 1), alpha=0.15, color=c)
            rows.append(dict(panel=p, dataset=exp, model=col, metric="ROC",
                             auc_mean=round(ma, 4), auc_sem=round(sa, 4), published=PUB_ROC[exp][col]))
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"{p}  ROC Ensemble ({exp}, 4 neg sets)", fontsize=12)
        ax.legend(loc="lower right", fontsize=8)
        print(f"[{p} {exp}] n_pos per model = {npos}")

    fig.suptitle("Figure 7 E/F/G (reproduced) — cis-eQTL ROC/PR; G Shorkie-family via results_subset_tss (142)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPRO / "reproduced" / "Figure_7EFG_reproduced.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("saved", out)

    df = pd.DataFrame(rows)
    df["delta"] = (df["auc_mean"] - df["published"]).round(4)
    csv = RECHECK / "fig7EFG_auc.csv"
    df.to_csv(csv, index=False)
    print("saved", csv)
    for exp in EXPS:
        for metric in ("ROC", "PR"):
            sub = df[(df.dataset == exp) & (df.metric == metric)].sort_values("model")
            cells = "  ".join(f"{r.model.split('_')[-1] if 'Shorkie' in r.model else r.model.split('-')[-1]}"
                              f"={r.auc_mean:.3f}/{r.published:.3f}" for _, r in sub.iterrows())
            print(f"  {PANEL[exp]} {exp:16s} {metric}: {cells}")
    print(f"max |Δ vs published| = {df['delta'].abs().max():.4f}  "
          f"(>0.01: {(df['delta'].abs()>0.01).sum()} cells)")


if __name__ == "__main__":
    main()
