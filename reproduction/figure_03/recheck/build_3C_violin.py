#!/usr/bin/env python3
"""Figure 3C — bin-level Pearson's R distribution by track type, as the PUBLISHED
split violin (not the KDE the reproduction previously rendered).

Faithful to the source-of-truth figure code
``scripts/03_eval/supervised/track_prediction_eval/2_bin_gene_level_metrics/
1_bin_level_freq_viz.py::plot_box_violin()``:
  * models: Shorkie = ``self_supervised_unet_small_bert_drop``;
    Shorkie_Random_Init = the **lr=5e-4** supervised variant
    ``supervised_unet_small_bert_drop_variants/learning_rate_0.0005`` — this is
    what the figure plots (the manuscript-text "0.67" is a different plain
    supervised baseline; the figure's RNA-Seq Random median is 0.703).
  * aggregation: median per identifier across the 8 folds.
  * split violin, inner="quart", palette {Shorkie:#377eb8, Random:#ff7f00},
    x-order RNA-Seq / 1000 strains RNA-Seq / ChIP-MNase / ChIP-exo, detailed
    x-labels "(n=..)\\nShorkie med=..\\nRandom med=..".

Outputs ``reproduced/Figure_3C_reproduced.png`` and the 8 medians to
``recheck/fig3C_violin_medians.csv``.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from shorkie import config
config.load()
WORK = str(config.path("work_root"))
REPRO = config.repo_root() / "reproduction" / "figure_03"
BASE = f"{WORK}/seq_experiment/exp_histone__chip_exo__rna_seq_no_norm_5215_tracks/16bp"
SHK = f"{BASE}/self_supervised_unet_small_bert_drop"
RND = f"{BASE}/supervised_unet_small_bert_drop_variants/learning_rate_0.0005"

TRAIN_LABEL = {"supervised": "Shorkie_Random_Init", "self_supervised": "Shorkie"}
PALETTE = {"Shorkie": "#377eb8", "Shorkie_Random_Init": "#ff7f00"}


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


def load_bin():
    rows = []
    for ttype, root in [("supervised", RND), ("self_supervised", SHK)]:
        for fold in range(8):
            p = f"{root}/train/f{fold}c0/eval/acc.txt"
            try:
                d = pd.read_csv(p, sep="\t")
            except FileNotFoundError:
                print(f"missing {p}", file=sys.stderr)
                continue
            d["track_type"] = d["description"].apply(categorize)
            d["training_type"] = ttype
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    metric = "pearsonr"
    label = "Pearson's R"
    df = load_bin()

    # median per identifier across folds
    df_agg = (df.groupby(["track_type", "training_type", "identifier"])[metric]
              .median().reset_index())
    df_agg = df_agg[df_agg["track_type"] != "Other"]
    df_agg["training_type"] = df_agg["training_type"].map(TRAIN_LABEL)

    counts = df_agg.groupby("track_type")["identifier"].nunique()
    medians = df_agg.groupby(["track_type", "training_type"])[metric].median()

    def detailed(t):
        c = counts.get(t, 0)
        ms = medians.get((t, "Shorkie"), np.nan)
        mr = medians.get((t, "Shorkie_Random_Init"), np.nan)
        return f"{t}\n(n={c})\nShorkie med={ms:.3f}\nRandom med={mr:.3f}"

    # RNA-Seq first, then the rest alphabetically (matches the published x-order)
    uniq = sorted(df_agg["track_type"].unique())
    if "RNA-Seq" in uniq:
        uniq.remove("RNA-Seq")
        uniq = ["RNA-Seq"] + uniq
    order_detailed = [detailed(t) for t in uniq]
    df_agg["track_type_detailed"] = df_agg["track_type"].apply(detailed)

    plt.figure(figsize=(9.2, 6.4))
    sns.violinplot(
        data=df_agg, x="track_type_detailed", y=metric, hue="training_type",
        hue_order=["Shorkie", "Shorkie_Random_Init"], split=True, inner="quart",
        palette=PALETTE, order=order_detailed, cut=0,
    )
    plt.title(f"{label} Distribution by Track Type (Violin)", fontsize=16)
    plt.xlabel("Track Type", fontsize=14)
    plt.ylabel(label, fontsize=14)
    plt.legend(title="Model", loc="upper right", fontsize=12, title_fontsize=14)
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    out = REPRO / "reproduced" / "Figure_3C_reproduced.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print("saved", out)

    # record the 8 medians
    recs = []
    for t in uniq:
        recs.append(dict(track_type=t, n=int(counts.get(t, 0)),
                         Shorkie_median=round(float(medians.get((t, "Shorkie"), np.nan)), 4),
                         Random_median=round(float(medians.get((t, "Shorkie_Random_Init"), np.nan)), 4)))
    csv = REPRO / "recheck" / "fig3C_violin_medians.csv"
    pd.DataFrame(recs).to_csv(csv, index=False)
    print("wrote", csv)
    print(pd.DataFrame(recs).to_string(index=False))


if __name__ == "__main__":
    main()
