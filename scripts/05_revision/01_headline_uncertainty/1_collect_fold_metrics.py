#!/usr/bin/env python3
"""Revision experiment 01, step 1 — collect every per-fold accuracy record behind
the paper's headline expression-prediction numbers.

Faithful to the source-of-truth figure code that produced Figure 3:
  * bin level        -> ``reproduction/figure_03/recheck/build_3C_violin.py`` and
                        ``build_3DEFG_scatter.py::groups_bin`` -- ``train/f{f}c0/eval/acc.txt``
  * gene/track level -> ``build_3DEFG_scatter.py::load_avg`` --
                        ``gene_level_eval_rc/f{f}c0/{data_type}/{acc.txt,gene_acc.txt}``

Three model trees are read, because the manuscript text and Figure 3C do not use the
same random-init baseline:
  * ``Shorkie``                     -- self_supervised_unet_small_bert_drop
  * ``Shorkie_Random_Init``         -- supervised_unet_small_bert_drop_variants/learning_rate_0.0005
                                       (the LR-optimised baseline Figure 3C plots)
  * ``Shorkie_Random_Init_untuned`` -- supervised_unet_small_bert_drop (lr 1e-4;
                                       the baseline the manuscript text's "0.67" comes from)

Writes a single tidy long-form table, one row per (model, level, track_type, fold,
unit, metric):
    results/fold_metrics.csv

CPU only, ~30 s, no model weights needed -- it reads evaluation artifacts that the
training runs already wrote under ``datasets.supervised_root``.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from shorkie import config

# Model tree -> label. Kept in this order so downstream tables read Shorkie first.
MODEL_TREES = {
    "Shorkie": "self_supervised_unet_small_bert_drop",
    "Shorkie_Random_Init": "supervised_unet_small_bert_drop_variants/learning_rate_0.0005",
    "Shorkie_Random_Init_untuned": "supervised_unet_small_bert_drop",
}

# Gene-level evaluation is stored per data type; these two are the RNA-seq families
# the paper reports. (ChIP-exo / ChIP-MNase gene-level dirs exist but are not part
# of the headline claim.)
GENE_DATA_TYPES = ["RNA-Seq", "1000-RNA-seq"]

def categorize(desc):
    """Bin-level track_type assignment -- identical to build_3C_violin.py::categorize."""
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect per-fold accuracy records for the Figure 3 headline metrics."
    )
    parser.add_argument(
        "--supervised_root",
        default=None,
        help="Root holding the trained model trees [default: config datasets.supervised_root]",
    )
    parser.add_argument(
        "--num_folds", type=int, default=None,
        help="Number of cross-validation folds [default: config models.num_folds]",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory [default:./results next to this script]",
    )
    return parser.parse_args()

def read_bin_level(root, label, num_folds):
    """One row per (fold, track) from the top-level bin evaluation."""
    rows = []
    for fold in range(num_folds):
        path = root / "train" / f"f{fold}c0" / "eval" / "acc.txt"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        d = pd.read_csv(path, sep="\t")
        rows.append(pd.DataFrame({
            "model": label,
            "level": "bin",
            "track_type": d["description"].apply(categorize),
            "fold": fold,
            "unit": d["identifier"],
            "metric": "pearsonr",
            "value": d["pearsonr"],
            "weight": pd.NA,
        }))
    return rows

def read_gene_level(root, label, num_folds):
    """Rows from the reverse-complement gene-level evaluation.

    ``acc.txt``      -> one row per TRACK (metrics pearsonr, pearsonr_norm)
    ``gene_acc.txt`` -> one row per GENE  (metric pearsonr_gene; coverage_norm is
                        carried in ``weight`` because Figure 3G drops the bottom
                        10% of genes by coverage before averaging)
    """
    rows = []
    for fold in range(num_folds):
        for dt in GENE_DATA_TYPES:
            base = root / "gene_level_eval_rc" / f"f{fold}c0" / dt

            track_path = base / "acc.txt"
            if track_path.exists():
                d = pd.read_csv(track_path, sep="\t")
                for metric in ("pearsonr", "pearsonr_norm"):
                    if metric not in d.columns:
                        continue
                    rows.append(pd.DataFrame({
                        "model": label, "level": "gene_track", "track_type": dt,
                        "fold": fold, "unit": d["identifier"], "metric": metric,
                        "value": d[metric], "weight": pd.NA,
                    }))

            gene_path = base / "gene_acc.txt"
            if gene_path.exists():
                d = pd.read_csv(gene_path, sep="\t")
                if "pearsonr_gene" in d.columns:
                    rows.append(pd.DataFrame({
                        "model": label, "level": "gene", "track_type": dt,
                        "fold": fold, "unit": d["gene_id"], "metric": "pearsonr_gene",
                        "value": pd.to_numeric(d["pearsonr_gene"], errors="coerce"),
                        "weight": pd.to_numeric(d.get("coverage_norm"), errors="coerce"),
                    }))
    return rows

def main():
    args = parse_args()
    config.load()

    here = Path(__file__).resolve().parent
    supervised_root = Path(args.supervised_root) if args.supervised_root \
        else config.path("datasets.supervised_root")
    num_folds = args.num_folds if args.num_folds else int(config.get("models.num_folds", 8))
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"supervised_root : {supervised_root}", flush=True)
    print(f"num_folds       : {num_folds}", flush=True)

    frames = []
    for label, rel in MODEL_TREES.items():
        root = supervised_root / rel
        if not root.exists():
            print(f"SKIPPED: model tree not found for {label}: {root}", file=sys.stderr)
            continue
        print(f"reading {label} <- {rel}", flush=True)
        frames += read_bin_level(root, label, num_folds)
        frames += read_gene_level(root, label, num_folds)

    if not frames:
        sys.exit("error: no evaluation artifacts found; check datasets.supervised_root")

    big = pd.concat(frames, ignore_index=True)
    big = big.dropna(subset=["value"])

    out_csv = out_dir / "fold_metrics.csv"
    big.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(big):,} rows)", flush=True)
    print(big.groupby(["model", "level", "metric"])["fold"].agg(
        folds="nunique", rows="size").to_string())

if __name__ == "__main__":
    main()
