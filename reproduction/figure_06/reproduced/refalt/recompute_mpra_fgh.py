#!/usr/bin/env python3
"""
Reproduce Figure 6 panels F / G / H (and E) of the Shorkie paper from the
on-disk Shorkie MPRA logSED prediction NPZ files.

This is a faithful re-implementation of the two original analysis scripts:

  * scripts/04_analysis/shorkie/mpra/5_mpra_viz/MPRA_scatter_regression_dual_trim.py
        -> panels 6F (SNV), 6G (motif perturbation), 6H (motif tiling)
        Variant-effect Delta:  Shorkie predicted logSED (Alt - Ref) vs
        measured MAUDE expression (Alt - Ref).

  * scripts/04_analysis/shorkie/mpra/5_mpra_viz/MPRA_scatter_regression_single.py
        -> panel 6E (challenging sequences)
        Absolute:  Shorkie predicted logSED vs absolute MAUDE expression
        (single index per sequence, NOT an Alt - Ref difference).

METHOD (exactly as in the originals)
-------------------------------------
Ground truth (MAUDE expression):
    The file `filtered_test_data_with_MAUDE_expression.txt` is read with a
    tab-delimited csv.reader; column 0 is the tagged DNA sequence, column 1 is
    the MAUDE expression float.  GROUND_TRUTH_EXP[i] = float(line[i][1]).

Per-sequence-type index mapping:
    Each seq_type CSV (all_SNVs_seqs / motif_perturbation / motif_tiling_seqs)
    carries `alt_pos` and `ref_pos` columns -- 0/1-based row indices into
    GROUND_TRUTH_EXP.  The released NPZ files only contain the 1000-sequence
    *subsample* selected by `fix/<seq_type>_sample_ids.tsv` (column
    `original_row_id`, 1-indexed).  The subsample is built as
        subset = CSV.iloc[original_row_id - 1]
    and the NPZ rows are stored in `seq1..seqN` order, which is exactly the
    `sample_id` order of that TSV (verified: npz['seq_ids'] == tsv['sample_id']).
    Therefore NPZ row k aligns positionally with subset row k.

Predictions (per gene):
    For each gene there are ~11 context NPZ files (one per insertion offset,
    `_ctx{0..10}_`).  Each NPZ stores logSED_ALT_ORIG, logSED_REF_ORIG and
    logSED, each shape (Nseq, 384) and dtype float16.  We CAST TO float64
    before any arithmetic / averaging (float16 accumulation is lossy).
      F/G/H:  logSED_delta = logSED_ALT_ORIG - logSED_REF_ORIG  (== stored logSED)
              seq_means    = mean over the 384 output bins (axis=1)
      E    :  seq_means    = mean over the 384 bins of the single `logSED`
    seq_means is averaged across the ~11 context files  -> per-gene vector,
    then averaged across all available genes               -> aggregated_pred.

Ground-truth aggregate:
      F/G/H:  aggregated_gt[k] = GROUND_TRUTH_EXP[alt_k] - GROUND_TRUTH_EXP[ref_k]
      E    :  aggregated_gt[k] = GROUND_TRUTH_EXP[pos_k]      (absolute)

Filtering:
      F/G/H:  keep finite (gt & pred), then drop exact zeros
              non_zero_mask = (gt != 0) & (pred != 0)  (matches the original).
      E    :  keep finite only (the original single-index script does not
              drop zeros for the challenging panel).

Correlation: scipy.stats.pearsonr / spearmanr on the surviving points.

Published rendered-panel anchors (R baked into legends, verified on disk):
      6F  SNV                  Pearson 0.539  Spearman 0.420
      6G  motif perturbation   Pearson 0.819  Spearman 0.745
      6H  motif tiling         Pearson 0.561  Spearman 0.546
      6E  challenging          Pearson 0.695  Spearman 0.692
"""

import os
import csv
import glob
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------- #
# Absolute paths (this is a verification/reproduction script).                #
# --------------------------------------------------------------------------- #
WORK_ROOT = "/scratch4/ssalzbe1/khchao/Yeast_ML"
DATA_ROOT = f"{WORK_ROOT}/data/MPRA"
NPZ_ROOT = (
    f"{WORK_ROOT}/experiments/SUM_data_process/MPRA/MPRA_promoter_seqs/"
    f"results/single_measurement_stranded/all_seq_types"
)
OUT_DIR = "/scratch4/ssalzbe1/khchao/shorkie-paper/reproduction/figure_06/reproduced/refalt"
os.makedirs(OUT_DIR, exist_ok=True)

MAUDE_FILE = f"{DATA_ROOT}/filtered_test_data_with_MAUDE_expression.txt"

# Published Pearson anchors keyed by seq_type.
PUBLISHED_PEARSON = {
    "all_SNVs_seqs": 0.539,
    "motif_perturbation": 0.819,
    "motif_tiling_seqs": 0.561,
    "challenging_seqs": 0.695,
}
PANEL_NAME = {
    "all_SNVs_seqs": "6F",
    "motif_perturbation": "6G",
    "motif_tiling_seqs": "6H",
    "challenging_seqs": "6E",
}
SEQ_NAME = {
    "all_SNVs_seqs": "SNV Sequences",
    "motif_perturbation": "Motif Perturbation Sequences",
    "motif_tiling_seqs": "Motif Tiling Sequences",
    "challenging_seqs": "Challenging Sequences",
}

# Gene symbol -> systematic name, with strand (pos/neg), exactly as in the
# original scripts.  Genes whose output dir is absent are silently skipped
# (matches process_gene_plots returning None).
POS_GENES = {
    "GPM3": "YOL056W", "SLI1": "YGR212W", "VPS52": "YDR484W",
    "YMR160W": "YMR160W", "MRPS28": "YDR337W", "YCT1": "YLL055W",
    "RDL2": "YOR286W", "PHS1": "YJL097W", "RTC3": "YHR087W", "MSN4": "YKL062W",
}
NEG_GENES = {
    "COA4": "YLR218C", "ERI1": "YPL096C-A", "RSM25": "YIL093C",
    "ERD1": "YDR414C", "MRM2": "YGL136C", "SNT2": "YGL131C",
    "CSI2": "YOL007C", "RPE1": "YJL121C", "PKC1": "YBL105C",
    "AIM11": "YER093C-A", "MAE1": "YKL029C", "MRPL1": "YDR116C",
}


# --------------------------------------------------------------------------- #
# Ground truth                                                                #
# --------------------------------------------------------------------------- #
def load_ground_truth():
    with open(MAUDE_FILE) as f:
        lines = list(csv.reader(f, delimiter="\t"))
    exp = np.array([float(line[1]) for line in lines], dtype=np.float64)
    return exp


def get_ctx_position(fname):
    """Insertion offset from `_ctx{N}_` or `_context_{N}_`; 100 + N*10.

    The Delta panels (F/G/H) name files `{gene}_ctx{N}_...`; the challenging
    panel (E) names them `{gene}_context_{N}_...`.  Order is all that matters.
    """
    base = os.path.basename(fname)
    m = re.search(r"_ctx(\d+)_", base) or re.search(r"_context_(\d+)_", base)
    return 100 + int(m.group(1)) * 10 if m else 0


def per_gene_pred(input_dir, target_gene, key_alt, key_ref, ctx_glob):
    """Average over the context NPZ files for one gene.

    key_alt/key_ref: NPZ keys to subtract for the Delta panels, or
    (key_alt, None) to use a single absolute key (challenging).
    ctx_glob: filename pattern relative to target_gene ('ctx' or 'context_').
    Returns the per-sequence mean vector (float64) or None if no files.
    """
    npz_files = sorted(
        glob.glob(os.path.join(input_dir, f"{target_gene}_{ctx_glob}*.npz")),
        key=get_ctx_position,
    )
    if not npz_files:
        return None
    per_ctx = []
    for fp in npz_files:
        d = np.load(fp)
        if key_ref is None:
            arr = d[key_alt].astype(np.float64)
        else:
            arr = d[key_alt].astype(np.float64) - d[key_ref].astype(np.float64)
        per_ctx.append(np.mean(arr, axis=1))           # mean over 384 bins
    return np.mean(np.array(per_ctx), axis=0)           # mean over contexts


def aggregate_predictions(seq_type, key_alt, key_ref, ctx_glob="ctx"):
    """Mean of per-gene prediction vectors across all available genes."""
    gene_vecs = []
    for genes, tag in ((POS_GENES, "pos"), (NEG_GENES, "neg")):
        for sym, sysname in genes.items():
            input_dir = os.path.join(NPZ_ROOT, seq_type, f"{sym}_{sysname}_{tag}_outputs")
            v = per_gene_pred(input_dir, sysname, key_alt, key_ref, ctx_glob)
            if v is not None:
                gene_vecs.append(v)
    if not gene_vecs:
        raise RuntimeError(f"No gene NPZ found for {seq_type}")
    return np.mean(np.array(gene_vecs), axis=0), len(gene_vecs)


# --------------------------------------------------------------------------- #
# Index lists (subsampled, in NPZ row order)                                  #
# --------------------------------------------------------------------------- #
def subsample_pairs(csv_name, sample_ids_name):
    """Return list of (alt_pos, ref_pos) in NPZ (sample_id) order."""
    df = pd.read_csv(f"{DATA_ROOT}/test_subset_ids/{csv_name}")
    ids = pd.read_csv(f"{DATA_ROOT}/test_subset_ids/fix/{sample_ids_name}", sep="\t")
    idx = ids["original_row_id"].astype(int).values - 1
    alt = list(df["alt_pos"]); ref = list(df["ref_pos"])
    pairs = list(zip(alt, ref))
    return [pairs[i] for i in idx]


def subsample_single(csv_name, sample_ids_name):
    """Return array of absolute `pos` indices in NPZ (sample_id) order."""
    df = pd.read_csv(f"{DATA_ROOT}/test_subset_ids/{csv_name}")
    ids = pd.read_csv(f"{DATA_ROOT}/test_subset_ids/fix/{sample_ids_name}", sep="\t")
    idx = ids["original_row_id"].astype(int).values - 1
    return df.iloc[idx]["pos"].values


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def scatter(gt, pred, seq_type, pear, spear, n, x_is_pred, out_png):
    """Match published convention: x = Shorkie predicted, y = expression."""
    plt.figure(figsize=(5, 5))
    if x_is_pred:
        x, y = pred, gt
        xlabel = ("Shorkie predicted logSED" if seq_type == "challenging_seqs"
                  else "Shorkie predicted logSED differences (Alt - Ref)")
        ylabel = ("Average expression levels (YFP fluorescence)" if seq_type == "challenging_seqs"
                  else "Average expression levels differences (YFP fluorescence, Alt - Ref)")
    else:
        x, y = gt, pred
        xlabel, ylabel = "ground truth", "prediction"
    plt.scatter(x, y, s=15, alpha=0.6, label=SEQ_NAME[seq_type])
    m, b = np.polyfit(x, y, 1)
    xr = np.linspace(np.min(x), np.max(x), 100)
    plt.plot(xr, m * xr + b, "r-", linewidth=2,
             label=f"Pearson: {pear:.3f}, Spearman: {spear:.3f}")
    plt.xlabel(xlabel, fontsize=8.5)
    plt.ylabel(ylabel, fontsize=9)
    plt.title(f"{PANEL_NAME[seq_type]}  {SEQ_NAME[seq_type]}\n"
              f"Aggregated across all genes (n={n})")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    gt_exp = load_ground_truth()
    print(f"Ground truth sequences: {len(gt_exp)}  "
          f"(min={gt_exp.min():.4f}, max={gt_exp.max():.4f})")

    rows = []

    # ---- Delta panels: 6F, 6G, 6H ----
    delta_specs = [
        ("all_SNVs_seqs", "all_SNVs_seqs.csv", "all_SNVs_seqs_sample_ids.tsv"),
        ("motif_perturbation", "motif_perturbation.csv", "motif_perturbation_sample_ids.tsv"),
        ("motif_tiling_seqs", "motif_tiling_seqs.csv", "motif_tiling_seqs_sample_ids.tsv"),
    ]
    for seq_type, csv_name, ids_name in delta_specs:
        pred, n_genes = aggregate_predictions(seq_type, "logSED_ALT_ORIG", "logSED_REF_ORIG")
        pairs = subsample_pairs(csv_name, ids_name)
        gt = np.array([gt_exp[a] - gt_exp[r] for a, r in pairs], dtype=np.float64)
        if len(gt) != len(pred):
            raise RuntimeError(f"{seq_type}: gt len {len(gt)} != pred len {len(pred)}")

        finite = np.isfinite(gt) & np.isfinite(pred)
        gt_f, pred_f = gt[finite], pred[finite]
        nz = (gt_f != 0) & (pred_f != 0)
        gt_f, pred_f = gt_f[nz], pred_f[nz]
        pear, _ = pearsonr(gt_f, pred_f)
        spear, _ = spearmanr(gt_f, pred_f)
        n = len(gt_f)

        out_png = os.path.join(OUT_DIR, f"scatter_{seq_type}.png")
        scatter(gt_f, pred_f, seq_type, pear, spear, n, True, out_png)

        pub = PUBLISHED_PEARSON[seq_type]
        rows.append([PANEL_NAME[seq_type], seq_type, pear, spear, n, pub, pear - pub])
        print(f"[{PANEL_NAME[seq_type]}] {seq_type}: genes={n_genes} n={n} "
              f"Pearson={pear:.3f} Spearman={spear:.3f} "
              f"(published {pub}, delta {pear - pub:+.3f})")

    # ---- Absolute panel: 6E (challenging) ----
    seq_type = "challenging_seqs"
    pred, n_genes = aggregate_predictions(seq_type, "logSED", None, ctx_glob="context_")
    pos = subsample_single("challenging_seqs.csv", "challenging_seqs_sample_ids.tsv")
    gt = gt_exp[pos].astype(np.float64)
    if len(gt) != len(pred):
        raise RuntimeError(f"{seq_type}: gt len {len(gt)} != pred len {len(pred)}")
    finite = np.isfinite(gt) & np.isfinite(pred)
    gt_f, pred_f = gt[finite], pred[finite]
    pear, _ = pearsonr(gt_f, pred_f)
    spear, _ = spearmanr(gt_f, pred_f)
    n = len(gt_f)
    out_png = os.path.join(OUT_DIR, f"scatter_{seq_type}.png")
    scatter(gt_f, pred_f, seq_type, pear, spear, n, True, out_png)
    pub = PUBLISHED_PEARSON[seq_type]
    rows.append([PANEL_NAME[seq_type], seq_type, pear, spear, n, pub, pear - pub])
    print(f"[{PANEL_NAME[seq_type]}] {seq_type}: genes={n_genes} n={n} "
          f"Pearson={pear:.3f} Spearman={spear:.3f} "
          f"(published {pub}, delta {pear - pub:+.3f})")

    # ---- write CSV (ordered 6F, 6G, 6H, 6E) ----
    order = {"6F": 0, "6G": 1, "6H": 2, "6E": 3}
    rows.sort(key=lambda r: order[r[0]])
    out_csv = os.path.join(OUT_DIR, "mpra_fgh_R.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "seq_type", "pearson_repro", "spearman_repro",
                    "n", "published_pearson", "delta"])
        for panel, st, pe, sp, nn, pub, dl in rows:
            w.writerow([panel, st, f"{pe:.4f}", f"{sp:.4f}", nn, pub, f"{dl:.4f}"])
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
