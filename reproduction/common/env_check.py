#!/usr/bin/env python3
"""Pre-flight check for reproducing a figure: report which external tools,
Python packages, config keys, and on-disk inputs are available vs missing.

Usage:
    python common/env_check.py            # general check
    python common/env_check.py --figure 1 # also check Figure 1's specific inputs
"""
import argparse
import importlib.util
import shutil

# External CLI tools used across the figures (Figure 1 needs mummer/mash/ete).
TOOLS = ["nucmer", "mummerplot", "show-coords", "mash", "dashing2",
         "pdftoppm", "gsutil", "modisco", "meme", "fimo"]

PYPKGS = ["shorkie", "numpy", "pandas", "matplotlib", "h5py", "sklearn",
          "logomaker", "PIL", "ete3", "Bio"]

# config keys exercised by the figure notebooks
CONFIG_KEYS = [
    "work_root", "models.shorkie_lm", "models.shorkie_finetuned",
    "genome.fasta", "genome.gtf", "datasets.lm_corpus_split_root",
    "lm_experiment_root", "results.lm_eval_logs", "results.modisco_lm",
    "results.umap", "results.ism_scores", "results.eqtl_scores",
    "results.mpra_viz", "results.train_logs",
]


def check_tools():
    print("== external tools ==")
    for t in TOOLS:
        print(f"  {t:14s} {'OK ' + (shutil.which(t) or '') if shutil.which(t) else 'MISSING'}")


def check_pkgs():
    print("== python packages ==")
    for m in PYPKGS:
        ok = importlib.util.find_spec(m) is not None
        print(f"  {m:12s} {'OK' if ok else 'MISSING'}")


def check_config():
    print("== config keys ==")
    try:
        from shorkie import config
    except Exception as e:  # pragma: no cover
        print(f"  cannot import shorkie.config: {e}")
        return
    for k in CONFIG_KEYS:
        val = config.get(k)
        print(f"  {k:34s} {'-> ' + str(val) if val else 'UNSET'}")


def check_figure1():
    from pathlib import Path
    from shorkie import config
    print("== Figure 1 inputs ==")
    lcs = config.path("datasets.lm_corpus_split_root")
    for tier, n in [("r64", 1), ("strains", 80), ("saccharomycetales", 165), ("fungi_1385", 1361)]:
        d = lcs / f"data_{tier}_gtf" / "fasta"
        cnt = len(list(d.glob("*.cleaned.fasta"))) if d.exists() else 0
        print(f"  genomes data_{tier:<16} {cnt} fasta {'OK' if cnt else 'MISSING'} ({d})")
    lmr = config.path("lm_experiment_root") / "test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI"
    for tier, sub in [("r64", "lm_r64_gtf/lm_r64_gtf_unet_small"),
                      ("strains", "lm_strains_gtf/lm_strains_gtf_unet_small"),
                      ("saccharomycetales", "LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small"),
                      ("fungi_1385", "LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small")]:
        t = lmr / sub / "train" / "train.out"
        print(f"  1F train.out {tier:<16} {'OK' if t.exists() else 'MISSING'} ({t})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", type=int, default=None)
    args = ap.parse_args()
    check_tools(); check_pkgs(); check_config()
    if args.figure == 1:
        check_figure1()


if __name__ == "__main__":
    main()
