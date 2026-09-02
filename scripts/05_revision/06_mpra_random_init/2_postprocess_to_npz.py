#!/usr/bin/env python3
"""Revision experiment 06, step 2 — convert the Shorkie_Random_Init MPRA HDF5 output into
the per-context NPZ layout the Figure-6 loaders expect.

This is a thin driver, not a reimplementation: it imports ``main()`` from the published
post-processing scripts
``scripts/04_analysis/shorkie/mpra/3_process_hdf5_logsed/1_analyze_MPR_given_gene_{single_index,dual_indices}.py``
and calls them with the Shorkie_Random_Init paths. Reusing the published code is the point
-- if the two models were post-processed differently the comparison would be worthless.

Output layout matches ``reproduction/figure_06/recheck/mpra_common.py``:

    results/npz/<seq_type>/<SYMBOL>_<ORF>_{pos|neg}_outputs/<ORF>_context_<i>_<ctx_id>.npz

CPU only. Run after 1_run_mpra_random_init.sh has produced the fold HDF5 files.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

from shorkie import config

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
SINGLE_TYPES = ["yeast_seqs", "high_exp_seqs", "low_exp_seqs",
                "challenging_seqs", "all_random_seqs"]
DUAL_TYPES = ["all_SNVs_seqs", "motif_perturbation", "motif_tiling_seqs"]
STAGE3 = "scripts/04_analysis/shorkie/mpra/3_process_hdf5_logsed"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process the Shorkie_Random_Init MPRA HDF5 output into NPZ.")
    parser.add_argument("--model_dir", default=None,
                        help="Model whose fold dirs hold the scores "
                             "[default: config models.shorkie_random_init]")
    parser.add_argument("--run_name", default="MPRA_random_init",
                        help="The relative -o passed to hound_MPRA_folds.py in step 1")
    parser.add_argument("--t0_indices", default=None,
                        help="T0 index TSV [default: <work_root>/experiments/"
                             "SUM_data_process/MPRA/results/t0_indices.tsv]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--num_folds", type=int, default=None,
                        help="Folds to average [default: config models.num_folds]")
    return parser.parse_args()

def load_stage3(repo, dual):
    """Import the published post-processing module by path (its filename starts with a
    digit, so it cannot be imported as a normal module name)."""
    name = ("1_analyze_MPR_given_gene_dual_indices.py" if dual
            else "1_analyze_MPR_given_gene_single_index.py")
    path = repo / STAGE3 / name
    if not path.exists():
        sys.exit(f"error: published post-processing script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"stage3_{'dual' if dual else 'single'}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    model_dir = Path(args.model_dir) if args.model_dir \
        else Path(config.path("models.shorkie_random_init"))
    n_folds = args.num_folds if args.num_folds else int(config.get("models.num_folds", 8))
    t0_indices = Path(args.t0_indices) if args.t0_indices else \
        Path(config.path("work_root")) / "experiments/SUM_data_process/MPRA/results/t0_indices.tsv"

    if not model_dir.exists():
        sys.exit(f"error: {model_dir} not found -- fetch it with "
                 "`bash data/download.sh --models random_init`")
    if not t0_indices.exists():
        sys.exit(f"error: T0 index file not found: {t0_indices} (see "
                 f"{STAGE3}/0_get_T0_rnaseq_index.py)")

    done = skipped = 0
    for dual, seq_types in ((False, SINGLE_TYPES), (True, DUAL_TYPES)):
        mod = load_stage3(repo, dual)
        for seq_type in seq_types:
            for genes, tag in ((POS_GENES, "pos"), (NEG_GENES, "neg")):
                for symbol, orf in genes.items():
                    # hound_MPRA.py writes '<out_dir>/sed.h5' where out_dir is
                    # '<model>/train/f{fold}c{cross}/<relative -o>'  (hound_MPRA_folds.py:228,
                    # hound_MPRA.py:386) — matching the published post-processor's
                    # f"{base}/f{fold}c0/MPRA/{seq_type}/{gene}_pos/sed.h5".
                    rel = Path(args.run_name) / seq_type / f"{symbol}_{tag}"
                    files = [model_dir / "train" / f"f{f}c0" / rel / "sed.h5"
                             for f in range(n_folds)]
                    files = [str(f) for f in files if f.exists()]
                    if not files:
                        print(f"SKIPPED: no sed.h5 under "
                              f"{model_dir}/train/f*c0/{rel}", file=sys.stderr)
                        skipped += 1
                        continue
                    target = (out_dir / "npz" / seq_type /
                              f"{symbol}_{orf}_{tag}_outputs")
                    target.mkdir(parents=True, exist_ok=True)
                    # The single-index module exposes main(); the dual-indices module
                    # exposes main_from_files() and has no main() at all.
                    entry = getattr(mod, "main", None) or getattr(mod, "main_from_files")
                    try:
                        entry(files, symbol, orf, str(t0_indices), str(target))
                        done += 1
                    except Exception as e:                     # fail-soft per gene
                        print(f"SKIPPED: {seq_type}/{symbol} ({e})", file=sys.stderr)
                        skipped += 1

    print(f"\nprocessed {done} gene x category combinations, skipped {skipped}")
    if done == 0:
        sys.exit("error: nothing was processed -- check that step 1 completed and that "
                 "--run_name matches the -o it used")
    print(f"NPZ tree: {out_dir/'npz'}")

if __name__ == "__main__":
    main()
