#!/usr/bin/env python3
"""Revision experiment 03 — reconcile the 3,053 new RNA-seq tracks.

Everything needed is already committed: ``minimal_example/sheet.txt`` is the full 5,215-row
targets sheet, and every RNA-seq track identifier encodes its design as
``<GENE>_T<minutes>_S<sample>``. This script parses that and emits the arithmetic that
takes the reader from "8 TFs plus 460 genes" to 3,053 tracks.

Outputs (all under results/):
  * ``atlas_summary.csv``     the partition table that sums to 3,053, plus the other
                              three track groups checked against the Methods text
  * ``atlas_by_gene.csv``     per induced gene: tracks, timepoints covered, replicate depth
  * ``atlas_timepoints.csv``  the two sampling schedules and their occupancy
  * ``verify_revision_03.csv`` PASS/FAIL against the counts stated in the paper

CPU only, seconds. No downloads, no model weights.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from shorkie import config

sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa: E402

# The 8 transcription factors of Methods partition (2), "matched to previously measured
# microarray data". They are not hard-coded as an assumption: the script derives the set
# from the sampling schedule (below) and cross-checks it against this list.
EXPECTED_TF8 = ["ARG80", "CUP2", "GAT4", "MET4", "MSN2", "MSN4", "NDT80", "RPN4"]

# The two induction sampling schedules, in minutes after beta-estradiol addition.
SCHEDULE_TF = (0, 5, 10, 15, 30, 45, 60, 90)        # 8-TF microarray-matched series
SCHEDULE_ATLAS = (0, 5, 10, 20, 40, 70, 120, 180)   # the broader gene panel

# Counts stated in the paper, used as anchors.
PUBLISHED = {
    "RNA-Seq tracks": 3053,
    "1000-RNA-Seq tracks": 1014,
    "ChIP-exo tracks": 1128,
    "ChIP-MNase tracks": 20,
    "total tracks": 5215,
    "genes in the broader panel (attempted)": 460,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconcile the 3,053 induction RNA-seq tracks from the targets sheet."
    )
    parser.add_argument("--sheet", default=None,
                        help="Targets sheet [default: minimal_example/sheet.txt in the repo]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def load_rnaseq(sheet_path):
    d = pd.read_csv(sheet_path, sep="\t")
    r = d[d.group == "RNA-Seq"].copy()
    parts = r.identifier.str.extract(r"^(?P<gene>.+?)_T(?P<timepoint>\d+)_S(?P<sample>\d+)$")
    unparsed = int(parts.gene.isna().sum())
    if unparsed:
        print(f"SKIPPED: {unparsed} RNA-seq identifiers did not match "
              f"<GENE>_T<min>_S<sample>", file=sys.stderr)
    r = pd.concat([r, parts], axis=1).dropna(subset=["gene"])
    r["timepoint"] = r.timepoint.astype(int)
    return d, r

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = Path(args.sheet) if args.sheet else config.repo_root() / "minimal_example" / "sheet.txt"
    if not sheet.exists():
        sys.exit(f"error: targets sheet not found: {sheet}")

    full, r = load_rnaseq(sheet)
    print(f"targets sheet : {sheet}  ({len(full):,} tracks)", flush=True)

    # --- partition genes by their sampling schedule -------------------------------
    schedules = r.groupby("gene")["timepoint"].apply(lambda s: tuple(sorted(set(s))))
    tf_genes = sorted(schedules[schedules == SCHEDULE_TF].index)
    atlas_genes = sorted(g for g in schedules.index if g not in tf_genes)

    if sorted(tf_genes) != sorted(EXPECTED_TF8):
        print(f"NOTE: schedule-derived TF set {tf_genes} differs from the expected "
              f"{EXPECTED_TF8}", file=sys.stderr)

    tf_tracks = r[r.gene.isin(tf_genes)]
    atlas_tracks = r[r.gene.isin(atlas_genes)]

    # --- summary table --------------------------------------------------------------
    # Derive the replicate range rather than quoting it: an earlier draft of the README
    # said "6-12" where the data says 4-12.
    tf_reps = tf_tracks.groupby(["gene", "timepoint"]).size()
    rows = [
        dict(partition="Induction RNA-seq: 8 TF perturbations in replicate",
             detail=f"{', '.join(tf_genes)}; schedule {SCHEDULE_TF} min; "
                    f"{tf_reps.min()}-{tf_reps.max()} replicates per timepoint",
             genes=len(tf_genes), tracks=len(tf_tracks)),
        dict(partition="Induction RNA-seq: broader gene panel",
             detail=f"kinases, phosphatases and other regulators; schedule {SCHEDULE_ATLAS} min",
             genes=len(atlas_genes), tracks=len(atlas_tracks)),
        dict(partition="Induction RNA-seq: TOTAL (group 'RNA-Seq')",
             detail="the 3,053 tracks generated for this study",
             genes=len(tf_genes) + len(atlas_genes), tracks=len(r)),
    ]
    for grp, label in [("1000-RNA-Seq", "Yeast-strain RNA-seq (Caudal et al.)"),
                       ("Other", "ChIP-exo (Rossi et al.)"),
                       ("Chip-MNase", "ChIP-MNase (Rossi et al.)")]:
        rows.append(dict(partition=label, detail=f"targets-sheet group '{grp}'",
                         genes=pd.NA, tracks=int((full.group == grp).sum())))
    rows.append(dict(partition="ALL TRACKS", detail="the supervised training set",
                     genes=pd.NA, tracks=len(full)))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "atlas_summary.csv", index=False)

    # --- per-gene table ---------------------------------------------------------------
    by_gene = (r.groupby("gene")
.agg(tracks=("identifier", "size"),
                      timepoints=("timepoint", "nunique"),
                      first_timepoint=("timepoint", "min"),
                      last_timepoint=("timepoint", "max"))
.reset_index())
    by_gene["partition"] = by_gene.gene.apply(lambda g: "8_TF" if g in tf_genes else "gene_panel")
    by_gene["mean_replicates_per_timepoint"] = (
        by_gene.tracks / by_gene.timepoints).round(2)
    by_gene = by_gene.sort_values(["partition", "tracks"], ascending=[True, False])
    by_gene.to_csv(out_dir / "atlas_by_gene.csv", index=False)

    # --- timepoint occupancy ------------------------------------------------------------
    occ = (r.assign(partition=r.gene.apply(lambda g: "8_TF" if g in tf_genes else "gene_panel"))
.groupby(["partition", "timepoint"])
.agg(tracks=("identifier", "size"), genes=("gene", "nunique"))
.reset_index())
    occ.to_csv(out_dir / "atlas_timepoints.csv", index=False)

    # --- checks ---------------------------------------------------------------------------
    checks = [
        Check(panel="atlas", metric="RNA-Seq tracks",
              reported=PUBLISHED["RNA-Seq tracks"], reproduced=len(r), rtol=0.0, atol=0.0),
        Check(panel="atlas", metric="1000-RNA-Seq tracks",
              reported=PUBLISHED["1000-RNA-Seq tracks"],
              reproduced=int((full.group == "1000-RNA-Seq").sum()), rtol=0.0, atol=0.0),
        Check(panel="atlas", metric="ChIP-exo tracks",
              reported=PUBLISHED["ChIP-exo tracks"],
              reproduced=int((full.group == "Other").sum()), rtol=0.0, atol=0.0),
        Check(panel="atlas", metric="ChIP-MNase tracks",
              reported=PUBLISHED["ChIP-MNase tracks"],
              reproduced=int((full.group == "Chip-MNase").sum()), rtol=0.0, atol=0.0),
        Check(panel="atlas", metric="total tracks",
              reported=PUBLISHED["total tracks"], reproduced=len(full), rtol=0.0, atol=0.0),
        Check(panel="atlas", metric="8 TF perturbations",
              reported=8, reproduced=len(tf_genes), rtol=0.0, atol=0.0),
        Check(panel="atlas", metric="partitions sum to the RNA-Seq total",
              reported=len(r), reproduced=len(tf_tracks) + len(atlas_tracks), rtol=0.0, atol=0.0),
    ]
    write_verdicts(checks, out_dir / "verify_revision_03.csv")
    print(summary(checks))

    print("\n" + summary_df.to_string(index=False))
    full_tp = atlas_tracks.groupby("gene")["timepoint"].nunique()
    print(f"\n8-TF replicates per timepoint: {tf_reps.min()}-{tf_reps.max()}")
    print(f"Broader-panel genes with all {len(SCHEDULE_ATLAS)} timepoints: "
          f"{int((full_tp == len(SCHEDULE_ATLAS)).sum())}")
    print(f"\nGenes in the broader panel: {len(atlas_genes)} released vs "
          f"{PUBLISHED['genes in the broader panel (attempted)']} attempted "
          f"({PUBLISHED['genes in the broader panel (attempted)'] - len(atlas_genes)} "
          f"did not yield released tracks).")
    print(f"wrote 4 tables to {out_dir}")

if __name__ == "__main__":
    main()
