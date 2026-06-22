#!/usr/bin/env python3
"""Run TomTom (MEME 5.5.7, built from source) to match Shorkie RP TF-MoDISco patterns
to the yeast motif database, reproducing the original pipeline's matching
(scripts/.../motif_lm__RP_TSS/1_map_modisco_pattern_to_meme_db.py).

For each MoDISco pos/neg pattern: take its PPM ('sequence'), trim by the CWM
('contrib_scores') as the original did, write a MEME query file, run TomTom vs
merged_meme_high_conf.meme, and emit recheck/tomtom_RP_matches.tsv
(pattern -> ranked TF matches with q-values).
"""
import sys, subprocess, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F

TOMTOM = Path.home() / "tools" / "meme" / "bin" / "tomtom"
DB = str(F.MEME_HIGH_CONF)
OUTDIR = F.RECHECK / "_tomtom_RP"


def trim_idx(cwm, thr=0.3, pad=2):
    sc = np.sum(np.abs(cwm), axis=1)
    if sc.max() == 0:
        return 0, len(sc)
    w = np.where(sc >= sc.max() * thr)[0]
    return (0, len(sc)) if len(w) == 0 else (max(w.min() - pad, 0), min(w.max() + pad + 1, len(sc)))


def write_query(meme_path):
    mh = F.modisco_h5("gene_exp_motif_test_RP")
    lines = ["MEME version 5", "", "ALPHABET= ACGT", "", "strands: + -", "",
             "Background letter frequencies (from unknown source):",
             " A 0.250 C 0.250 G 0.250 T 0.250", ""]
    tags = []
    with h5py.File(mh, "r") as f:
        for grp in ["pos_patterns", "neg_patterns"]:
            if grp not in f:
                continue
            for pn in sorted(f[grp].keys(), key=lambda x: int(x.split("_")[-1])):
                ppm = np.array(f[grp][pn]["sequence"][:])
                cwm = np.array(f[grp][pn]["contrib_scores"][:])
                s, e = trim_idx(cwm)
                ppm = ppm[s:e]
                ppm = np.clip(ppm, 1e-6, None); ppm = ppm / ppm.sum(axis=1, keepdims=True)
                tag = f"{grp}.{pn}"; tags.append(tag)
                lines.append(f"MOTIF {tag}")
                lines.append(f"letter-probability matrix: alength= 4 w= {ppm.shape[0]} nsites= 20 E= 0")
                for row in ppm:
                    lines.append(" " + " ".join(f"{x:.6f}" for x in row))
                lines.append("")
    Path(meme_path).write_text("\n".join(lines))
    return tags


def main():
    if not TOMTOM.exists():
        print("tomtom not found at", TOMTOM); sys.exit(1)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    qmeme = F.RECHECK / "_modisco_RP_query.meme"
    tags = write_query(qmeme)
    print(f"wrote {len(tags)} query motifs -> {qmeme}")
    cmd = [str(TOMTOM), "-no-ssc", "-oc", str(OUTDIR), "-verbosity", "1",
           "-min-overlap", "5", "-dist", "pearson", "-evalue", "-thresh", "10.0",
           str(qmeme), DB]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    tt = OUTDIR / "tomtom.tsv"
    if not tt.exists():
        print("tomtom.tsv not produced"); sys.exit(1)
    df = pd.read_csv(tt, sep="\t", comment="#")
    df.to_csv(F.RECHECK / "tomtom_RP_matches.tsv", sep="\t", index=False)
    print(f"saved recheck/tomtom_RP_matches.tsv ({len(df)} rows)")
    # quick per-TF lookup for the 12 panel-H TFs
    want = ["RAP1", "FHL1", "SFP1", "REB1", "ABF1", "TBF1", "CBF1", "UME6", "DOT6", "STB3", "TBP", "SPT15"]
    for tf in want:
        hit = df[df["Target_ID"].astype(str).str.upper().str.startswith(tf)]
        if len(hit):
            best = hit.sort_values("q-value").iloc[0]
            print(f"  {tf}: {best['Query_ID']} q={best['q-value']:.3g} target={best['Target_ID']}")
        else:
            print(f"  {tf}: no match")


if __name__ == "__main__":
    main()
