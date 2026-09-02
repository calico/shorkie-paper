#!/usr/bin/env python3
"""Revision experiment 05, step 2 — redraw the Figure 4D intron schematic so the shaded
region and the motif labels line up.

Two things are wrong with the published panel: the purple block is the intron and the
caption never says so, and the three motif labels sit at eyeballed positions rather than
at the sites they name. In the reproduction's port of the panel
(``reproduction/figure_04/recheck/build_4D.py``) the intron spans x=10..90 while the donor
label sits at x=18 and the branch-point label at x=55 -- the donor motif is drawn well
inside the intron instead of at its 5' boundary, and the branch point is drawn mid-intron
instead of near the 3' end.

Rather than move the labels by eye, this script derives the geometry from the actual
S. cerevisiae R64 annotation and sequence:

  * introns are recovered as the gaps between consecutive exons of each transcript;
  * the branch point is located by scanning each intron for the canonical TACTAAC
    (and, failing that, the degenerate [TC]ACTAA[CT]), and its distance to the 3' splice
    site is measured;
  * the schematic is then drawn to the MEDIAN intron length with the branch point at its
    MEDIAN measured offset, so every element sits where the data says it sits.

Leader lines connect each label to its exact interval, and the intron block is labelled as
such in-figure so the colour needs no explanation.

Outputs ``results/Figure_4D_schematic_revised.png``, ``results/intron_geometry.csv``
(the measured distributions) and ``results/fig4D_caption.md``.

CPU only, ~30 s.
"""
import argparse
import collections
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shorkie import config

DONOR = "GTATGT"
BRANCH = "TACTAAC"
BRANCH_DEGENERATE = re.compile(r"[TC]ACTAA[CT]")
ACCEPTOR = "YAG"

EXON_COLOR = "#c8771f"
INTRON_COLOR = "#8a7fb8"      # the unexplained "purple region" of the published panel
MOTIF_COLOR = "#3b3b3b"

def parse_args():
    parser = argparse.ArgumentParser(description="Redraw the Figure 4D intron schematic.")
    parser.add_argument("--gtf", default=None, help="R64 GTF [default: config genome.gtf]")
    parser.add_argument("--fasta", default=None, help="R64 FASTA [default: config genome.fasta]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def revcomp(s):
    return s.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def load_fasta(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    return seqs

def transcript_exons(gtf_path):
    """{transcript_id: (chrom, strand, [(start0, end),...])} from the GTF exon rows."""
    tx = collections.defaultdict(lambda: [None, None, []])
    for line in open(gtf_path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "exon":
            continue
        m = re.search(r'transcript_id "([^"]+)"', f[8])
        if not m:
            continue
        rec = tx[m.group(1)]
        rec[0], rec[1] = f[0], f[6]
        rec[2].append((int(f[3]) - 1, int(f[4])))
    return tx

def measure_introns(tx, seqs):
    """Intron length and branch-point offset from the 3' splice site, per intron."""
    rows = []
    for tid, (chrom, strand, exons) in tx.items():
        if len(exons) < 2:
            continue
        key = chrom if chrom in seqs else (f"chr{chrom}" if f"chr{chrom}" in seqs else None)
        if key is None:
            continue
        exons = sorted(exons)
        for (_, e1), (s2, _) in zip(exons[:-1], exons[1:]):
            if s2 <= e1:
                continue
            seq = seqs[key][e1:s2].upper()
            if strand == "-":
                seq = revcomp(seq)
            if len(seq) < 20:
                continue
            hits = [m.start() for m in re.finditer(f"(?={BRANCH})", seq)]
            exact = bool(hits)
            if not hits:
                hits = [m.start() for m in BRANCH_DEGENERATE.finditer(seq)]
            # the branch point is the 3'-most match; report its distance to the 3' end
            bp_offset = (len(seq) - hits[-1]) if hits else np.nan
            rows.append(dict(transcript=tid, chrom=chrom, strand=strand, intron_len=len(seq),
                             starts_GTATGT=seq.startswith(DONOR),
                             ends_AG=seq.endswith("AG"),
                             branch_found=bool(hits), branch_exact=exact,
                             branch_offset_from_3ss=bp_offset))
    return pd.DataFrame(rows)

def draw(geom, out_png):
    """Draw the schematic to the median intron length with measured branch placement."""
    med_len = float(geom.intron_len.median())
    med_bp = float(geom.branch_offset_from_3ss.dropna().median())

    # Layout in nucleotide coordinates, with fixed-width flanking exons for context.
    flank = max(40.0, med_len * 0.12)
    x0 = 0.0
    intron_s, intron_e = flank, flank + med_len
    total = intron_e + flank

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(x0 - total * 0.02, total * 1.02)
    ax.set_ylim(-0.9, 1.25)
    ax.axis("off")

    # exons (thick) and intron (thin), so the intron reads as the spliced-out region
    ax.add_patch(plt.Rectangle((x0, 0.30), flank, 0.40, fc=EXON_COLOR, ec="none"))
    ax.add_patch(plt.Rectangle((intron_e, 0.30), flank, 0.40, fc=EXON_COLOR, ec="none"))
    ax.add_patch(plt.Rectangle((intron_s, 0.40), med_len, 0.20, fc=INTRON_COLOR, ec="none"))
    ax.text(flank / 2, 0.50, "5' exon", ha="center", va="center", color="white",
            fontsize=9, weight="bold")
    ax.text(intron_e + flank / 2, 0.50, "3' exon", ha="center", va="center", color="white",
            fontsize=9, weight="bold")
    ax.text((intron_s + intron_e) / 2, 0.20,
            f"intron (purple) — median {med_len:.0f} nt", ha="center", va="top",
            fontsize=9, color=INTRON_COLOR, style="italic")

    # motif intervals, in real nucleotide coordinates within the intron
    motifs = [
        ("GUAUGU", "5' splice site\n(donor)", intron_s, intron_s + len(DONOR)),
        ("UACUAAC", "branch point", intron_e - med_bp, intron_e - med_bp + len(BRANCH)),
        ("YAG", "3' splice site\n(acceptor)", intron_e - 3, intron_e),
    ]
    label_y = [1.05, 1.05, 1.05]
    for (seq, name, ms, me), ly in zip(motifs, label_y):
        ax.add_patch(plt.Rectangle((ms, 0.38), max(me - ms, total * 0.004), 0.24,
                                   fc="none", ec=MOTIF_COLOR, lw=1.4, zorder=4))
        mid = (ms + me) / 2
        ax.plot([mid, mid], [0.64, ly - 0.16], color=MOTIF_COLOR, lw=0.9, zorder=3)
        ax.text(mid, ly, seq, ha="center", va="bottom", fontsize=11,
                family="monospace", weight="bold", color=MOTIF_COLOR)
        ax.text(mid, ly - 0.13, name, ha="center", va="top", fontsize=7.5, color=MOTIF_COLOR)

    ax.annotate("", xy=(intron_e, -0.30), xytext=(intron_e - med_bp, -0.30),
                arrowprops=dict(arrowstyle="<->", color=MOTIF_COLOR, lw=1.0))
    ax.text(intron_e - med_bp / 2, -0.42,
            f"branch point to 3' splice site: median {med_bp:.0f} nt "
            f"(n={int(geom.branch_found.sum())} introns)",
            ha="center", va="top", fontsize=8, color=MOTIF_COLOR)

    ax.set_title("Figure 4D (revised) — canonical S. cerevisiae splicing motifs, "
                 "drawn at measured scale", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_png, flush=True)
    return med_len, med_bp

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    gtf = Path(args.gtf) if args.gtf else config.path("genome.gtf")
    fasta = Path(args.fasta) if args.fasta else config.path("genome.fasta")
    for p, flag in ((gtf, "--gtf"), (fasta, "--fasta")):
        if p is None or not Path(p).exists():
            sys.exit(f"error: {flag} not resolved ({p}). Fetch the genome with "
                     "`bash data/download.sh --genome -u PROJECT` or pass the flag.")

    print(f"gtf   : {gtf}\nfasta : {fasta}", flush=True)
    seqs = load_fasta(fasta)
    geom = measure_introns(transcript_exons(gtf), seqs)
    if geom.empty:
        sys.exit("error: no multi-exon transcripts found in the GTF")
    geom.to_csv(out_dir / "intron_geometry.csv", index=False)

    n = len(geom)
    print(f"\nintrons measured                : {n}")
    print(f"  start with {DONOR}            : {geom.starts_GTATGT.sum()} "
          f"({100*geom.starts_GTATGT.mean():.1f}%)")
    print(f"  end with AG                   : {geom.ends_AG.sum()} "
          f"({100*geom.ends_AG.mean():.1f}%)")
    print(f"  branch point located          : {geom.branch_found.sum()} "
          f"({100*geom.branch_found.mean():.1f}%; exact {BRANCH}: {geom.branch_exact.sum()})")
    print(f"  median intron length          : {geom.intron_len.median():.0f} nt")
    print(f"  median branch->3'SS distance  : "
          f"{geom.branch_offset_from_3ss.dropna().median():.0f} nt")

    med_len, med_bp = draw(geom, out_dir / "Figure_4D_schematic_revised.png")

    (out_dir / "fig4D_caption.md").write_text(
        "# Figure 4D — revised caption fragment\n\n"
        "Replace the panel-D sentence with wording that names the shaded region and states "
        "that the motifs are drawn at their measured positions:\n\n"
        "> **(D)** Canonical *S. cerevisiae* splicing motifs. The schematic shows a "
        "representative intron (**purple**) flanked by its 5' and 3' exons (orange), drawn "
        f"at the median intron length ({med_len:.0f} nt). The 5' splice site (donor, GUAUGU) "
        "is drawn at the first six nucleotides of the intron, the 3' splice site (acceptor, "
        "YAG) at its final three, and the branch point (UACUAAC) at its median measured "
        f"position, {med_bp:.0f} nt upstream of the 3' splice site "
        f"(n={int(geom.branch_found.sum())} annotated *S. cerevisiae* introns). Below, "
        "database consensus logos are compared with the motif Shorkie reconstructs from "
        "in-silico mutagenesis at the corresponding splice sites.\n\n"
        f"Measured from `{Path(gtf).name}` + `{Path(fasta).name}` by "
        "`scripts/05_revision/05_figure_and_caption_fixes/2_redraw_fig4D_schematic.py`; "
        "per-intron measurements in `results/intron_geometry.csv`.\n"
    )
    print(f"wrote {out_dir/'fig4D_caption.md'}")

if __name__ == "__main__":
    main()
