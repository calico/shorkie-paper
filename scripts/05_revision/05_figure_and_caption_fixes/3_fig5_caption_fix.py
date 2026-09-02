#!/usr/bin/env python3
"""Revision experiment 05, step 3 — confirm and correct the Figure 5 panel labelling
.

The error is confined to the caption: the figure itself and the Results text agree with
each other and disagree with the caption.

  caption (manuscript L299-300)  (D) TF-MoDISco-identified motifs from the DeltaT ISM
                                     matrices
                                 (E) Boxplot of normalised Pearson's R across all genes
  Results text (L274)            "...between experimental and predicted RNA-seq ranging
                                 from 0.55 to 0.65 (Figure 5D)"   <- the boxplot
  Results text (L276)            "...genome-wide TF-MoDISco... (Figure 5E)"  <- the motifs
  the figure itself              reproduction/figure_05 builds the MSN2 boxplot as panel D
                                 and the MSN4 boxplot as panel I

Because the caption also says panels F-J are "analogous to (A-E)", the same swap is
inherited by I and J.

Rather than assert this, the script checks it against the reproduction tree: the panel
builder ``build_DI_boxplots.py`` and the artifacts it writes are the independent record of
which panel letter the boxplot actually carries.

Outputs ``results/fig5_caption_corrected.md`` and ``results/verify_revision_05_fig5.csv``.
CPU only, instant. Reads only committed files.
"""
import argparse
import sys
from pathlib import Path

from shorkie import config

sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
from compare import Check, write_verdicts, summary  # noqa: E402

# What the reproduction writes, and therefore what panel letter each content type carries.
EXPECTED_BOXPLOTS = {
    "D": "Figure_5D_MSN2_boxplot.png",
    "I": "Figure_5I_MSN4_boxplot.png",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Verify and correct the Figure 5 caption.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    repro = repo / "reproduction" / "figure_05" / "reproduced"
    checks = []
    for letter, fname in EXPECTED_BOXPLOTS.items():
        present = (repro / fname).exists()
        checks.append(Check(
            panel=f"5{letter}",
            metric=f"panel {letter} is the normalised-Pearson boxplot "
                   f"(reproduction writes {fname})",
            reported=1, reproduced=1 if present else 0, rtol=0.0, atol=0.0))
        print(f"{'OK  ' if present else 'MISS'} {repro.relative_to(repo)}/{fname}", flush=True)

    write_verdicts(checks, out_dir / "verify_revision_05_fig5.csv")
    print(summary(checks))

    (out_dir / "fig5_caption_corrected.md").write_text("""# Figure 5 — corrected caption

Panels D and E are reversed in the caption. The figure and the
Results text agree with each other; only the caption is wrong, and because the caption
declares panels F–J "analogous to (A–E)", the same swap propagates to I and J.

## Change

| Panel | Caption currently says | Should say |
|---|---|---|
| D | TF-MoDISco-identified motifs from ΔT ISM matrices | Boxplot of normalised Pearson's R |
| E | Boxplot of normalised Pearson's R | TF-MoDISco-identified motifs from ΔT ISM matrices |
| I | (inherits D via "analogous to (A–E)") | Boxplot of normalised Pearson's R, MSN4 |
| J | (inherits E) | TF-MoDISco-identified motifs, MSN4 |

## Corrected caption

> **Figure 5. Time-course analysis of stress-responsive transcription factor induction.**
> **(A–E)** MSN2 induction at the ATG42 promoter region (−450 to +50 bp relative to the TSS;
> chrII:515,214–515,714), sampled at seven time points labelled in minutes.
> **(A)** Shorkie ISM sequence logos: rows correspond to successive time points (top to
> bottom), with the bottom row showing the reference. Key TF-binding motifs are annotated.
> **(B)** Experimental fold-change in reads per million (RPM) (blue) versus Shorkie-predicted
> signal (orange) across the ATG42 locus at each time point.
> **(C)** Heatmap of pairwise Euclidean distances between ISM logos, illustrating temporal
> divergence in motif strength and composition.
> **(D)** Boxplot of normalised Pearson's R between experimental and predicted profiles
> across all *S. cerevisiae* genes for MSN2 induction at each time point.
> **(E)** TF-MoDISco-identified motifs extracted from ΔT ISM matrices relative to *T₀*.
> **(F–J)** MSN4 induction at the TSL1 promoter region (−450 to +50 bp relative to the TSS;
> chrXIII:70,173–70,673), with panels analogous to (A–E).

## Cross-check

The Results text already uses the corrected assignment — "(Figure 5D)" is cited for the
0.55–0.65 correlation range (the boxplot) and "(Figure 5E)" for the genome-wide TF-MoDISco
analysis (the motifs) — so this correction makes the caption consistent with the text
rather than changing any claim.
""")
    print(f"wrote {out_dir/'fig5_caption_corrected.md'}")

if __name__ == "__main__":
    main()
