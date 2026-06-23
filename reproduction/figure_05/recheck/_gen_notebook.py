#!/usr/bin/env python3
"""Regenerate reproduce_figure_05.ipynb as a clean orchestrator that delegates to the
recheck/ builders with the CORRECTED published panel letters. One-off generator (not a
reproduction artifact); run then execute with jupyter nbconvert."""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "reproduce_figure_05.ipynb"
old = nbf.read(str(NB), as_version=4)
meta = dict(old.metadata)

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# Figure 5 — Time-course stress-responsive TF induction (MSN2 & MSN4)

> *"Time-course analysis of stress-responsive transcription factor induction."*

Reproduction of **main-text Figure 5** (published: [`../../paper/Figures/Figure_5.pdf`](../../paper/Figures/Figure_5.pdf)).
Two genes / two halves (β-estradiol induction RNA-seq; T0,T5,T10,T15,T30,T45,T60,T90):
MSN2 @ ATG42 (YBR139W), chrII:515,214–515,714; MSN4 @ TSL1 (YML100W), chrXIII:70,173–70,673.

**Published panel map** — this reproduction covers the 8 panels A–D, F–I; **panels E/J (TF-Modisco motif
progression) are intentionally skipped** (see `recheck/DISCREPANCIES.md`):

| Row | MSN2 | MSN4 |
|---|---|---|
| ISM logos (8 rows T0..T90, full 500 bp) | **A** | **F** |
| Fold-change bars (Measurement / Prediction vs T0) | **B** | **G** |
| Pairwise Euclidean-distance heatmap (viridis) | **C** | **H** |
| **Normalized Pearson's R boxplot** (per timepoint) | **D** | **I** |
| TF-Modisco binding-site motif over ΔT | E *(skipped)* | J *(skipped)* |

This notebook delegates to the deep-recheck builders in [`recheck/`](recheck/). See
[`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md) for: the **GPU recompute of the ATG42 ISM**
(panels A/C — the locus was never released); the full-window logo fix; the boxplots being the published
**D/I** (the prior draft mislabeled them 5E/5J); and **why E/J are skipped** (the published MSN2 panel E
uses an extended timepoint series not in the released artifacts)."""))

cells.append(md(r"""## Prerequisite — ATG42 ISM (GPU, one-time)

Panels **A/C** need the ATG42 promoter ISM, which is **absent from the released artifacts**. It is recomputed
once on GPU with the *exact* released driver/model (`self_supervised_unet_small_bert_drop` f0c0;
`hound_ism_bed.py -l 500 --rc --stats logSED`; BED = the 500 bp ATG42 promoter window):

```bash
sbatch reproduction/figure_05/panels/run_atg42_ism.sbatch   # -> reproduced/ism_atg42/scores.h5
```

All other panels are CPU-reproducible from released ISM / eval / modisco artifacts."""))

cells.append(code(r"""import subprocess, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
from shorkie import config

REPO = Path(config.repo_root())
RD = REPO / "reproduction" / "figure_05" / "reproduced"
RECHECK = REPO / "reproduction" / "figure_05" / "recheck"

def run_builder(script, *args):
    cmd = [sys.executable, str(RECHECK / script), *map(str, args)]
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode:
        print("STDERR:", r.stderr[-2000:])

def show(*relpaths, width=900):
    for rp in relpaths:
        p = RD / rp
        print(("OK  " if p.exists() else "MISS"), rp)
        if p.exists():
            display(Image(filename=str(p), width=width))

print("ready")"""))

cells.append(md(r"""## Panels A/F (ISM logos, full 500 bp) + C/H (pairwise distance)

`recheck/build_logos_distance.py` loads each locus' ISM `scores.h5` (TSL1 = released MSN4 part2 idx7;
**ATG42 = the recomputed `ism_atg42/scores.h5`**), renders the full-window per-timepoint logo stack and the
8×8 viridis Euclidean-distance heatmap. *(Logo rendering over 500 bp is slow — a few minutes per locus.)*
The published A/F additionally carry **manual red-box / gene-track overlays** the pipeline never drew — see
DISCREPANCIES."""))

cells.append(code(r"""run_builder("build_logos_distance.py", "--locus", "both")
show("Figure_5A_ATG42_logos.png", "Figure_5C_ATG42_distance.png",
     "Figure_5F_TSL1_logos.png", "Figure_5H_TSL1_distance.png", width=1000)"""))

cells.append(md(r"""## Panels B/G — experimental vs predicted fold-change at the locus

Re-run of `…/motif_shorkie__time_series/1_time_track_metrics_viz.py` (`--gene YBR139W` / `YML100W`):
Measurement-FC (blue) vs Prediction-FC (orange) ±SEM, vs T0. Outputs cached under `reproduced/eval_*/`."""))

cells.append(code(r"""show("eval_MSN2/YBR139W_ATG42/fold_change_by_timepoint_bar.png",
     "eval_MSN4/YML100W_TSL1/fold_change_by_timepoint_bar.png", width=620)"""))

cells.append(md(r"""## Panels D/I — normalized Pearson's R per timepoint (the 0.55–0.65 anchor)

`recheck/build_DI_boxplots.py` re-renders the per-track `pearsonr_norm` boxplot (grouped by timepoint, `n=`
annotated) from `eval_{MSN2,MSN4}/eval.txt` into the top-level `reproduced/Figure_5D_MSN2_boxplot.png` /
`Figure_5I_MSN4_boxplot.png`. Medians **MSN2 0.591 / MSN4 0.618** (published band 0.55–0.65); per-timepoint
n-counts **8,12,8,12,9,9,7,9** (MSN2) / **11,7,8,10,6,8,12,8** (MSN4)."""))

cells.append(code(r"""run_builder("build_DI_boxplots.py")
show("Figure_5D_MSN2_boxplot.png", "Figure_5I_MSN4_boxplot.png", width=600)"""))

cells.append(md(r"""## Panels E/J — intentionally skipped

The published **E/J** (TF-Modisco binding-site motif progression) are **not reproduced**: the published MSN2
panel E uses an extended timepoint series (T5,T10,T20,T40,T70,T120,T180 — the series the released code assigns
to SWI4) that is **not in the released MSN2 artifacts**, so panel E cannot be faithfully reproduced; the
analogous MSN4 panel J is skipped alongside it. See [`recheck/DISCREPANCIES.md`](recheck/DISCREPANCIES.md)."""))

cells.append(md(r"""## Verification

`recheck/build_verify_fig05.py` rebuilds `reproduced/verify_fig05.csv` with figure-based targets and the
corrected panel letters (windows exact; norm-R medians in band; per-timepoint n-counts; distance divergence;
STRE motif recovery; global ΔlogFC R)."""))

cells.append(code(r"""run_builder("build_verify_fig05.py")
display(pd.read_csv(RD / "verify_fig05.csv"))"""))

cells.append(md(r"""## Side-by-side vs published

`recheck/make_sidebyside_fig05.py` crops the published panels A–J and pairs them with the reproduced ones."""))

cells.append(code(r"""run_builder("make_sidebyside_fig05.py")
sbs = RECHECK / "Figure_5_published_vs_reproduced.png"
if sbs.exists(): display(Image(filename=str(sbs), width=1200))
pub = REPO / "reproduction" / "figure_05" / "published" / "Figure_5_full.png"
if pub.exists(): display(Image(filename=str(pub), width=700))"""))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = meta
nbf.write(nb, str(NB))
print(f"wrote {NB} with {len(cells)} cells; kernelspec={meta.get('kernelspec')}")
