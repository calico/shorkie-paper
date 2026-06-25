#!/usr/bin/env python3
"""Figure 4 panels A/B/C (recheck) — promoter ISM saliency, CLEAN logos.

RPL26A / FUN12 / KRE33 over the exact published 500 bp windows. Each panel stacks the
three model rows present on disk — Shorkie LM (masked-prediction IC logo), Shorkie ISM
(fine-tuned logSED), Shorkie Random-Init ISM (scratch logSED) — plus a strand-aware
gene-model track, all aligned on the panel's genomic x-axis at the published ~33:1 per-row
aspect. NO red/purple TF boxes, TF-name text, TSS dividers, '450/50 nt' labels, or
Reference-DB insets (mirrors the original 2_modisco_DNA_logo.py --no_motif_annotation).

Windows/entries are driven by fig4_common.PANELS (single source of truth):
  A RPL26A chrXII:818,862-819,362  LM eval_RP r90 · ISM _RP p4 i10 · Random _RP p4 i10
  B FUN12  chrI:75,977-76,477       LM eval_TSS r39 · ISM _TSS p0 i39 · Random _TSS p0 i39
  C KRE33  chrXIV:374,871-375,371   LM eval_TSS r5134 · ISM _RRB p11 i3 · Random _TSS_select p0 i5

Output: reproduced/Figure_4{A,B,C}_reproduced.png + Figure_4ABC_reproduced.png
        + recheck/fig4ABC_metrics.csv
"""
import sys
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F

PANELS_ABC = [p for p in F.PANELS if p["panel"] in ("A", "B", "C")]


def _combine(pngs, out):
    ims = [Image.open(p).convert("RGB") for p in pngs if p.exists()]
    if not ims:
        return
    w = max(i.width for i in ims); H = sum(i.height for i in ims)
    canvas = Image.new("RGB", (w, H), "white"); y = 0
    for im in ims:
        canvas.paste(im, (0, y)); y += im.height
    canvas.save(out); print("saved", out)


def main():
    meta = []
    for spec in PANELS_ABC:
        m = F.render_ism_panel(spec)
        meta.append(m)
        print(f"saved {m['out']} | rows={m['n_rows']} (LM={m['has_LM']} ISM={m['has_ISM']} "
              f"Random={m['has_Random']}) ism_off={m['ism_offset']} cov={m['ism_covered']}")
    _combine([F.RD / f"Figure_4{p['panel']}_reproduced.png" for p in PANELS_ABC],
             F.RD / "Figure_4ABC_reproduced.png")
    df = pd.DataFrame(meta).drop(columns=["out"])
    df.to_csv(F.RECHECK / "fig4ABC_metrics.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
