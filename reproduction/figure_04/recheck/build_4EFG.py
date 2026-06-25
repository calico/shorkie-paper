#!/usr/bin/env python3
"""Figure 4 panels E/F/G (recheck) — splicing-locus ISM saliency, CLEAN logos.

  E DTD1 chrIV:65,235-65,431    — Shorkie ISM only (SS p22 i0, cropped to window)
  F MMS2 chrVII:346,669-347,169 — 3 model rows: Shorkie LM (eval_TSS r2175 sliced) +
                                   Shorkie ISM (_TSS p33 i31) + Random-Init ISM (recomputed,
                                   gene_exp_motif_test_MMS2_panel) — the MMS2/MAD1 locus
  G HOP2 chrVII:435,625-436,401 — Shorkie ISM only (SS p80 i0, cropped to window)

Each panel stacks the model rows present on disk + a strand-aware gene-model track at the
published ~33:1 per-row aspect. NO red splice boxes, arrow labels, or any text annotation
(mirrors the original --no_motif_annotation clean logos). Row count is data-driven via
fig4_common.PANELS.

Output: reproduced/Figure_4{E,F,G}_reproduced.png + Figure_4EFG_reproduced.png
        + recheck/fig4EFG_metrics.csv
"""
import sys
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F

PANELS_EFG = [p for p in F.PANELS if p["panel"] in ("E", "F", "G")]


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
    for spec in PANELS_EFG:
        m = F.render_ism_panel(spec)
        meta.append(m)
        print(f"saved {m['out']} | rows={m['n_rows']} (LM={m['has_LM']} ISM={m['has_ISM']} "
              f"Random={m['has_Random']}) ism_off={m['ism_offset']} cov={m['ism_covered']}")
    _combine([F.RD / f"Figure_4{p['panel']}_reproduced.png" for p in PANELS_EFG],
             F.RD / "Figure_4EFG_reproduced.png")
    df = pd.DataFrame(meta).drop(columns=["out"])
    df.to_csv(F.RECHECK / "fig4EFG_metrics.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
