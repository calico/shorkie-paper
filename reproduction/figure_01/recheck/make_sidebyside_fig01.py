#!/usr/bin/env python3
"""Figure 1 deep-recheck — Step 4: build clean per-panel [published | reproduced]
side-by-side composites for the data-driven panels (B, C, D, F, G), plus a single
stacked overview. Schematic panels (A, E) are skipped per the recheck scope.

Published sub-panels were cropped from published/Figure_1_full.png (see the crop
bboxes recorded in DISCREPANCIES.md). Reproduced panels come from reproduced/.

Run (env yeast_ml):
    python reproduction/figure_01/recheck/make_sidebyside_fig01.py
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shorkie import config

F1 = Path(config.repo_root()) / "reproduction" / "figure_01"
PUB = F1 / "published"
RD = F1 / "reproduced"
RECHECK = F1 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

PANELS = {
    "B": (PUB / "Figure_1_B_pub.png", RD / "panelB_tree" / "Figure_1B_reproduced.png",
          "1B — phylogeny (4 datasets; Saccharomycetales clade)"),
    "C": (PUB / "Figure_1_C_pub.png", RD / "panelC_mummer" / "Figure_1C_reproduced.png",
          "1C — MUMmer dot plots (R64 vs representative genomes; strain = YJM195)"),
    "D": (PUB / "Figure_1_D_pub.png", RD / "Figure_1D_reproduced.png",
          "1D — Mash distance from R64 (165_Saccharomycetales / 80_Strains)"),
    "F": (PUB / "Figure_1_F_pub.png", RD / "Figure_1F_reproduced.png",
          "1F — validation loss (legend = min valid loss)"),
    "G": (PUB / "Figure_1_G_pub.png", RD / "Figure_1G_reproduced.png",
          "1G — gene/intergenic test perplexity"),
}

H = 720          # common panel height
GAP = 28
HEADER = 60
PAD = 16


def _font(sz):
    for p in ["/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def fit_h(im, h):
    w = max(1, int(im.width * h / im.height))
    return im.convert("RGB").resize((w, h), Image.LANCZOS)


def compose(pub_p, repro_p, title, out_p):
    pub = fit_h(Image.open(pub_p), H)
    repro = fit_h(Image.open(repro_p), H)
    W = PAD * 2 + pub.width + GAP + repro.width
    canvas = Image.new("RGB", (W, HEADER + H + PAD), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 12), title, fill="black", font=_font(26))
    d.text((PAD, HEADER - 6), "PUBLISHED", fill=(150, 0, 0), font=_font(20))
    d.text((PAD + pub.width + GAP, HEADER - 6), "REPRODUCED", fill=(0, 90, 0), font=_font(20))
    canvas.paste(pub, (PAD, HEADER))
    canvas.paste(repro, (PAD + pub.width + GAP, HEADER))
    # divider
    ImageDraw.Draw(canvas).line([(PAD + pub.width + GAP // 2, HEADER),
                                 (PAD + pub.width + GAP // 2, HEADER + H)], fill=(200, 200, 200), width=2)
    canvas.save(out_p)
    return canvas


def main():
    tiles = []
    for name, (pub_p, repro_p, title) in PANELS.items():
        if not pub_p.exists():
            print(f"[skip] {name}: missing published crop {pub_p}")
            continue
        if not repro_p.exists():
            print(f"[skip] {name}: missing reproduced {repro_p}")
            continue
        out = RECHECK / f"panel_{name}_sidebyside.png"
        c = compose(pub_p, repro_p, title, out)
        tiles.append(c)
        print(f"[OK] {out}  ({c.width}x{c.height})")

    # stacked overview of all panels
    if tiles:
        W = max(t.width for t in tiles)
        Htot = sum(t.height for t in tiles) + GAP * (len(tiles) - 1)
        over = Image.new("RGB", (W, Htot), "white")
        y = 0
        for t in tiles:
            over.paste(t, (0, y))
            y += t.height + GAP
        over.save(RECHECK / "Figure_1_published_vs_reproduced.png")
        print(f"[OK] overview -> {RECHECK / 'Figure_1_published_vs_reproduced.png'} ({W}x{Htot})")


if __name__ == "__main__":
    main()
