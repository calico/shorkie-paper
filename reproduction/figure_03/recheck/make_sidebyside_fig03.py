#!/usr/bin/env python3
"""Build [published | reproduced] side-by-side composites for Figure 3.

Crops the relevant sub-panels out of published/Figure_3_full.png (fractional
bboxes) and stacks them next to the reproduced PNGs:
  panel_C_sidebyside.png      (3C split violin)
  panel_DEFG_sidebyside.png   (3D/E/F/G scatter row)
  panel_HIJ_sidebyside.png    (3H/I/J coverage row)
  Figure_3_published_vs_reproduced.png  (overall)
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../reproduction
from common.extract_panels import crop_fractional
from shorkie import config
config.load()
REPRO = config.repo_root() / "reproduction" / "figure_03"
PUB = REPRO / "published" / "Figure_3_full.png"
RC = REPRO / "recheck"
REP = REPRO / "reproduced"

# fractional bboxes (l,t,r,b) of the published panels in Figure_3_full.png
BBOX = {
    "C":    (0.55, 0.00, 1.00, 0.33),
    "DEFG": (0.00, 0.33, 1.00, 0.63),
    "HIJ":  (0.00, 0.63, 1.00, 1.00),
}


def label(img, text, pad=44):
    """Add a top banner label to a PIL image."""
    from PIL import ImageDraw
    w, h = img.size
    canvas = Image.new("RGB", (w, h + pad), "white")
    canvas.paste(img, (0, pad))
    d = ImageDraw.Draw(canvas)
    d.text((10, 12), text, fill="black")
    return canvas


def hstack(imgs, gap=20):
    h = max(im.size[1] for im in imgs)
    scaled = []
    for im in imgs:
        w = int(im.size[0] * h / im.size[1])
        scaled.append(im.resize((w, h)))
    W = sum(im.size[0] for im in scaled) + gap * (len(scaled) - 1)
    out = Image.new("RGB", (W, h), "white")
    x = 0
    for im in scaled:
        out.paste(im, (x, 0)); x += im.size[0] + gap
    return out


def vstack(imgs, gap=24):
    w = max(im.size[0] for im in imgs)
    scaled = []
    for im in imgs:
        hh = int(im.size[1] * w / im.size[0])
        scaled.append(im.resize((w, hh)))
    H = sum(im.size[1] for im in scaled) + gap * (len(scaled) - 1)
    out = Image.new("RGB", (w, H), "white")
    y = 0
    for im in scaled:
        out.paste(im, (0, y)); y += im.size[1] + gap
    return out


def main():
    crops = {}
    for k, bb in BBOX.items():
        out = RC / f"_pub_{k}.png"
        crop_fractional(PUB, bb, out)
        crops[k] = Image.open(out)

    pairs = [
        ("C",    REP / "Figure_3C_reproduced.png",     "panel_C_sidebyside.png"),
        ("DEFG", REP / "Figure_3DEFG_reproduced.png",  "panel_DEFG_sidebyside.png"),
        ("HIJ",  REP / "Figure_3HIJ_coverage.png",     "panel_HIJ_sidebyside.png"),
    ]
    composites = []
    for k, rep_png, out_name in pairs:
        pub = label(crops[k], f"PUBLISHED  3{k}")
        rep = label(Image.open(rep_png), f"REPRODUCED  3{k}")
        comp = hstack([pub, rep])
        comp.save(RC / out_name)
        composites.append(comp)
        print("wrote", RC / out_name)

    overall = vstack(composites)
    overall.save(RC / "Figure_3_published_vs_reproduced.png")
    print("wrote", RC / "Figure_3_published_vs_reproduced.png")
    for k in BBOX:
        (RC / f"_pub_{k}.png").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
