#!/usr/bin/env python3
"""Build published-vs-reproduced composites for Figure 4 (recheck).

Left: the published Figure_4_full.png (per-panel-band crops). Right: the reproduced
panel PNGs. Writes recheck/panel_{ABC,D,EFG,H}_sidebyside.png and an overall
recheck/published_vs_reproduced.png.
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from extract_panels import crop_fractional
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F

PUB = F.REPRO / "published" / "Figure_4_full.png"
RD = F.RD
OUT = F.RECHECK

# published vertical bands (l,t,r,b) in [0,1]. Panels D & H are removed from the
# reproduction (scripts kept, not rendered), so only A/B/C + E/F/G are composited.
BANDS = {
    "ABC": (0.0, 0.00, 1.0, 0.46),
    "EFG": (0.0, 0.46, 1.0, 0.86),
}
REPRO_PNG = {
    "ABC": RD / "Figure_4ABC_reproduced.png",
    "EFG": RD / "Figure_4EFG_reproduced.png",
}


def _label(img, text, h=46):
    from PIL import ImageDraw
    canvas = Image.new("RGB", (img.width, img.height + h), "white")
    canvas.paste(img, (0, h))
    d = ImageDraw.Draw(canvas)
    d.text((8, 12), text, fill="black")
    return canvas


def stack_v(imgs, gap=18):
    w = max(i.width for i in imgs)
    H = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    c = Image.new("RGB", (w, H), "white"); y = 0
    for im in imgs:
        c.paste(im, (0, y)); y += im.height + gap
    return c


def side_by_side(left, right, gap=24, target_w=1400):
    def scale(im):
        return im.resize((target_w, int(im.height * target_w / im.width)))
    l, r = scale(left), scale(right)
    H = max(l.height, r.height)
    c = Image.new("RGB", (target_w * 2 + gap, H), "white")
    c.paste(l, (0, 0)); c.paste(r, (target_w + gap, 0))
    return c


def main():
    pub = Image.open(PUB).convert("RGB")
    w, h = pub.size
    panel_sbs = []
    for key, band in BANDS.items():
        tmp = OUT / f"_pub_{key}.png"
        crop_fractional(PUB, band, tmp)
        pub_crop = _label(Image.open(tmp).convert("RGB"), f"PUBLISHED 4{key}")
        rep_path = REPRO_PNG[key]
        if not rep_path.exists():
            continue
        rep = _label(Image.open(rep_path).convert("RGB"), f"REPRODUCED 4{key}")
        sbs = side_by_side(pub_crop, rep)
        sbs.save(OUT / f"panel_{key}_sidebyside.png")
        panel_sbs.append(sbs)
        tmp.unlink(missing_ok=True)
        print("saved", OUT / f"panel_{key}_sidebyside.png")
    if panel_sbs:
        stack_v(panel_sbs).save(OUT / "published_vs_reproduced.png")
        print("saved", OUT / "published_vs_reproduced.png")


if __name__ == "__main__":
    main()
