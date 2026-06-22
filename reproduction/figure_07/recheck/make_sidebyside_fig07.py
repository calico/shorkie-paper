#!/usr/bin/env python3
"""Side-by-side published-vs-reproduced composites for Figure 7 (deep-recheck QA).

Renders the published Figure_7.pdf, crops each panel group by fractional bbox, and pastes
it next to the corresponding reproduced PNG. Also stacks a full published | reproduced
overview. Crops are approximate (visual QA, not pixel registration).

Outputs (recheck/): panel_{AB,C,D,EFG,HI,JO}_sidebyside.png, Figure_7_published_vs_reproduced.png
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.extract_panels import render_pdf, crop_fractional
from shorkie import config

config.load()
REPRO = config.repo_root() / "reproduction" / "figure_07"
PUB_DIR = REPRO / "published"
RECHECK = REPRO / "recheck"
RD = REPRO / "reproduced"

# published panel-group fractional bboxes (l,t,r,b) — approximate from the figure layout
PUB_BBOX = {
    "AB":  (0.00, 0.00, 0.47, 0.30),
    "C":   (0.47, 0.00, 1.00, 0.17),
    "D":   (0.58, 0.17, 1.00, 0.30),
    "EFG": (0.00, 0.30, 0.60, 0.52),
    "HI":  (0.60, 0.30, 1.00, 0.52),
    "JO":  (0.00, 0.52, 1.00, 1.00),
}
# reproduced image(s) per group (stacked vertically if several)
REPRO_IMGS = {
    "AB":  ["Figure_7A_reproduced.png", "Figure_7B_reproduced.png"],
    "C":   ["Figure_7C_reproduced.png"],
    "D":   ["Figure_7D_reproduced.png"],
    "EFG": ["Figure_7EFG_reproduced.png"],
    "HI":  ["Figure_7HI_reproduced.png"],
    "JO":  ["Figure_7JO_reproduced.png"],
}


def _font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def fit_w(img, W):
    return img.resize((W, max(1, int(img.height * W / img.width))), Image.LANCZOS)


def vstack(imgs, W):
    imgs = [fit_w(i, W) for i in imgs]
    H = sum(i.height for i in imgs)
    canvas = Image.new("RGB", (W, H), "white")
    y = 0
    for i in imgs:
        canvas.paste(i, (0, y)); y += i.height
    return canvas


def banner(img, text, color):
    bar = Image.new("RGB", (img.width, 34), color)
    ImageDraw.Draw(bar).text((10, 7), text, fill="white", font=_font(20))
    out = Image.new("RGB", (img.width, img.height + 34), "white")
    out.paste(bar, (0, 0)); out.paste(img, (0, 34))
    return out


def compose(pub, repro, title, out):
    W = 760
    pub = banner(fit_w(pub, W), "PUBLISHED", "#444")
    repro = banner(fit_w(repro, W), "REPRODUCED", "#1f3fd6")
    H = max(pub.height, repro.height)
    canvas = Image.new("RGB", (2 * W + 30, H + 30), "white")
    ImageDraw.Draw(canvas).text((10, 4), title, fill="black", font=_font(18))
    canvas.paste(pub, (5, 26)); canvas.paste(repro, (W + 25, 26))
    canvas.save(out)
    print("saved", out)


def main():
    full = PUB_DIR / "Figure_7_full.png"
    render_pdf(7, 200, full)
    for grp, bbox in PUB_BBOX.items():
        pub_crop = crop_fractional(full, bbox, RECHECK / f"_pub_{grp}.png")
        pub_img = Image.open(pub_crop)
        repro = vstack([Image.open(RD / f) for f in REPRO_IMGS[grp]], 760)
        compose(pub_img, repro, f"Figure 7 — panel {grp}", RECHECK / f"panel_{grp}_sidebyside.png")
    # full overview
    pub_full = Image.open(full)
    repro_all = vstack([Image.open(RD / f) for grp in ["AB", "C", "D", "EFG", "HI", "JO"]
                        for f in REPRO_IMGS[grp]], 900)
    compose(pub_full, repro_all, "Figure 7 — full published vs reproduced",
            RECHECK / "Figure_7_published_vs_reproduced.png")


if __name__ == "__main__":
    main()
