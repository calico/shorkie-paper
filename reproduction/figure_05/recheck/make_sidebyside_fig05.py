#!/usr/bin/env python3
"""Figure 5 deep-recheck: build [published | reproduced] side-by-side composites for
panels A–J, plus a stacked overview. Published sub-panels are cropped from
published/Figure_5_full.png by fractional bbox (calibrated below); reproduced panels
come from reproduced/. Panel letters follow the PUBLISHED figure (D/I = boxplots,
E/J = motif progression).

Run (env yeast_ml):  python reproduction/figure_05/recheck/make_sidebyside_fig05.py
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from extract_panels import crop_fractional
from shorkie import config

F5 = Path(config.repo_root()) / "reproduction" / "figure_05"
PUB_FULL = F5 / "published" / "Figure_5_full.png"
PUB = F5 / "published"
RD = F5 / "reproduced"
RECHECK = F5 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

# fractional bbox (l,t,r,b) of each published panel in Figure_5_full.png (portrait 4725x5787)
# + the reproduced PNG + a title.
PANELS = {
    "A": ((0.00, 0.020, 1.00, 0.250), RD / "Figure_5A_ATG42_logos.png",
          "5A — MSN2 @ ATG42 ISM logos over time (full 500 bp; recomputed locus)"),
    "B": ((0.00, 0.262, 0.345, 0.430), RD / "eval_MSN2/YBR139W_ATG42/fold_change_by_timepoint_bar.png",
          "5B — ATG42 fold-change vs T0 (Measurement / Prediction)"),
    "C": ((0.345, 0.262, 0.66, 0.430), RD / "Figure_5C_ATG42_distance.png",
          "5C — ATG42 ISM pairwise distance (viridis)"),
    "D": ((0.66, 0.262, 1.00, 0.430), RD / "eval_MSN2/YBR139W_ATG42/pearsonr_norm_by_timepoint_boxplot.png",
          "5D — MSN2 normalized Pearson's R per timepoint"),
    "E": ((0.00, 0.435, 1.00, 0.505), RD / "Figure_5E_MSN2_motif_progression.png",
          "5E — MSN2 TF-Modisco binding-site motif over ΔT"),
    "F": ((0.00, 0.520, 1.00, 0.752), RD / "Figure_5F_TSL1_logos.png",
          "5F — MSN4 @ TSL1 ISM logos over time (full 500 bp)"),
    "G": ((0.00, 0.765, 0.345, 0.928), RD / "eval_MSN4/YML100W_TSL1/fold_change_by_timepoint_bar.png",
          "5G — TSL1 fold-change vs T0 (Measurement / Prediction)"),
    "H": ((0.345, 0.765, 0.66, 0.928), RD / "Figure_5H_TSL1_distance.png",
          "5H — TSL1 ISM pairwise distance (viridis)"),
    "I": ((0.66, 0.765, 1.00, 0.928), RD / "eval_MSN4/YML100W_TSL1/pearsonr_norm_by_timepoint_boxplot.png",
          "5I — MSN4 normalized Pearson's R per timepoint"),
    "J": ((0.00, 0.930, 1.00, 1.000), RD / "Figure_5J_MSN4_motif_progression.png",
          "5J — MSN4 TF-Modisco binding-site motif over ΔT"),
}

H = 560
GAP = 28
HEADER = 56
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


def compose(pub_im, repro_im, title, out_p):
    pub = fit_h(pub_im, H); repro = fit_h(repro_im, H)
    W = PAD * 2 + pub.width + GAP + repro.width
    canvas = Image.new("RGB", (W, HEADER + H + PAD), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 10), title, fill="black", font=_font(22))
    d.text((PAD, HEADER - 6), "PUBLISHED", fill=(150, 0, 0), font=_font(18))
    d.text((PAD + pub.width + GAP, HEADER - 6), "REPRODUCED", fill=(0, 90, 0), font=_font(18))
    canvas.paste(pub, (PAD, HEADER)); canvas.paste(repro, (PAD + pub.width + GAP, HEADER))
    d.line([(PAD + pub.width + GAP // 2, HEADER), (PAD + pub.width + GAP // 2, HEADER + H)],
           fill=(200, 200, 200), width=2)
    canvas.save(out_p)
    return canvas


def main():
    tiles = []
    for name, (bbox, repro_p, title) in PANELS.items():
        if not repro_p.exists():
            print(f"[skip] {name}: missing reproduced {repro_p}")
            continue
        pub_crop = PUB / f"Figure_5_{name}_pub.png"
        crop_fractional(PUB_FULL, bbox, pub_crop)
        c = compose(Image.open(pub_crop), Image.open(repro_p), title, RECHECK / f"panel_{name}_sidebyside.png")
        tiles.append(c)
        print(f"[OK] panel_{name}_sidebyside.png  ({c.width}x{c.height})")
    if tiles:
        W = max(t.width for t in tiles)
        Htot = sum(t.height for t in tiles) + GAP * (len(tiles) - 1)
        over = Image.new("RGB", (W, Htot), "white"); y = 0
        for t in tiles:
            over.paste(t, (0, y)); y += t.height + GAP
        over.save(RECHECK / "Figure_5_published_vs_reproduced.png")
        print(f"[OK] overview -> Figure_5_published_vs_reproduced.png ({W}x{Htot})")


if __name__ == "__main__":
    main()
