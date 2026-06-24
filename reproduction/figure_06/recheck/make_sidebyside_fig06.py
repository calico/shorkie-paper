#!/usr/bin/env python3
"""Side-by-side published-vs-reproduced composites for Figure 6.

Renders the published Figure_6.pdf, crops each panel region by fractional bbox, and
stacks it above the corresponding reproduced PNG. Also assembles a full
published-vs-reproduced montage. Outputs land in recheck/.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from extract_panels import render_pdf, crop_fractional  # noqa
from shorkie import config

config.load()
F6 = config.repo_root() / "reproduction" / "figure_06"
PUB_DIR = F6 / "published"
REPRO = F6 / "reproduced"
RECHECK = F6 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

# Fractional bboxes (l, t, r, b) of each panel in the published full render.
PANEL_BBOX = {
    # 6B/6C tightly bound the published panel content (measured at 300 dpi: ratio ~1.93).
    "6B": (0.000, 0.200, 0.455, 0.487),
    "6C": (0.520, 0.200, 0.987, 0.487),
    "6D": (0.00, 0.42, 0.34, 0.68),
    "6E": (0.33, 0.42, 0.67, 0.68),
    "6F": (0.66, 0.42, 1.00, 0.68),
    "6G": (0.00, 0.68, 0.34, 1.00),
    "6H": (0.33, 0.68, 0.67, 1.00),
    "6I": (0.66, 0.68, 1.00, 1.00),
}
REPRO_PNG = {
    "6B": "Figure_6B.png", "6C": "Figure_6C.png",
    "6D": "Figure_6D.png", "6E": "Figure_6E.png", "6F": "Figure_6F.png",
    "6G": "Figure_6G.png", "6H": "Figure_6H.png", "6I": "Figure_6I.png",
}


def stack(panel, pub_crop, repro_png, out):
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    for ax, img, lab in ((axes[0], pub_crop, f"published {panel}"),
                         (axes[1], repro_png, f"reproduced {panel}")):
        if img.exists():
            ax.imshow(Image.open(img)); ax.set_title(lab, fontsize=11)
        else:
            ax.text(0.5, 0.5, f"missing {img.name}", ha="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    full = render_pdf(6, 200, PUB_DIR / "Figure_6_full.png")
    print(f"[OK] rendered {full}")
    for panel, bbox in PANEL_BBOX.items():
        crop = PUB_DIR / f"Figure_6_{panel}.png"
        crop_fractional(full, bbox, crop)
        out = RECHECK / f"panel_{panel}_sidebyside.png"
        stack(panel, crop, REPRO / REPRO_PNG[panel], out)
        print(f"[OK] {out.name}")

    # full montage: published (left) vs reproduced grid (right)
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    axes[0].imshow(Image.open(full)); axes[0].set_title("Published Figure 6", fontsize=14); axes[0].axis("off")
    axes[1].text(0.5, 0.5, "reproduced panels:\n" + ", ".join(REPRO_PNG), ha="center", va="center", fontsize=11)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RECHECK / "Figure_6_published_vs_reproduced.png", dpi=110)
    plt.close(fig)
    print("[OK] Figure_6_published_vs_reproduced.png")


if __name__ == "__main__":
    main()
