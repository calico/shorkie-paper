#!/usr/bin/env python3
"""Rasterize a published main-text figure PDF (and optionally crop panels) for
side-by-side visual verification against reproduced panels.

The published figures live in ``paper/Figures/Figure_{1..7}.pdf`` (one page each).
This utility renders a figure to PNG via ``pdftoppm`` (poppler), and can crop a
fractional bounding box out of the full render to isolate a single panel.

Usage:
    # full render at 150 dpi -> figure_01/published/Figure_1_full.png
    python common/extract_panels.py --figure 1 --dpi 150

    # crop a panel by fractional bbox (left top right bottom in [0,1])
    python common/extract_panels.py --figure 1 --crop 0.66 0.78 1.0 1.0 --name 1G

No hardcoded machine paths: the repo root is resolved from this file's location.
"""
import argparse
import subprocess
from pathlib import Path

REPRO_DIR = Path(__file__).resolve().parent.parent           # reproduction/
REPO_ROOT = REPRO_DIR.parent                                 # shorkie-paper/
FIGURES_DIR = REPO_ROOT / "paper" / "Figures"


def render_pdf(fig_num: int, dpi: int, out_png: Path) -> Path:
    pdf = FIGURES_DIR / f"Figure_{fig_num}.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"Published figure not found: {pdf}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # pdftoppm appends '-1' for the first page; render to a stem then rename.
    stem = out_png.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(stem)],
        check=True,
    )
    produced = Path(str(stem) + ".png")
    if produced != out_png and produced.exists():
        produced.replace(out_png)
    return out_png


def crop_fractional(src_png: Path, bbox, out_png: Path) -> Path:
    """Crop a fractional bbox=(l,t,r,b) in [0,1] from src_png -> out_png."""
    from PIL import Image
    img = Image.open(src_png)
    w, h = img.size
    l, t, r, b = bbox
    box = (int(l * w), int(t * h), int(r * w), int(b * h))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.crop(box).save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", type=int, required=True, help="main figure number 1-7")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("L", "T", "R", "B"),
                    default=None, help="fractional bbox in [0,1] to crop a panel")
    ap.add_argument("--name", type=str, default=None,
                    help="panel name for the cropped output, e.g. 1G")
    args = ap.parse_args()

    pub_dir = REPRO_DIR / f"figure_{args.figure:02d}" / "published"
    full = pub_dir / f"Figure_{args.figure}_full.png"
    render_pdf(args.figure, args.dpi, full)
    print(f"[OK] rendered {full}")

    if args.crop is not None:
        name = args.name or "panel"
        out = pub_dir / f"Figure_{args.figure}_{name}.png"
        crop_fractional(full, tuple(args.crop), out)
        print(f"[OK] cropped {out}")


if __name__ == "__main__":
    main()
