#!/usr/bin/env python3
"""Figure 2 deep-recheck — build [published | reproduced] side-by-side composites
for panels A–E + a stacked overview. Mirrors the Figure-1 recheck approach.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from shorkie import config

F2 = Path(config.repo_root()) / "reproduction" / "figure_02"
PUB = F2 / "published"
RD = F2 / "reproduced"
RECHECK = F2 / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)

# 2B is removed; 2C is provided as upstream scripts (not re-rendered) -> published-only tile.
PANELS = {
    "A": (PUB / "Figure_2_A_pub.png", RD / "Figure_2A_reproduced.png",
          "2A — SMT3 promoter logos: SpeciesLM / Shorkie LM / Shorkie 15% iterative"),
    "C": (PUB / "Figure_2_C_pub.png", None,
          "2C — TF-MoDISco motif conservation across 6 fungal tiers (reproduction skipped — see motif_lm[/__unseen_species]/4_viz_motif.py)"),
    "D": (PUB / "Figure_2_D_pub.png", RD / "Figure_2D_reproduced.png",
          "2D — motif enrichment vs TSS (published 6-panel grid: MIG3.4/Abf1.1/Rap1.1/Reb1p/CHA4.11/SWI5.7)"),
    "E": (PUB / "Figure_2_E_pub.png", RD / "Figure_2E_reproduced.png",
          "2E — t-SNE of 1st-attn LM embeddings by genomic element"),
}
H = 720
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
    if repro_p is None:                      # published-only tile (panel skipped)
        W = PAD * 2 + pub.width
        canvas = Image.new("RGB", (W, HEADER + H + PAD), "white")
        d = ImageDraw.Draw(canvas)
        d.text((PAD, 12), title, fill="black", font=_font(22))
        d.text((PAD, HEADER - 6), "PUBLISHED (reproduction skipped)", fill=(150, 0, 0), font=_font(20))
        canvas.paste(pub, (PAD, HEADER))
        canvas.save(out_p)
        return canvas
    repro = fit_h(Image.open(repro_p), H)
    W = PAD * 2 + pub.width + GAP + repro.width
    canvas = Image.new("RGB", (W, HEADER + H + PAD), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 12), title, fill="black", font=_font(26))
    d.text((PAD, HEADER - 6), "PUBLISHED", fill=(150, 0, 0), font=_font(20))
    d.text((PAD + pub.width + GAP, HEADER - 6), "REPRODUCED", fill=(0, 90, 0), font=_font(20))
    canvas.paste(pub, (PAD, HEADER))
    canvas.paste(repro, (PAD + pub.width + GAP, HEADER))
    ImageDraw.Draw(canvas).line([(PAD + pub.width + GAP // 2, HEADER),
                                 (PAD + pub.width + GAP // 2, HEADER + H)], fill=(200, 200, 200), width=2)
    canvas.save(out_p)
    return canvas


def main():
    tiles = []
    for name, (pub_p, repro_p, title) in PANELS.items():
        if not pub_p.exists() or (repro_p is not None and not repro_p.exists()):
            print(f"[skip] {name}: missing {pub_p if not pub_p.exists() else repro_p}")
            continue
        out = RECHECK / f"panel_{name}_sidebyside.png"
        tiles.append(compose(pub_p, repro_p, title, out))
        print(f"[OK] {out} ({tiles[-1].width}x{tiles[-1].height})")
    if tiles:
        W = max(t.width for t in tiles)
        Htot = sum(t.height for t in tiles) + GAP * (len(tiles) - 1)
        over = Image.new("RGB", (W, Htot), "white")
        y = 0
        for t in tiles:
            over.paste(t, (0, y)); y += t.height + GAP
        over.save(RECHECK / "Figure_2_published_vs_reproduced.png")
        print(f"[OK] overview -> {RECHECK/'Figure_2_published_vs_reproduced.png'} ({W}x{Htot})")


if __name__ == "__main__":
    main()
