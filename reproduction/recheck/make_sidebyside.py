#!/usr/bin/env python3
"""Build per-figure side-by-side verification composites:
  figure_NN/recheck/published_vs_reproduced.png  =  [ published full figure | stacked reproduced panels ]
  figure_NN/recheck/original_vs_reproduced.png    =  [ original-script reference | reproduced ]  (where an
                                                       on-disk original-script output exists)

This is the visual half of the recheck: it shows that each data-driven panel we
regenerated matches the published panel, and — where the paper's own
figure-generation script left an output on disk — that reproduced == original.
Schematic panels are included in the published image but not separately diffed.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from shorkie import config
config.load()
REPRO = config.repo_root() / "reproduction"
WORK = Path(str(config.path("work_root")))

PANEL_W = 900          # common width for reproduced panels in the stack
PUB_H_CAP = 2600       # cap published full-figure height
PAD = 16
LABEL_H = 30


def _font(sz=20):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _labeled(img, text, width):
    """Scale img to `width`, add a label bar on top."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    h = max(1, int(img.height * width / img.width))
    img = img.resize((width, h), Image.LANCZOS)
    out = Image.new("RGB", (width, h + LABEL_H), "white")
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, width, LABEL_H], fill=(235, 235, 235))
    d.text((6, 5), text, fill=(0, 0, 0), font=_font(20))
    out.paste(img, (0, LABEL_H))
    return out


def stack(labeled_imgs, width):
    imgs = [i for i in labeled_imgs if i is not None]
    if not imgs:
        return Image.new("RGB", (width, 100), "white")
    H = sum(i.height + PAD for i in imgs) + PAD
    canvas = Image.new("RGB", (width + 2 * PAD, H), "white")
    y = PAD
    for i in imgs:
        canvas.paste(i, (PAD + (width - i.width) // 2, y))
        y += i.height + PAD
    return canvas


def side_by_side(left, right):
    H = max(left.height, right.height)
    W = left.width + right.width + PAD
    out = Image.new("RGB", (W, H), "white")
    out.paste(left, (0, (H - left.height) // 2))
    out.paste(right, (left.width + PAD, (H - right.height) // 2))
    return out


def load(p):
    p = Path(p)
    return Image.open(p) if p.exists() else None


# ── Per-figure manifest: published full + (label -> reproduced panel) + (label -> original-script ref) ──
R = lambda n, *parts: REPRO / f"figure_{n}" / "reproduced" / Path(*parts)
EQ = WORK / "revision_experiments" / "eQTL" / "viz_new" / "results"
TS = WORK / "experiments/SUM_data_process/motifs/motif_shorkie_time_series"
MPRA_VIZ = WORK / "experiments/SUM_data_process/MPRA/MPRA_promoter_seqs/results/single_measurement_stranded/viz/scatterplots"

MANIFEST = {
    "01": dict(reproduced=[
        ("1B phylogeny (repro)", R("01", "panelB_tree", "Figure_1B_reproduced.png")),
        ("1C MUMmer dot plots (repro)", R("01", "panelC_mummer", "Figure_1C_reproduced.png")),
        ("1D Mash distances (repro)", R("01", "Figure_1D_reproduced.png")),
        ("1F validation loss (repro)", R("01", "Figure_1F_reproduced.png")),
        ("1G test perplexity (repro)", R("01", "Figure_1G_reproduced.png")),
    ], originals=[]),
    "02": dict(reproduced=[
        ("2A SMT3 logo (repro)", R("02", "Figure_2A_reproduced.png")),
        ("2C TF-MoDISco motifs (repro)", R("02", "Figure_2C_reproduced.png")),
        ("2D TSS-distance hist (repro)", R("02", "Figure_2D_reproduced.png")),
        ("2E t-SNE of attn (repro)", R("02", "Figure_2E_reproduced.png")),
    ], originals=[]),
    "03": dict(reproduced=[
        ("3C bin-R distribution (repro)", R("03", "Figure_3C_reproduced.png")),
        ("3D bin-R scatter (repro)", R("03", "Figure_3D_reproduced.png")),
        ("3EFG gene-R panels (repro)", R("03", "Figure_3EFG_reproduced.png")),
        ("3H-J RNA-seq coverage (repro)", R("03", "Figure_3HIJ_coverage.png")),
    ], originals=[]),
    "04": dict(reproduced=[
        ("4ABC promoter ISM (repro)", R("04", "Figure_4ABC_reproduced.png")),
        ("4EFG splicing ISM (repro)", R("04", "Figure_4EFG_reproduced.png")),
        ("4H TF-MoDISco (repro)", R("04", "Figure_4H_reproduced.png")),
    ], originals=[]),
    "05": dict(reproduced=[
        ("5A ATG42 ISM logos (repro)", R("05", "Figure_5A_ATG42_logos.png")),
        ("5C ATG42 distance heatmap (repro)", R("05", "Figure_5C_ATG42_distance.png")),
        ("5F TSL1 ISM logos (repro)", R("05", "Figure_5F_TSL1_logos.png")),
        ("5H TSL1 distance heatmap (repro)", R("05", "Figure_5H_TSL1_distance.png")),
        ("5I MSN4 norm-R boxplot (repro)", R("05", "eval_MSN4", "YML100W_TSL1", "pearsonr_norm_by_timepoint_boxplot.png")),
    ], originals=[
        ("5I MSN4 norm-R boxplot (ORIGINAL script)", TS / "eval_MSN4" / "YML100W_TSL1" / "pearsonr_norm_by_timepoint_boxplot.png",
         R("05", "eval_MSN4", "YML100W_TSL1", "pearsonr_norm_by_timepoint_boxplot.png")),
    ]),
    "06": dict(reproduced=[
        ("6BC AUROC/AUPRC (repro)", R("06", "Figure_6BC_auroc_auprc.png")),
        ("6D native scatter (repro)", R("06", "Figure_6D_native_scatter.png")),
        ("6F SNV Δ scatter (repro)", R("06", "refalt", "scatter_all_SNVs_seqs.png")),
        ("6G motif-perturbation Δ scatter (repro)", R("06", "refalt", "scatter_motif_perturbation.png")),
        ("6H motif-tiling Δ scatter (repro)", R("06", "refalt", "scatter_motif_tiling_seqs.png")),
    ], originals=[
        ("6F SNV scatter (ORIGINAL script)", MPRA_VIZ / "all_SNVs_seqs" / "aggregated" / "scatter_aggregated_all_all_SNVs_seqs.png",
         R("06", "refalt", "scatter_all_SNVs_seqs.png")),
        ("6G motif-pert scatter (ORIGINAL script)", MPRA_VIZ / "motif_perturbation" / "aggregated" / "scatter_aggregated_all_motif_perturbation.png",
         R("06", "refalt", "scatter_motif_perturbation.png")),
        ("6H motif-tiling scatter (ORIGINAL script)", MPRA_VIZ / "motif_tiling_seqs" / "aggregated" / "scatter_aggregated_all_motif_tiling_seqs.png",
         R("06", "refalt", "scatter_motif_tiling_seqs.png")),
    ]),
    "07": dict(reproduced=[
        ("7D matched controls (repro)", R("07", "panel_D_matched_controls.png")),
        ("7EFG ROC/PR (repro)", R("07", "panel_EFG_roc_pr.png")),
        ("7HI AUPRC by TSS-dist (repro)", R("07", "panel_HI_auc_by_distance.png")),
        ("7A OMA1 coverage (repro)", R("07", "panel_A_oma1_coverage.png")),
        ("7B LAP3 coverage (repro)", R("07", "panel_B_lap3_coverage.png")),
        ("7J-K OMA1 ISM (repro)", R("07", "panel_J-K_oma1_ism.png")),
    ], originals=[
        ("7E Caudal ROC (ORIGINAL script)", EQ / "caudal_etal" / "combined_plots" / "roc_ensemble_all_sets.png",
         R("07", "panel_EFG_roc_pr.png")),
    ]),
}


def main():
    for n, man in MANIFEST.items():
        outdir = REPRO / f"figure_{n}" / "recheck"
        outdir.mkdir(parents=True, exist_ok=True)
        pub = load(REPRO / f"figure_{n}" / "published" / f"Figure_{int(n)}_full.png")

        # published vs reproduced
        repro_imgs = [_labeled(load(p), lbl, PANEL_W) for lbl, p in man["reproduced"] if load(p) is not None]
        right = stack(repro_imgs, PANEL_W)
        if pub is not None:
            ph = min(PUB_H_CAP, right.height)
            pub_lab = _labeled(pub, f"Figure {int(n)} — PUBLISHED", int(pub.width * ph / pub.height))
            comp = side_by_side(pub_lab, right)
        else:
            comp = right
        comp.save(outdir / "published_vs_reproduced.png")
        print(f"figure_{n}: published_vs_reproduced.png  ({len(repro_imgs)} reproduced panels)")

        # original-script vs reproduced (bit-level figure identity)
        for lbl, orig_p, repro_p in man.get("originals", []):
            o, r = load(orig_p), load(repro_p)
            if o is None or r is None:
                print(f"  [skip original-vs-repro {lbl}] missing ({orig_p if o is None else repro_p})")
                continue
            comp2 = side_by_side(_labeled(o, lbl, PANEL_W), _labeled(r, lbl.replace("ORIGINAL script", "reproduced"), PANEL_W))
            safe = lbl.split()[0].replace("/", "_")
            comp2.save(outdir / f"original_vs_reproduced_{safe}.png")
            print(f"  original_vs_reproduced_{safe}.png")


if __name__ == "__main__":
    main()
