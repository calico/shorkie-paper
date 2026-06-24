#!/usr/bin/env python3
"""GPU predicted Ref/Alt coverage for the Figure 7 J-O coverage tracks.

The published J-O coverage panels show the 8-fold Shorkie ensemble's PREDICTED
RNA-Seq(T0) coverage for the reference vs SNP-alt sequence (mean over folds and
tracks), over the gene window — matching ``1_viz_rnaseq_cov.py::plot_cov_plot``.
That renderer places a simple SNP-centred 16,384 bp input
(``start = center_pos - seq_len // 2``), so the predicted output bins align 1:1
with the published bin grid (``bin = (genomic - start)//stride - target_crops``).

Only the coverage is computed here (``cov_ref`` / ``cov_alt``) — the J-O ISM
*logos* come from the released ``ism_results`` cache (see ``build_7JO_ism.py``),
so the expensive per-position ISM scan in ``run_ism_eqtl.py`` is skipped
(2 forward passes / locus x 8 folds).

For each panel saves ``reproduced/ism/cov_<panel>.npz``:
  cov_ref / cov_alt (n_bins,), seq_out_start, stride, start, chrom, pos, gene.

Run on GPU:  sbatch reproduction/figure_07/panels/run_cov_eqtl_jo.sbatch
"""
import os
import sys

import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from shorkie import config
from shorkie.models.ensemble import load_ensemble, make_input, ensemble_predict

NT = ["A", "C", "G", "T"]
NT_IX = {b: i for i, b in enumerate(NT)}
SEQ_LEN = 16384

# All six Figure 7 J-O eQTL loci (panel, gene, chrom, SNP pos, ref/alt allele) —
# ref/alt as printed on the published figure (and verified in build_7JO_ism.py).
LOCI = [
    dict(panel="J", gene="YER080W", chrom="chrV",   pos=321901, ref="G", alt="T"),
    dict(panel="K", gene="YLR036C", chrom="chrXII",  pos=221376, ref="A", alt="G"),
    dict(panel="L", gene="YKL078W", chrom="chrXI",   pos=288774, ref="G", alt="A"),
    dict(panel="M", gene="YKR087C", chrom="chrXI",   pos=604356, ref="A", alt="G"),
    dict(panel="N", gene="YNL239W", chrom="chrXIV",  pos=200328, ref="G", alt="A"),
    dict(panel="O", gene="YGR046W", chrom="chrVII",  pos=584683, ref="C", alt="G"),
]


def main():
    config.load()
    mf = config.path("models.shorkie_finetuned")
    params_file = str(mf / "params.json")
    # RNA-Seq(T0) target subset -- the sheet the eQTL benchmark scoring used.
    targets_file = str(mf.parent / "cleaned_sheet_RNA-Seq_T0.txt")
    fasta_file = str(config.path("genome.fasta"))
    num_folds = int(config.get("models.num_folds", 8))

    out_dir = config.repo_root() / "reproduction" / "figure_07" / "reproduced" / "ism"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    print("Tracks (RNA-Seq T0):", len(targets_df.index), flush=True)
    models = load_ensemble(str(mf), params_file, targets_df.index, num_folds=num_folds)
    m0 = models[0]
    off = m0.model_strides[0] * m0.target_crops[0]
    stride = m0.model_strides[0]
    print(f"stride={stride} off={off} (= target_crops*stride)", flush=True)

    fasta = pysam.Fastafile(fasta_file)
    for L in LOCI:
        try:
            pos = L["pos"]
            start = pos - SEQ_LEN // 2          # simple SNP-centred window (matches the renderer)
            end = start + SEQ_LEN
            seq_out_start = int(start + off)

            x_ref = make_input(fasta, L["chrom"], start, end, SEQ_LEN)
            x_ref_np = x_ref.numpy()
            y_ref = ensemble_predict(models, x_ref)
            cov_ref = np.mean(y_ref, axis=(0, 1, 3)).astype(np.float32)   # (bins,)

            ci = pos - start - 1
            extracted_ref = NT[int(np.argmax(x_ref_np[ci, :4]))] if x_ref_np[ci, :4].sum() > 0 else "N"

            x_alt = np.copy(x_ref_np)
            x_alt[ci, :4] = 0.0
            x_alt[ci, NT_IX[L["alt"]]] = 1.0
            y_alt = ensemble_predict(models, tf.constant(x_alt))
            cov_alt = np.mean(y_alt, axis=(0, 1, 3)).astype(np.float32)

            np.savez(
                os.path.join(out_dir, f"cov_{L['panel']}.npz"),
                cov_ref=cov_ref, cov_alt=cov_alt,
                seq_out_start=seq_out_start, stride=stride, start=start,
                chrom=L["chrom"], pos=pos, gene=L["gene"],
                ref=L["ref"], alt=L["alt"], extracted_ref=extracted_ref,
            )
            print(f"[{L['panel']}] {L['gene']} {L['chrom']}:{pos} {L['ref']}>{L['alt']} "
                  f"extracted_ref={extracted_ref} cov_ref max={cov_ref.max():.3f} "
                  f"cov_alt max={cov_alt.max():.3f} n_bins={len(cov_ref)} -> cov_{L['panel']}.npz",
                  flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{L['panel']}] ERROR: {e}", file=sys.stderr, flush=True)

    fasta.close()
    print("DONE_COV", flush=True)


if __name__ == "__main__":
    main()
