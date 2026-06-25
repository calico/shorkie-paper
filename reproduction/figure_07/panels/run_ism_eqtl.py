#!/usr/bin/env python3
"""GPU ISM scoring for Figure 7 panels A/B (OMA1, LAP3 locus) and J-O (eQTL-SNP
ISM saliency maps).

Faithfully reproduces the variant-effect logSED of the eQTL benchmark scorer
(``scripts/04_analysis/shorkie/eqtl/2_variant_scoring/score_variants_shorkie.py``):
the same 8-fold Shorkie ensemble, the same RNA-Seq(T0) target slice, the same
variant-aware 16 kb window placement (variant inside the input, gene body in the
cropped output), and the same gene-body ``output_slice`` for the logSED sum.

For each locus it saves to ``reproduced/ism/<name>.npz``:
  - cov_ref / cov_alt : per-bin mean predicted coverage (ref vs SNP-alt)  -> panel A/B
  - snp_logsed        : gene-body logSED at the eQTL SNP (sign = effect direction)
  - grid              : (n_scan, 4) per-position logSED for every alt base   -> panel J-O
  - bookkeeping       : scan_positions, ref_bases, gene_slice idx, window, etc.

Run on GPU (a100/ica100). CPU works but is slow.
"""
import os
import sys

import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
from baskerville import gene as bgene

from shorkie import config
from shorkie.models.ensemble import load_ensemble, make_input, ensemble_predict, logSED

NT = ["A", "C", "G", "T"]
NT_IX = {b: i for i, b in enumerate(NT)}
SEQ_LEN = 16384
SCAN_HALF = 40                          # +/- bp around the SNP -> 81 scanned positions

# The two positive cis-eQTL loci named in the Figure 7 caption. The eQTL SNP
# (dominant-effect variant for each gene; |logSED| >> sibling SNPs) sits in the
# promoter just outside the gene-body display window. ref/alt/pos taken verbatim
# from the released positive-variant scores CSVs.
LOCI = [
    dict(name="oma1", gene="YKR087C", chrom="chrXI",  pos=604356, ref="A", alt="G",
         window=(603195, 604232), caption="reduced expr. with alt (logSED<0)"),
    dict(name="lap3", gene="YNL239W", chrom="chrXIV", pos=200328, ref="G", alt="A",
         window=(200569, 201933), caption="increased expr. with alt (logSED>0)"),
]


def slice_to_idx(gs):
    if isinstance(gs, slice):
        return np.arange(gs.start, gs.stop, gs.step or 1)
    return np.asarray(gs).ravel()


def main():
    config.load()
    mf = config.path("models.shorkie_finetuned")
    params_file = str(mf / "params.json")
    # RNA-Seq(T0) target subset -- the sheet the eQTL benchmark scoring used.
    targets_file = str(mf.parent / "cleaned_sheet_RNA-Seq_T0.txt")
    fasta_file = str(config.path("genome.fasta"))
    gtf_file = str(config.path("genome.gtf"))
    num_folds = int(config.get("models.num_folds", 8))

    out_dir = config.repo_root() / "reproduction" / "figure_07" / "reproduced" / "ism"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("model_dir   :", mf, flush=True)
    print("targets     :", targets_file, flush=True)
    print("num_folds   :", num_folds, flush=True)

    targets_df = pd.read_csv(targets_file, index_col=0, sep="\t")
    target_index = targets_df.index
    print("Tracks (RNA-Seq T0):", len(target_index), flush=True)

    models = load_ensemble(str(mf), params_file, target_index, num_folds=num_folds)
    m0 = models[0]
    off = m0.model_strides[0] * m0.target_crops[0]
    olen = m0.model_strides[0] * m0.target_lengths[0]
    stride = m0.model_strides[0]

    fasta = pysam.Fastafile(fasta_file)
    transcriptome = bgene.Transcriptome(gtf_file)

    for L in LOCI:
        try:
            keys = [k for k in transcriptome.genes if L["gene"] in k]
            assert keys, f"gene {L['gene']} not in GTF"
            gene = transcriptome.genes[keys[0]]
            chrom = gene.chrom if gene.chrom.startswith("chr") else "chr" + gene.chrom
            gc = gene.midpoint()
            pos = L["pos"]

            # Variant-aware window placement (mirrors score_variants_shorkie.py).
            min_start = max(pos - SEQ_LEN + 1, gc - off - olen + 1)
            max_start = min(pos - 1, gc - off)
            start = int((min_start + max_start) // 2) if min_start <= max_start \
                else int(gc - SEQ_LEN // 2)
            end = start + SEQ_LEN
            seq_out_start = int(start + off)
            gs = gene.output_slice(seq_out_start, int(olen), stride, False)
            gs_idx = slice_to_idx(gs)

            # Reference prediction.
            x_ref = make_input(fasta, chrom, start, end, SEQ_LEN)
            x_ref_np = x_ref.numpy()
            y_ref = ensemble_predict(models, x_ref)
            cov_ref = np.mean(y_ref, axis=(0, 1, 3))           # (bins,)

            ci = pos - start - 1
            extracted_ref = NT[int(np.argmax(x_ref_np[ci, :4]))] \
                if x_ref_np[ci, :4].sum() > 0 else "N"

            # SNP-alt prediction.
            x_alt = np.copy(x_ref_np)
            x_alt[ci, :4] = 0.0
            x_alt[ci, NT_IX[L["alt"]]] = 1.0
            y_alt = ensemble_predict(models, tf.constant(x_alt))
            cov_alt = np.mean(y_alt, axis=(0, 1, 3))
            snp_logsed = logSED(y_ref, y_alt, gs_idx)

            print(f"[{L['name']}] {chrom}:{pos} {L['ref']}>{L['alt']} "
                  f"extracted_ref={extracted_ref} snp_logSED={snp_logsed:+.4f} "
                  f"({L['caption']})", flush=True)

            # ISM saliency scan centered on the SNP (panel J-O).
            scan_positions = list(range(pos - SCAN_HALF, pos + SCAN_HALF + 1))
            grid = np.full((len(scan_positions), 4), np.nan, dtype=np.float32)
            ref_bases = []
            for i, p in enumerate(scan_positions):
                cj = p - start - 1
                if not (0 <= cj < SEQ_LEN):
                    ref_bases.append("N")
                    continue
                rix = int(np.argmax(x_ref_np[cj, :4]))
                ref_bases.append(NT[rix] if x_ref_np[cj, :4].sum() > 0 else "N")
                for aix in range(4):
                    if aix == rix:
                        grid[i, aix] = 0.0
                        continue
                    xm = np.copy(x_ref_np)
                    xm[cj, :4] = 0.0
                    xm[cj, aix] = 1.0
                    ym = ensemble_predict(models, tf.constant(xm))
                    grid[i, aix] = logSED(y_ref, ym, gs_idx)
                if (i + 1) % 20 == 0:
                    print(f"    [{L['name']}] scanned {i+1}/{len(scan_positions)}", flush=True)

            np.savez(
                os.path.join(out_dir, f"{L['name']}.npz"),
                cov_ref=cov_ref, cov_alt=cov_alt,
                gene_slice_idx=gs_idx,
                snp_logsed=np.float32(snp_logsed),
                extracted_ref=extracted_ref,
                snp_pos=pos, snp_ref=L["ref"], snp_alt=L["alt"],
                start=start, end=end, seq_out_start=seq_out_start, stride=stride,
                grid=grid, scan_positions=np.array(scan_positions),
                ref_bases=np.array(ref_bases),
                chrom=chrom, gene=L["gene"], window=np.array(L["window"]),
            )
            print(f"[{L['name']}] saved {out_dir}/{L['name']}.npz", flush=True)
        except Exception as e:
            print(f"[{L['name']}] ERROR: {e}", file=sys.stderr, flush=True)

    fasta.close()
    print("DONE_ISM", flush=True)


if __name__ == "__main__":
    main()
