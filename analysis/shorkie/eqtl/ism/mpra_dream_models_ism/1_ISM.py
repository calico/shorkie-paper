#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from prixfixe.autosome import (
    AutosomeDataProcessor,
    AutosomeFinalLayersBlock,
    AutosomePredictor
)
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

# -------------------------------------------------------------------
# Model setup (identical to your existing setup_model)
# -------------------------------------------------------------------
def setup_model(model_dir: str, cuda_device_id: int = 0):
    # reproducibility
    generator = torch.Generator().manual_seed(2147483647)

    # data processor (we only need its metadata)
    dataprocessor = AutosomeDataProcessor(
        path_to_training_data="data/demo_train.txt",
        path_to_validation_data="data/demo_val.txt",
        train_batch_size=512,
        batch_per_epoch=10,
        train_workers=8,
        valid_batch_size=4096,
        valid_workers=8,
        shuffle_train=True,
        shuffle_val=False,
        plasmid_path="data/plasmid.json",
        seqsize=150,
        generator=generator
    )
    # init internals
    next(dataprocessor.prepare_train_dataloader())

    # build model blocks
    first = BHIFirstLayersBlock(
        in_channels=dataprocessor.data_channels(),
        out_channels=320,
        seqsize=dataprocessor.data_seqsize(),
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout=0.2
    )
    core = BHICoreBlock(
        in_channels=first.out_channels,
        out_channels=320,
        seqsize=first.infer_outseqsize(),
        lstm_hidden_channels=320,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout1=0.2,
        dropout2=0.5
    )
    final = AutosomeFinalLayersBlock(
        in_channels=core.out_channels,
        seqsize=core.infer_outseqsize()
    )

    model = PrixFixeNet(first=first, core=core, final=final, generator=generator)
    # load weights
    state_dict = torch.load(os.path.join(model_dir, 'model_best.pth'),
                            map_location=torch.device(f"cuda:{cuda_device_id}"))
    model.load_state_dict(state_dict)

    # wrap predictor
    predictor = AutosomePredictor(
        model=model,
        model_pth=os.path.join(model_dir, 'model_best.pth'),
        device=torch.device(f"cuda:{cuda_device_id}")
    )
    return predictor

# -------------------------------------------------------------------
# ISM implementation
# -------------------------------------------------------------------
def run_ism(predictor, input_tsv, output_tsv, ref_or_alt, bases=('A','C','G','T')):
    """
    For each reference sequence in input_tsv, do single-base ISM across
    every position. Outputs a long-form TSV with columns:
      sid_index, Chr, ChrPos, tss_dist, pos, mut_base, delta
    """
    df = pd.read_csv(input_tsv, sep="\t")
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="ISM over sequences"):
        if ref_or_alt == "ref":
            ref_seq = row["final_ref_seq"].upper()
        elif ref_or_alt == "alt":
            ref_seq = row["final_alt_seq"].upper()
        seq_len = len(ref_seq)

        # predict on reference
        ref_pred = float(predictor.predict(ref_seq))

        for pos in range(seq_len):
            orig_nt = ref_seq[pos]
            for b in bases:
                if b == orig_nt:
                    continue
                # build mutant
                mut_seq = ref_seq[:pos] + b + ref_seq[pos+1:]
                mut_pred = float(predictor.predict(mut_seq))
                delta = mut_pred - ref_pred

                records.append({
                    "sid_index": row.get("sid_index", idx),
                    "Chr":         row.get("Chr", ""),
                    "ChrPos":      row.get("ChrPos", ""),
                    "tss_dist":    row.get("tss_dist", ""),
                    "pos":         pos,
                    "orig_base":   orig_nt,
                    "mut_base":    b,
                    "ref_pred":    ref_pred,
                    "mut_pred":    mut_pred,
                    "delta":       delta
                })

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_tsv, sep="\t", index=False)
    print(f"ISM complete: results written to {output_tsv}")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Perform single-nucleotide in silico mutagenesis (ISM)"
    )
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing prixfixe model_best.pth")
    parser.add_argument("--input", required=True,
                        help="Input TSV with final_ref_seq (and metadata)")
    parser.add_argument("--output", default="ism_results.tsv",
                        help="Output TSV for ISM deltas")
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA device ID (or use -1 for CPU)")
    parser.add_argument("--ref-or-alt", default="ref",
                        choices=["ref", "alt"],
                        help="Use reference or alternate sequence for ISM")        
    args = parser.parse_args()

    predictor = setup_model(args.model_dir, cuda_device_id=args.cuda)
    run_ism(predictor, args.input, args.output, args.ref_or_alt)

if __name__ == "__main__":
    main()
# 