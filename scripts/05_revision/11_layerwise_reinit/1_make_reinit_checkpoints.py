#!/usr/bin/env python3
"""Revision experiment 11, step 1 — build partially re-initialised Shorkie_LM checkpoints
.

The manuscript (L216-218) attributes the downstream gain to transfer of "exon-intron structure
and regulatory grammar". That specific attribution is hard to support: the improvement could
equally come from motif representations, or from more general inductive biases acquired during
pretraining. Selectively re-initialising the convolution layers versus the transformer blocks,
keeping the rest of the LM weights, gets at where the transferred information lives -- though it
stays hard to interpret if that information is distributed across the model.

The architecture makes the conv-versus-transformer split clean. From
``scripts/02_train/shorkie_lm/params.json`` the trunk is exactly three stages:

    conv_dna + res_tower x7   ->   transformer_tower x8   ->   unet_conv x7
    (convolutional tower)          (transformer blocks)       (U-Net decoder)

so there are three re-initialisable groups, not two -- the decoder is a third arm the
decoder is a third arm that is free to add and separates "local sequence features"
from "output reconstruction".

**How the groups are identified.** Not by guessing Keras layer names, which depend on
construction order, but from the model graph: the transformer group is every layer from the
first ``MultiHeadAttention`` to the last, the convolutional tower is everything before it,
and the decoder is everything after. That partition is derived at runtime and printed by
``--list_layers`` so it can be checked before any checkpoint is written.

**How re-initialisation is done.** A second SeqNN is built from the same config with fresh
random weights, and the pretrained weights are copied in for every layer OUTSIDE the target
group. So the retained layers are bit-exact Shorkie_LM and the reset layers are drawn from
the architecture's own initialisers -- identical to what a from-scratch model would start
from.

Writes ``results/checkpoints/<arm>.h5`` for each requested arm, plus a layer-partition
report. CPU is sufficient (no training here); a GPU-capable TensorFlow build is not needed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shorkie import config
from shorkie.models.ensemble import NUM_FEATURES

ARMS = {
    "conv_reset": "conv_tower",
    "transformer_reset": "transformer",
    "decoder_reset": "decoder",
}
GROUPS = ["conv_tower", "transformer", "decoder"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build partially re-initialised Shorkie_LM checkpoints.")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help=f"Comma-separated arms to build: {', '.join(ARMS)}")
    parser.add_argument("--list_layers", action="store_true",
                        help="Print the derived layer partition and exit")
    parser.add_argument("--lm_checkpoint", default=None,
                        help="[default: config models.shorkie_lm_checkpoint]")
    parser.add_argument("--lm_params", default=None,
                        help="[default: <models.shorkie_lm>/params.json]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def partition_layers(model):
    """Split the layers into conv_tower / transformer / decoder from the model graph.

    The transformer group runs from the first MultiHeadAttention layer to the last; the
    convolutional tower is everything before, the U-Net decoder everything after. This
    mirrors the trunk in params.json without depending on generated layer names.
    """
    layers = model.layers
    mha = [i for i, l in enumerate(layers)
           if "multiheadattention" in type(l).__name__.lower()
           or "multihead_attention" in l.name.lower()]
    if not mha:
        sys.exit("error: no MultiHeadAttention layer found; cannot locate the "
                 "transformer tower (is this the unet_small_bert_drop architecture?)")
    first, last = min(mha), max(mha)
    group = {}
    for i, l in enumerate(layers):
        if i < first:
            group[l.name] = "conv_tower"
        elif i <= last:
            group[l.name] = "transformer"
        else:
            group[l.name] = "decoder"
    return group

def build(params, checkpoint=None):
    from baskerville import seqnn
    prm = json.loads(Path(params).read_text())
    prm["model"]["num_features"] = NUM_FEATURES
    model = seqnn.SeqNN(prm["model"])
    if checkpoint:
        model.restore(str(checkpoint), trunk=False, by_name=False)
    return model

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    ckpt_dir = out_dir / "checkpoints"
    ckpt = Path(args.lm_checkpoint) if args.lm_checkpoint \
        else config.path("models.shorkie_lm_checkpoint")
    params = Path(args.lm_params) if args.lm_params \
        else config.path("models.shorkie_lm") / "params.json"
    for p, what in ((ckpt, "LM checkpoint"), (params, "LM params.json")):
        if p is None or not Path(p).exists():
            sys.exit(f"error: {what} not found at {p}\n"
                     "       fetch it with `bash data/download.sh --models lm`")

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)

    pretrained = build(params, ckpt)
    group = partition_layers(pretrained.model)
    counts = pd.Series(group).value_counts().reindex(GROUPS).fillna(0).astype(int)
    weighted = {g: 0 for g in GROUPS}
    for layer in pretrained.model.layers:
        weighted[group[layer.name]] += int(
            sum(int(np.prod(w.shape)) for w in layer.weights))

    report = pd.DataFrame({
        "group": GROUPS,
        "layers": [int(counts.get(g, 0)) for g in GROUPS],
        "parameters": [weighted[g] for g in GROUPS],
    })
    report["pct_parameters"] = (100 * report.parameters / report.parameters.sum()).round(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "layer_partition.csv", index=False)
    pd.Series(group, name="group").rename_axis("layer").to_csv(
        out_dir / "layer_assignment.csv")
    print(report.to_string(index=False))
    print(f"\ntotal parameters: {report.parameters.sum():,}")
    if args.list_layers:
        for name, g in group.items():
            print(f"  {g:12s} {name}")
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            print(f"SKIPPED: unknown arm '{arm}'", file=sys.stderr)
            continue
        target = ARMS[arm]
        fresh = build(params)                      # fresh random initialisation
        copied = reset = 0
        for src, dst in zip(pretrained.model.layers, fresh.model.layers):
            if group[src.name] == target:
                reset += 1                          # leave the fresh weights in place
                continue
            if src.weights:
                dst.set_weights(src.get_weights())
                copied += 1
        path = ckpt_dir / f"{arm}.h5"
        fresh.model.save_weights(str(path))
        print(f"{arm:18s} reset {reset:3d} layers ({target}), copied {copied:3d} "
              f"-> {path}", flush=True)

    print(f"\nwrote {ckpt_dir}")
    print("Next: 2_train_arms.sh fine-tunes each arm with the published supervised recipe.")

if __name__ == "__main__":
    main()
