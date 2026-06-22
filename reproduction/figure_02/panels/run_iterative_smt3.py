#!/usr/bin/env python3
"""Figure 2 deep-recheck — GPU iterative-masking inference for panels 2A-iterative
and 2B (PPM reconstruction).

Reproduces the manuscript method (Fig 2A/B): *"we randomly mask 15% of the
positions, predict the masked bases, and repeatedly mask and predict a new set of
positions until all positions are covered."* The masking convention is taken
verbatim from the Shorkie LM evaluator
(`baskerville/scripts/hound_eval_mlm_perplexity_region.py`, the same code that
produced the Figure-1G perplexities): masked positions have their 4 base channels
zeroed and the mask channel (index 4) set to 1; the model output `[..., :4]` is the
per-position predicted A/C/G/T probability.

Model + locus follow `scripts/04_analysis/shorkie_lm/lm_SMT3_viz/3_inference_smt3_unmasked.py`
(Shorkie LM `unet_small_bert_drop`, SMT3 = YDR510W windows on chrIV, R64 species idx 9).

Outputs `reproduced/iterative_smt3/preds_smt3_iterative.npz`:
  x_true            (W, L, 4)  one-hot truth per window
  x_pred_iter       (W, L, 4)  iterative-reconstruction predicted probabilities
  iter_assignment   (W, L)     int: which iteration masked each position
  label             (W,)       species index
plus the SMT3 promoter sub-window offsets for plotting.

Run on GPU via panels/run_iterative_smt3.sbatch (a100). Deterministic: np.random.seed(0).
"""
import os
import json
import numpy as np
import pysam
from baskerville import seqnn

from shorkie import config

WORK_ROOT = str(config.path("work_root"))
SEED = 0

MODEL_DIR = f"{WORK_ROOT}/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train"
MODEL_FILE = os.path.join(MODEL_DIR, "model_best.h5")
PARAMS_FILE = os.path.dirname(MODEL_DIR) + "/params.json"
FASTA_FILE = f"{WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_saccharomycetales_gtf/fasta/GCA_000146045_2.cleaned.fasta.masked.dust.softmask"

OUT_DIR = os.path.join(str(config.repo_root()), "reproduction", "figure_02", "reproduced", "iterative_smt3")

SPECIES_INDEX = 9          # GCA_000146045_2 (R64) index, per 3_inference_smt3_unmasked.py
NUM_SPECIES = 165
SEQ_LENGTH = 16384

# SMT3 (YDR510W) windows on chrIV (same as the unmasked inference script)
SMT3_WINDOWS = [
    ("chrIV", 1454592, 1470976),
    ("chrIV", 1458688, 1475072),
    ("chrIV", 1462784, 1479168),
    ("chrIV", 1466880, 1483264),
]
# published 2A sub-window: chrIV:1,469,090-1,469,198 (the SMT3 promoter)
SMT3_PROMOTER = (1469090, 1469198)


def dna_1hot(seq):
    seq = seq.upper()
    code = np.zeros((len(seq), 4), dtype="float32")
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, nt in enumerate(seq):
        if nt in m:
            code[i, m[nt]] = 1.0
    return code


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    with open(PARAMS_FILE) as f:
        params = json.load(f)
    params_model = params["model"]
    params_train = params["train"]
    params_model["num_features"] = NUM_SPECIES + 5 if params_train["loss"] == "mlm" else 4
    mask_rate = params_train["mask_rate"]
    mask_size = int(mask_rate * SEQ_LENGTH)
    print(f"mask_rate={mask_rate} mask_size={mask_size} -> ~{int(np.ceil(SEQ_LENGTH/mask_size))} iterations/window")

    print("Initializing + restoring model:", MODEL_FILE)
    model = seqnn.SeqNN(params_model)
    model.restore(MODEL_FILE, 0)

    fasta = pysam.Fastafile(FASTA_FILE)
    label_vec = np.zeros(NUM_SPECIES, dtype="float32")
    label_vec[SPECIES_INDEX] = 1.0

    x_trues, x_preds, assigns = [], [], []
    for wi, (chrom, start, end) in enumerate(SMT3_WINDOWS):
        seq = fasta.fetch(chrom, start, end)
        if len(seq) < SEQ_LENGTH:
            seq = seq + "N" * (SEQ_LENGTH - len(seq))
        else:
            seq = seq[:SEQ_LENGTH]
        x = dna_1hot(seq)  # (L,4)

        # build full input: [DNA(4) | mask(1)=0 | species(165)]
        x_inp = np.concatenate([
            x[np.newaxis],
            np.zeros((1, SEQ_LENGTH, 1), dtype="float32"),
            np.tile(label_vec[np.newaxis, np.newaxis, :], (1, SEQ_LENGTH, 1)),
        ], axis=-1)

        # shuffle positions into mask_size chunks; pad last chunk so all covered
        inds = np.arange(SEQ_LENGTH, dtype="int32")
        np.random.shuffle(inds)
        if SEQ_LENGTH % mask_size > 0:
            missing = mask_size - (SEQ_LENGTH % mask_size)
            extra = np.arange(SEQ_LENGTH, dtype="int32")
            np.random.shuffle(extra)
            inds = np.concatenate([inds, extra[:missing]], axis=0)

        x_pred = np.zeros((SEQ_LENGTH, 4), dtype="float32")
        assign = np.full(SEQ_LENGTH, -1, dtype="int32")
        b_pred = np.zeros(SEQ_LENGTH, dtype=bool)
        it = 0
        cur = inds
        while cur.shape[0] > 0:
            ind = cur[:mask_size]
            cur = cur[mask_size:]
            xm = np.copy(x_inp)
            xm[0, ind, :4] = 0.0
            xm[0, ind, 4] = 1.0
            yp = model.model.predict([xm], batch_size=1, verbose=False)[..., :4]
            for j in ind:
                if not b_pred[j]:
                    x_pred[j] = yp[0, j]
                    assign[j] = it
                    b_pred[j] = True
            it += 1
        print(f"window {wi} {chrom}:{start}-{end} -> {it} iterations, "
              f"recon mean max-prob={x_pred.max(axis=1).mean():.3f}")
        x_trues.append(x)
        x_preds.append(x_pred)
        assigns.append(assign)

    out = os.path.join(OUT_DIR, "preds_smt3_iterative.npz")
    np.savez_compressed(
        out,
        x_true=np.array(x_trues, dtype="float16"),
        x_pred_iter=np.array(x_preds, dtype="float16"),
        iter_assignment=np.array(assigns, dtype="int32"),
        label=np.array([SPECIES_INDEX] * len(SMT3_WINDOWS), dtype="int32"),
        windows=np.array(SMT3_WINDOWS, dtype=object),
        smt3_promoter=np.array(SMT3_PROMOTER, dtype="int64"),
        seq_length=SEQ_LENGTH,
        mask_rate=mask_rate,
        seed=SEED,
    )
    print("[OK] saved", out)


if __name__ == "__main__":
    main()
