#!/usr/bin/env python3
"""Figure 2A row 3 ("Shorkie LM 15% iterative inference") — extract the gene-averaged
SMT3 upstream PWM from the PRECOMPUTED 3-architecture LM ensemble.

The published row 3 is NOT a single-model freshly-run GPU job. It is the masked-LM's
15%-iterative reconstruction, averaged over the **3 architectures** the LM was trained
as (`unet_small_bert_drop` + `_retry_1` + `_retry_2`) — exactly what
`2_viz_dna_pwm_shorkie_lm.py` plots. Those per-arch predictions are already on disk as
the standard eval outputs `preds_train.npz` (their argmax matches the genome at ~0.42 =
the 15%-iterative reconstruction; the *unmasked* row 2 is ~0.96).

This script reuses those precomputed scores (no GPU): it finds the SMT3 (YDR510W) gene
windows in `preds_train.npz`, averages the 3-arch `x_pred` over their 512 bp upstream,
and writes the gene-averaged iterative upstream PWM that `build_2A_logos.py` renders.

SMT3 (chrIV) is in the TRAIN set (test=chrXI/XIII/XV, valid=chrXII/XIV/XVI).

Output: reproduction/figure_02/reproduced/iterative_smt3/preds_smt3_iterative_3arch.npz
Run (env yeast_ml): python scripts/04_analysis/shorkie_lm/lm_SMT3_viz/6_extract_iterative_3arch.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from shorkie import config

WORK = str(config.path("work_root"))
REPO = Path(config.repo_root())
OUT_DIR = REPO / "reproduction" / "figure_02" / "reproduced" / "iterative_smt3"

LM = (f"{WORK}/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/"
      f"LM_Johannes/lm_saccharomycetales_gtf")
ARCHS = ["unet_small_bert_drop", "unet_small_bert_drop_retry_1", "unet_small_bert_drop_retry_2"]
PREDS = "test_trainset/preds_train.npz"          # SMT3/chrIV is in the train set

ATG = 1469400          # SMT3 (YDR510W) start codon, + strand
PAD = 512              # 512 bp upstream (same as 2_viz / 4_viz / build_2A)
SMT3_WINDOWS_START = [1454592, 1458688, 1462784, 1466880]   # the 4 SMT3 tiling windows

ALPH = np.array(list("ACGT"))


def seq_of(pwm):
    return "".join(ALPH[i] for i in np.asarray(pwm).argmax(-1))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # SMT3 windows' true one-hot sequence (defines which train rows are the SMT3 windows)
    uz = np.load(f"{WORK}/experiments/Shorkie_LM_SMT3_viz/inference_smt3_output/preds_smt3_unmasked.npz")
    xt_smt3 = np.asarray(uz["x_true"])                       # (4, 16384, 4)
    smt3_seqs = [seq_of(xt_smt3[w]) for w in range(4)]

    # locate the SMT3 windows in preds_train (exact argmax-string match; order shared across archs)
    base = np.load(f"{LM}/lm_saccharomycetales_gtf_{ARCHS[0]}/{PREDS}")
    xt_tr = np.asarray(base["x_true"])                       # (N, 16384, 4)
    tr_seqs = [seq_of(xt_tr[i]) for i in range(xt_tr.shape[0])]
    matched = {}                                             # window index -> train row
    for w in range(4):
        hits = [i for i, s in enumerate(tr_seqs) if s == smt3_seqs[w]]
        if hits:
            matched[w] = hits[0]
    print(f"[info] SMT3 windows matched to preds_train rows: {matched} "
          f"({len(matched)}/4; window(s) not present are not train-bed rows)")
    if not matched:
        raise RuntimeError("no SMT3 window found in preds_train.npz")

    # 3-arch x_pred for the matched windows
    xpred_per_arch = []
    for a in ARCHS:
        z = np.load(f"{LM}/lm_saccharomycetales_gtf_{a}/{PREDS}")
        xpred_per_arch.append(np.asarray(z["x_pred"]))       # (N, 16384, 4)

    # gene-averaged upstream PWM: average the 3 archs per window, then average windows' 512bp upstream
    acc = np.zeros((PAD, 4)); per_window = []
    for w, row in sorted(matched.items()):
        off = ATG - SMT3_WINDOWS_START[w]                    # rel_start of ATG within the window
        win_3arch = np.mean([xpred_per_arch[a][row] for a in range(len(ARCHS))], axis=0)  # (16384,4)
        up = np.asarray(win_3arch[off - PAD:off, :], float)  # (512,4)
        acc += up
        per_window.append(up)
    upstream_pwm = acc / len(matched)

    # true upstream (for the agreement self-check), same windows
    acc_t = np.zeros((PAD, 4))
    for w in sorted(matched.keys()):
        off = ATG - SMT3_WINDOWS_START[w]
        acc_t += np.asarray(xt_smt3[w, off - PAD:off, :], float)
    true_up = acc_t / len(matched)

    SHO_S, SHO_E = 201, 311
    agree = float(np.mean([a == b for a, b in
                           zip(seq_of(upstream_pwm[SHO_S:SHO_E]), seq_of(true_up[SHO_S:SHO_E]))]))
    print(f"[info] 3-arch iterative agreement vs genome over [{SHO_S}:{SHO_E}] = {agree:.4f}")

    out = OUT_DIR / "preds_smt3_iterative_3arch.npz"
    np.savez_compressed(
        out,
        upstream_pwm=upstream_pwm.astype("float32"),
        per_window_upstream=np.asarray(per_window, dtype="float32"),
        matched_rows=np.asarray([matched[w] for w in sorted(matched)], dtype="int64"),
        matched_windows=np.asarray(sorted(matched), dtype="int64"),
        n_windows=len(matched),
        n_archs=len(ARCHS),
        archs=np.asarray(ARCHS, dtype=object),
        agreement_vs_genome=agree,
    )
    print(f"[OK] -> {out}  (upstream_pwm {upstream_pwm.shape}, {len(matched)} windows x {len(ARCHS)} archs)")


if __name__ == "__main__":
    main()
