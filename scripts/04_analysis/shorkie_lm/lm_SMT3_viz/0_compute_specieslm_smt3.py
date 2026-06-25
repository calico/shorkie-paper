#!/usr/bin/env python3
"""Figure 2A — regenerate the SpeciesLM (fungi) reconstruction PWM for the SMT3 promoter.

The published Figure 2A's top row is the external SpeciesLM model's per-position DNA
reconstruction over the SMT3 (YDR510W) 1 kb 5'-upstream sequence. The repo's cached
`dependencies_DNALM/all_prbs.npy` is the WRONG locus (it is byte-identical to
`all_prbs_chrX_607855_608355.npy`, a chrX:607,855-608,355 example) — so the SMT3
SpeciesLM PWM was never saved. This script regenerates it faithfully by invoking the
released HuggingFace checkpoint, exactly as the upstream authors' notebook does
(`dependencies_DNALM/compute_SMT3_gene.ipynb` /
`compute_and_visualize_dep_maps_RPL43B.py`):

  all_prbs = softmax( BertForMaskedLM( tokenizer(proxy_species + ' '.join(seq)) ).logits )
             [:, special_token_slice, acgt_idxs]            # -> (1003, 4) reconstruction PWM

We need only the real (unmutated) sequence's reconstruction — one forward pass, no
3*1003 mutation sweep — so this is fast (CPU is fine).

Model:   johahi/specieslm-fungi-upstream-k1   (Tomaz da Silva / Karollus et al., external)
Proxy:   kazachstania_africana_cbs_2517_gca_000304475   (the authors' fungi proxy species)
Output:  reproduction/figure_02/reproduced/specieslm_smt3/all_prbs_SMT3.npy   (1003, 4)
         (+ a copy into <work_root>/experiments/dependencies_DNALM/ if resolvable)

Run in an env with torch + transformers (this repo: conda env `pytorch_cuda`):
  conda activate pytorch_cuda
  python scripts/04_analysis/shorkie_lm/lm_SMT3_viz/0_compute_specieslm_smt3.py
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np

# --- SMT3 (YDR510W) 1 kb 5'-upstream sequence (ends in the ATG start codon) ----------
# Identical to compute_SMT3_gene.ipynb `SMT3_five_prime_seq` and
# 5_verify_alignment.py `SPECIES_SEQ_STRING`. The published SpeciesLM logo region is [690:986].
SMT3_FIVE_PRIME_SEQ = (
    "GCTTCCCTCATTATTCCGCCCATGGCGTCTATTACCAAGCGTCATAATGTGCAATATTTGATATTATATAAGCTACTTGAGAAAG"
    "CGATAGTTTTTTTTTCTTACACAAAAAAAAAAAAACATAAAGCACCTATAACTCTCAACTTTGAAGAAGCACGAAAGGAATATGT"
    "TTAAATCAACAGAAATGTGAAAAAAATCGGTTATATATACAGAATCCGATTCTTTCTAACATCAAAGAGGTGGGGGAAGAAGGGA"
    "CTCAAAAAAGAAACGACACTGCACAACCCGAGCCAAACTGACATACGAACACTAAAACCGATTTCCGAAAAAAACTTCAAATTTA"
    "CATTTCATTGTCCGTCTGCCATCGCATCATCGCCTTCATCTCTAAGAGTTGCCGTGCCTTTCCATCCGCTTTCTTTTCATGCGGC"
    "GTTATTCTTTTTTCCTATTTTTGATGGTCCCTGTGCCGTTTCTTTTTCATGTTCACCGGTTTTTGGCGCCGCATACCGTACGGCG"
    "GGGCACTTTTGAAACGTTTTTGTGCATCCTGATGCCGTTTTCAAGGATCGCAAGCACGTCGCATAATACGGTAATGCCGAATTAA"
    "GGCTACGTCGTCATAGTAGGTTAGTCATGCGCGTTGGAAAAAGAAATGACCAACGCGTTGATTACGTAGTCCCCAAGGAATAATG"
    "CTTTTGAAAGTGAAAAAAAAAAATAAAACTGAAAAAAGCCATGCTGTTTCCATCACGTGCATGTCACGTTTTTGCCGCCGAACTC"
    "TTTGATCATGTGATATGAATATGTTGGGTTACCCAGCTTTGCCAACACGCGCCGTCGGAAGGTGTTCAGGAAGCAGGAAAAGAGC"
    "AAAACACCAACAATCAAACAAACGAACACATTCTACTCTTTTAGTTGATTTTTCTTACCTTTTCCAAGCTCCCGTTTCTTGTTAC"
    "CACCTGTAGCATATAGGACAGAAGGACCCAGTTCAGTTCTAGTTTTACAAATAAATACACGAGCGATG"
)

MODEL_NAME = "johahi/specieslm-fungi-upstream-k1"
PROXY_SPECIES = "kazachstania_africana_cbs_2517_gca_000304475"
PUB_S, PUB_E = 690, 986     # the published SpeciesLM plot region within the 1003 bp seq
ALPH = np.array(list("ACGT"))


def repo_root_from_file() -> Path:
    # .../scripts/04_analysis/shorkie_lm/lm_SMT3_viz/0_compute_specieslm_smt3.py -> repo root (5 up)
    return Path(__file__).resolve().parents[4]


def resolve_work_root() -> str | None:
    """Best-effort work_root (only for the optional dependencies_DNALM copy)."""
    if os.environ.get("SHORKIE_WORK_ROOT"):
        return os.environ["SHORKIE_WORK_ROOT"]
    try:
        from shorkie import config  # available in yeast_ml, maybe not in pytorch_cuda
        return str(config.path("work_root"))
    except Exception:
        return None


def seq_of(pwm: np.ndarray) -> str:
    return "".join(ALPH[i] for i in np.asarray(pwm).argmax(1))


def agreement(pwm: np.ndarray, ref: str) -> float:
    s = seq_of(pwm)
    n = min(len(s), len(ref))
    return float(np.mean([s[i] == ref[i] for i in range(n)])) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--work-root", default=None, help="override work_root for the optional copy")
    args = ap.parse_args()

    import torch
    from transformers import BertForMaskedLM, AutoTokenizer

    seq = SMT3_FIVE_PRIME_SEQ.upper()
    L = len(seq)
    print(f"[info] SMT3_five_prime_seq length = {L} (ends '{seq[-6:]}'; published region [{PUB_S}:{PUB_E}])")
    assert L == 1003, f"expected 1003 bp, got {L}"

    print(f"[info] loading {MODEL_NAME} (trust_remote_code) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(args.device).eval()

    vocab = tokenizer.get_vocab()
    assert PROXY_SPECIES in vocab, f"proxy species {PROXY_SPECIES!r} not in tokenizer vocab"
    acgt_idxs = [vocab[n] for n in ["A", "C", "G", "T"]]
    print(f"[info] proxy_species in vocab; acgt token ids = {acgt_idxs}")

    def forward_probs(text: str) -> np.ndarray:
        enc = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(args.device)).logits.float().cpu()
        probs = torch.softmax(logits, dim=-1)[0, :, acgt_idxs].numpy()  # (n_tokens, 4)
        return probs

    # Replicate the authors' tokenization exactly: proxy_species + ' '.join(list(seq)).
    # Try a couple of (tokenization, special-token slice) candidates and pick the one whose
    # argmax best matches the true SMT3 sequence (guards against an off-by-one token offset).
    candidates = []
    for tok_label, text in [
        ("joined", PROXY_SPECIES + " ".join(list(seq))),
        ("spaced", PROXY_SPECIES + " " + " ".join(list(seq))),
    ]:
        probs = forward_probs(text)
        for lo in (1, 2):
            for hi in (-1, -2):
                sl = probs[lo:hi, :]
                if sl.shape[0] != L:
                    continue
                sl = sl + 1e-10
                sl = sl / sl.sum(axis=1, keepdims=True)
                candidates.append((tok_label, lo, hi, agreement(sl, seq), sl))

    if not candidates:
        raise RuntimeError("no (tokenization, slice) candidate produced a length-1003 PWM")

    candidates.sort(key=lambda c: c[3], reverse=True)
    print("[info] candidate (tokenization, slice) -> argmax-vs-SMT3 agreement:")
    for tok_label, lo, hi, agr, _ in candidates:
        print(f"        {tok_label:6s} [{lo}:{hi}]  agreement={agr:.4f}")
    tok_label, lo, hi, best_agr, all_prbs = candidates[0]
    print(f"[info] chosen: {tok_label} [{lo}:{hi}] -> all_prbs shape {all_prbs.shape}, "
          f"argmax-vs-SMT3 agreement = {best_agr:.4f}")
    if best_agr < 0.80:
        print(f"[WARN] agreement {best_agr:.4f} < 0.80 — region/token slicing may be off; "
              f"inspect before trusting the figure.")

    # Sanity print of the published region argmax vs truth
    print(f"[check] all_prbs[{PUB_S}:{PUB_E}] argmax: {seq_of(all_prbs[PUB_S:PUB_E])[:50]}")
    print(f"[check] SMT3_seq[{PUB_S}:{PUB_E}]      : {seq[PUB_S:PUB_E][:50]}")

    # --- write outputs ---
    repo = repo_root_from_file()
    out_dir = repo / "reproduction" / "figure_02" / "reproduced" / "specieslm_smt3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_prbs_SMT3.npy"
    np.save(out_path, all_prbs.astype(np.float32))
    print(f"[OK] wrote {out_path}  ({all_prbs.shape}, agreement={best_agr:.4f})")

    work_root = args.work_root or resolve_work_root()
    if work_root:
        dep_dir = Path(work_root) / "experiments" / "dependencies_DNALM"
        if dep_dir.is_dir():
            dep_path = dep_dir / "all_prbs_SMT3.npy"
            np.save(dep_path, all_prbs.astype(np.float32))
            print(f"[OK] also wrote {dep_path}")

    # tiny provenance sidecar
    meta = out_dir / "all_prbs_SMT3.meta.txt"
    meta.write_text(
        f"model={MODEL_NAME}\nproxy_species={PROXY_SPECIES}\nseq_len={L}\n"
        f"tokenization={tok_label}\nspecial_token_slice=[{lo}:{hi}]\n"
        f"argmax_vs_SMT3_agreement={best_agr:.4f}\npublished_region=[{PUB_S}:{PUB_E}]\n"
    )
    print(f"[OK] wrote {meta}")


if __name__ == "__main__":
    main()
