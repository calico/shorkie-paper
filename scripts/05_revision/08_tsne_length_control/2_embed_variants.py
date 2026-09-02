#!/usr/bin/env python3
"""Revision experiment 08, step 2 — recompute Figure 2E embeddings under pooling and
sampling schemes that remove the length confound.

Step 1 shows the confound is real: element length alone classifies the five classes at
~81% accuracy against a 20% chance baseline, and the two best-separated classes in the
published projection (tRNA, silhouette 0.85; transposable element, 0.64) are precisely the
length extremes. This step asks what survives once length is controlled.

Four embedding schemes over the same intervals and the same model layer:

  published      centre-pad with N to 16,384 and mean-pool the FULL padded axis
                 (exactly ``umap_cluster_promoter/1_predict_seqs_LM.py``: lines 117-119
                 pad, line 173 pools) -- the baseline to reproduce
  masked_pool    identical inputs, but pool ONLY over real (unpadded) positions, so the
                 embedding is no longer scaled by (real length / 16,384)
  length_matched every interval truncated/centred to a fixed 500 bp before padding, so all
                 five classes contribute the same amount of real sequence
  residualised   the published embedding with each dimension linearly regressed on
                 log(length) and the residual kept

The four differ in exactly one thing each, so step 3 can attribute what changes. Note that
`masked_pool` and `length_matched` are not redundant: the first removes the pooling
artifact while leaving genuine length differences in the sequence content, the second
removes both.

Writes ``results/embeddings_<scheme>.npz`` plus ``results/embedding_meta.csv``.

GPU strongly recommended. Cost scales with ``--max_per_class`` (default 400 intervals per
class, 2,000 forward passes total through Shorkie_LM).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from shorkie import config
from shorkie.models.ensemble import NUM_FEATURES, SCEREVISIAE_COL

SEQ_LEN = 16384
PROMOTER_BP = 500
FIXED_BP = 500                # length_matched scheme: common real-sequence budget
# The first self-attention layer, as in the published panel (Methods: ten layers are
# extracted; Figure 2E uses the first).
DEFAULT_LAYER = "multihead_attention"
BIOTYPE_CLASS = {"protein_coding": "Protein-coding gene", "tRNA": "tRNA",
                 "transposable_element": "Transposable element"}
SCHEMES = ["published", "masked_pool", "length_matched", "residualised"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute Figure 2E embeddings under length-controlled schemes.")
    parser.add_argument("--layer", default=DEFAULT_LAYER, help="Layer to tap")
    parser.add_argument("--max_per_class", type=int, default=400,
                        help="Intervals sampled per element class")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gtf", default=None, help="[default: config genome.gtf]")
    parser.add_argument("--fasta", default=None, help="[default: config genome.fasta]")
    parser.add_argument("--lm_checkpoint", default=None,
                        help="[default: config models.shorkie_lm_checkpoint]")
    parser.add_argument("--lm_params", default=None,
                        help="[default: <models.shorkie_lm>/params.json]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def collect_intervals(gtf_path, max_per_class, rng):
    """The five Figure-2E interval classes, sampled to a common count."""
    genes, chrom_end = [], defaultdict(int)
    for line in open(gtf_path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        chrom_end[f[0]] = max(chrom_end[f[0]], int(f[4]))
        if f[2] != "gene":
            continue
        m = re.search(r'gene_biotype "([^"]+)"', f[8])
        genes.append((f[0], int(f[3]) - 1, int(f[4]), f[6],
                      m.group(1) if m else "unknown"))

    rows = []
    for chrom, s, e, strand, biotype in genes:
        cls = BIOTYPE_CLASS.get(biotype)
        if cls:
            rows.append(dict(chrom=chrom, start=s, end=e, strand=strand, feature=cls))
        if biotype == "protein_coding":
            ps, pe = (s - PROMOTER_BP, s) if strand == "+" else (e, e + PROMOTER_BP)
            if ps >= 0:
                rows.append(dict(chrom=chrom, start=ps, end=pe, strand=strand,
                                 feature="Promoter"))
    by_chrom = defaultdict(list)
    for chrom, s, e, _, _ in genes:
        by_chrom[chrom].append((s, e))
    for chrom, spans in by_chrom.items():
        merged, cursor = [], 0
        for s, e in sorted(spans):
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        for s, e in merged:
            if s > cursor:
                rows.append(dict(chrom=chrom, start=cursor, end=s, strand="+",
                                 feature="Intergenic region"))
            cursor = max(cursor, e)

    df = pd.DataFrame(rows)
    df = df[(df.end - df.start) > 0]
    return (df.groupby("feature", group_keys=False)
.apply(lambda g: g.sample(n=min(len(g), max_per_class),
                                        random_state=int(rng.integers(1 << 31))))
.reset_index(drop=True))

def revcomp(s):
    return s.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def build_input(fasta, chrom, start, end, strand, fixed_bp=None):
    """Centre-padded 16,384 x 170 input, and the mask of real (non-padding) positions.

    Mirrors 1_predict_seqs_LM.py: centre-trim if longer than the window, centre-pad with N
    if shorter, reverse-complement minus-strand intervals, then one-hot the four bases and
    set the S. cerevisiae species channel.
    """
    refs = set(fasta.references)
    key = chrom if chrom in refs else (f"chr{chrom}" if f"chr{chrom}" in refs else chrom)
    seq = fasta.fetch(key, max(0, start), end).upper()
    if strand == "-":
        seq = revcomp(seq)
    if fixed_bp is not None and len(seq) != fixed_bp:
        if len(seq) > fixed_bp:                       # centre-trim to the common budget
            off = (len(seq) - fixed_bp) // 2
            seq = seq[off:off + fixed_bp]
        else:                                          # too short to use at fixed length
            return None, None
    if len(seq) > SEQ_LEN:
        off = (len(seq) - SEQ_LEN) // 2
        seq = seq[off:off + SEQ_LEN]
    pad = SEQ_LEN - len(seq)
    left = pad // 2
    real = np.zeros(SEQ_LEN, dtype=bool)
    real[left:left + len(seq)] = True

    x = np.zeros((SEQ_LEN, NUM_FEATURES), dtype="float32")
    ix = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, ch in enumerate(seq):
        j = ix.get(ch)
        if j is not None:
            x[left + i, j] = 1.0
    x[:, SCEREVISIAE_COL] = 1.0
    return x, real

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    gtf = Path(args.gtf) if args.gtf else config.path("genome.gtf")
    fasta_path = Path(args.fasta) if args.fasta else config.path("genome.fasta")
    ckpt = Path(args.lm_checkpoint) if args.lm_checkpoint \
        else config.path("models.shorkie_lm_checkpoint")
    params = Path(args.lm_params) if args.lm_params \
        else config.path("models.shorkie_lm") / "params.json"
    for p, what in ((gtf, "GTF"), (fasta_path, "FASTA"), (ckpt, "LM checkpoint"),
                    (params, "LM params.json")):
        if p is None or not Path(p).exists():
            sys.exit(f"error: {what} not found at {p}")

    import json
    import tensorflow as tf
    from baskerville import seqnn

    rng = np.random.default_rng(args.seed)
    meta = collect_intervals(gtf, args.max_per_class, rng)
    meta["length"] = meta.end - meta.start
    print(f"intervals: {len(meta)}", flush=True)
    print(meta.feature.value_counts().to_string(), flush=True)

    prm = json.loads(Path(params).read_text())
    prm["model"]["num_features"] = NUM_FEATURES
    model = seqnn.SeqNN(prm["model"])
    model.restore(str(ckpt), trunk=False, by_name=False)
    try:
        layer = model.model.get_layer(args.layer)
    except ValueError:
        names = [l.name for l in model.model.layers][:40]
        sys.exit(f"error: layer '{args.layer}' not found. First layers: {names}")
    sub = tf.keras.Model(inputs=model.model.inputs, outputs=layer.output)

    fasta = pysam.Fastafile(str(fasta_path))
    embeddings = {s: [] for s in ("published", "masked_pool", "length_matched")}
    keep = []

    def pooled(acts, real_full):
        """Full-window mean pool and real-positions-only mean pool of one activation map."""
        n = acts.shape[0]
        # The activation axis is a whole-number division of SEQ_LEN for every layer in this
        # architecture (res_tower pools by 2 seven times: 16384 -> 128), so reshaping is
        # exact and marks a bin real if ANY of its bases are real — unlike strided
        # sampling, which marks a bin by its first base only.
        if SEQ_LEN % n == 0:
            real_ds = real_full.reshape(n, SEQ_LEN // n).any(axis=1)
        else:
            real_ds = real_full[::max(1, SEQ_LEN // n)][:n]
        full = acts.mean(axis=0)
        masked = acts[real_ds].mean(axis=0) if real_ds.any() else full
        return full, masked

    # Build inputs one CHUNK at a time: a single (16384, 170) float32 window is 10.6 MiB,
    # so materialising all of them would need tens of GiB. Batching still matters — one
    # predict() call per interval pays TF's setup cost thousands of times over.
    records = list(meta.itertuples())
    for c0 in range(0, len(records), args.batch_size):
        chunk = records[c0:c0 + args.batch_size]
        built = []
        for r in chunk:
            x_full, real_full = build_input(fasta, r.chrom, r.start, r.end, r.strand)
            if x_full is None:
                continue
            x_fix, _ = build_input(fasta, r.chrom, r.start, r.end, r.strand,
                                   fixed_bp=FIXED_BP)
            built.append((r.Index, x_full, real_full, x_fix))
        if not built:
            continue

        acts_full = sub.predict(np.stack([b[1] for b in built]), verbose=0)
        fixed_rows = [i for i, b in enumerate(built) if b[3] is not None]
        acts_fixed = (sub.predict(np.stack([built[i][3] for i in fixed_rows]), verbose=0)
                      if fixed_rows else None)
        fixed_lookup = {row: k for k, row in enumerate(fixed_rows)}

        for i, (idx, _, real_full, x_fix) in enumerate(built):
            full, masked = pooled(acts_full[i], real_full)
            embeddings["published"].append(full)
            embeddings["masked_pool"].append(masked)
            if x_fix is not None and acts_fixed is not None:
                embeddings["length_matched"].append(acts_fixed[fixed_lookup[i]].mean(axis=0))
            else:
                embeddings["length_matched"].append(np.full(full.shape, np.nan))
            keep.append(idx)
        print(f"  {len(keep)}/{len(records)} intervals", flush=True)

    meta = meta.loc[keep].reset_index(drop=True)
    meta.to_csv(out_dir / "embedding_meta.csv", index=False)

    pub = np.stack(embeddings["published"])
    # residualised: regress each dimension on log(length) and keep the residual
    L = np.log10(meta.length.to_numpy() + 1)
    A = np.column_stack([np.ones_like(L), L])
    coef, *_ = np.linalg.lstsq(A, pub, rcond=None)
    embeddings["residualised"] = pub - A @ coef

    for scheme in SCHEMES:
        arr = np.asarray(embeddings[scheme], dtype=np.float32)
        np.savez_compressed(out_dir / f"embeddings_{scheme}.npz",
                            embedding=arr, feature=meta.feature.to_numpy(),
                            length=meta.length.to_numpy())
        print(f"{scheme:15s} {arr.shape}", flush=True)
    print(f"\nwrote {out_dir}")

if __name__ == "__main__":
    main()
