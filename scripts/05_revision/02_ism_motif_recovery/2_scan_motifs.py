#!/usr/bin/env python3
"""Revision experiment 02, step 2 — locate transcription-factor motif occurrences in the
promoter windows that step 1 extracted saliency for.

Motif scanning is implemented natively rather than shelling out to FIMO, for two reasons:
the MEME suite is an optional dependency in this repository (Figure 4 panel H caches its
TomTom output precisely so the panel reproduces without it), and the scan needed here is a
simple log-odds PWM sweep that numpy does in seconds.

Method, matching what FIMO does:

  * PWMs are read from the yeast MEME database (``motif_db_dir``); the high-confidence
    merged file is used by default.
  * Log-odds are taken against the observed base composition of the promoter windows
    themselves, not a uniform background -- yeast promoters are strongly AT-rich and a
    uniform background inflates AT-rich motif hits.
  * Significance is calibrated per motif against a DINUCLEOTIDE-shuffled null
    (Altschul-Erikson Euler-path shuffle), which preserves the poly-A tracts and local
    composition that dominate these windows; a mononucleotide null would let those tracts
    masquerade as signal. The threshold is the (1 - p) quantile of the null score pool.

Both strands are scanned. Hits are written with window index, motif, strand, offset and
score, so step 3 can score saliency inside each occurrence against its own flanks.

Writes ``results/motif_hits.tsv`` and ``results/motif_scan_summary.csv``.
CPU only, a few minutes for 139 motifs x 397 windows.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shorkie import config

NT = "ACGT"
NT_IX = {c: i for i, c in enumerate(NT)}
# Motifs named in the paper's Figure 4 / Figure 2 discussion, reported first in the
# summary. RRPE is Stb3 and PAC is Dot6, following the manuscript's own labelling.
FOCUS = ["RAP1", "REB1", "SPT15", "CBF1", "ABF1", "MCM1", "FHL1", "SFP1",
         "UME6", "DOT6", "STB3", "TBF1", "MSN2", "MSN4", "SWI4", "RPN4"]
FOCUS_ALIAS = {"SPT15": "TBP (Spt15)", "STB3": "RRPE (Stb3)", "DOT6": "PAC (Dot6)"}

# Consensus motifs quoted in the manuscript itself, scanned alongside the database PWMs.
# This matters for RRPE in particular. It is the standard counter-example -- the motif where
# Shorkie_Random_Init does better, but the database Stb3 matrix finds fewer sites in these
# RP/TSS windows than the p=1e-4 threshold expects by chance, so that case would be
# untestable on the database matrix alone. The manuscript defines RRPE as
# 5'-TGAAAAATTTT-3' and PAC as 5'-GCGATGA(G)-3' (Results, RRB promoters), and the Figure 4D
# splicing consensuses are included so the same measurement covers the splicing panel.
CONSENSUS_MOTIFS = {
    "cons_RRPE": "TGAAAAATTTT",     # RRPE (Stb3), manuscript Results
    "cons_PAC": "GCGATGAG",         # PAC (Dot6), manuscript Results
    "cons_TATA": "TATAAA",          # TATA box
    "cons_donor": "GTATGT",         # 5' splice site, Figure 4D
    "cons_branch": "TACTAAC",       # branch point, Figure 4D
}
CONSENSUS_ALIAS = {
    "cons_RRPE": "RRPE consensus (TGAAAAATTTT)", "cons_PAC": "PAC consensus (GCGATGAG)",
    "cons_TATA": "TATA box (TATAAA)", "cons_donor": "5' splice donor (GTATGT)",
    "cons_branch": "branch point (TACTAAC)",
}
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "W": "AT",
         "S": "CG", "K": "GT", "M": "AC", "N": "ACGT"}

def consensus_pwm(consensus, strength=0.85):
    """Turn a IUPAC consensus string into a probability matrix."""
    rows = []
    for ch in consensus:
        allowed = IUPAC[ch]
        if len(allowed) == 4:
            rows.append(np.full(4, 0.25))
            continue
        p = np.full(4, (1.0 - strength) / (4 - len(allowed)))
        for a in allowed:
            p[NT_IX[a]] = strength / len(allowed)
        rows.append(p / p.sum())
    return np.array(rows)

def parse_args():
    parser = argparse.ArgumentParser(description="Scan promoter windows for TF motifs.")
    parser.add_argument("--meme", default=None,
                        help="MEME motif file [default: <motif_db_dir>/merged_meme_high_conf.meme]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--pvalue", type=float, default=1e-4,
                        help="Per-position significance threshold against the shuffled null")
    parser.add_argument("--n_shuffle", type=int, default=3,
                        help="Dinucleotide-shuffled replicates per window for the null")
    parser.add_argument("--min_sites", type=int, default=15,
                        help="Drop motifs with fewer than this many hits")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# ------------------------------------------------------------------ MEME parsing ----
def read_meme(path):
    """{name: probability matrix (w, 4)} from a MEME-format motif file."""
    motifs, name, rows, reading = {}, None, [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("MOTIF"):
            if name and rows:
                motifs[name] = np.array(rows, dtype=float)
            name, rows, reading = s.split()[1], [], False
            continue
        if s.startswith("letter-probability matrix"):
            reading = True
            continue
        if reading:
            parts = s.split()
            if len(parts) == 4:
                try:
                    rows.append([float(x) for x in parts])
                    continue
                except ValueError:
                    pass
            reading = False
    if name and rows:
        motifs[name] = np.array(rows, dtype=float)
    return motifs

# --------------------------------------------------------- dinucleotide shuffling ----
def dinuc_shuffle(seq, rng):
    """Altschul-Erikson dinucleotide shuffle: preserves every dinucleotide count.

    Builds the doublet graph, draws a random arborescence into the final vertex so the
    Euler walk is guaranteed to complete, shuffles the remaining out-edges, then walks.
    """
    if len(seq) < 3:
        return seq
    verts = sorted(set(seq))
    edges = {v: [] for v in verts}
    for a, b in zip(seq[:-1], seq[1:]):
        edges[a].append(b)
    last = seq[-1]

    # Random last-edge arborescence rooted at `last`.
    while True:
        last_edge, ok = {}, True
        for v in verts:
            if v == last:
                continue
            last_edge[v] = edges[v][rng.integers(len(edges[v]))]
        for v in verts:                              # every vertex must reach `last`
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in last_edge:
                    ok = False
                    break
                seen.add(cur)
                cur = last_edge[cur]
            if not ok:
                break
        if ok:
            break

    order = {}
    for v in verts:
        rest = list(edges[v])
        if v in last_edge:
            rest.remove(last_edge[v])
        rng.shuffle(rest)
        order[v] = rest + ([last_edge[v]] if v in last_edge else [])

    out, cur, ptr = [seq[0]], seq[0], {v: 0 for v in verts}
    for _ in range(len(seq) - 1):
        nxt = order[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)

# ------------------------------------------------------------------- PWM scanning ----
def encode(seq):
    return np.array([NT_IX.get(c, -1) for c in seq], dtype=np.int8)

def log_odds(pwm, background, pseudocount=1e-3):
    p = (pwm + pseudocount) / (pwm + pseudocount).sum(axis=1, keepdims=True)
    return np.log2(p / background[None, :])

def scan(codes, lom):
    """Best-strand score at every start position of a single encoded sequence."""
    w = lom.shape[0]
    n = len(codes) - w + 1
    if n <= 0:
        return np.full(0, -np.inf), np.full(0, "+", dtype="<U1")
    rc = lom[::-1, ::-1]
    fwd = np.zeros(n, dtype=np.float32)
    rev = np.zeros(n, dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    for j in range(w):
        col = codes[j:j + n]
        bad = col < 0
        safe = np.where(bad, 0, col)
        fwd += lom[j][safe]
        rev += rc[j][safe]
        valid &= ~bad
    best = np.maximum(fwd, rev)
    best[~valid] = -np.inf
    return best, np.where(fwd >= rev, "+", "-")

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = out_dir / "saliency_cache.npz"
    if not cache.exists():
        sys.exit(f"error: {cache} not found -- run 1_extract_ism_saliency.py first")
    seqs = [str(s) for s in np.load(cache, allow_pickle=True)["sequences"]]
    print(f"windows: {len(seqs)} x {len(seqs[0])} bp", flush=True)

    meme = Path(args.meme) if args.meme else \
        Path(config.path("motif_db_dir")) / "merged_meme_high_conf.meme"
    if not meme.exists():
        sys.exit(f"error: motif file not found: {meme}\n"
                 "       set --meme or motif_db_dir in config/paths.yaml")
    motifs = read_meme(meme)
    n_db = len(motifs)
    for key, cons in CONSENSUS_MOTIFS.items():
        motifs[key] = consensus_pwm(cons)
    print(f"motifs : {n_db} from {meme.name} + {len(CONSENSUS_MOTIFS)} manuscript "
          f"consensus motifs", flush=True)

    counts = np.zeros(4, dtype=float)
    for s in seqs:
        for c in s:
            if c in NT_IX:
                counts[NT_IX[c]] += 1
    background = counts / counts.sum()
    print("background ACGT: " + " ".join(f"{b:.3f}" for b in background), flush=True)

    rng = np.random.default_rng(args.seed)
    codes = [encode(s) for s in seqs]
    null_codes = [encode(dinuc_shuffle(s, rng))
                  for _ in range(args.n_shuffle) for s in seqs]

    rows, summary = [], []
    for mi, (name, pwm) in enumerate(sorted(motifs.items()), start=1):
        lom = log_odds(pwm, background)
        null = np.concatenate([scan(c, lom)[0] for c in null_codes])
        null = null[np.isfinite(null)]
        if null.size == 0:
            continue
        thr = float(np.quantile(null, 1.0 - args.pvalue))
        n_hits = 0
        for wi, c in enumerate(codes):
            best, strand = scan(c, lom)
            for pos in np.flatnonzero(best >= thr):
                rows.append((wi, name, int(pos), int(pos + pwm.shape[0]),
                             strand[pos], round(float(best[pos]), 3)))
                n_hits += 1
        summary.append(dict(motif=name, width=pwm.shape[0], threshold=round(thr, 3),
                            hits=n_hits, hits_per_window=round(n_hits / len(seqs), 3)))
        if mi % 25 == 0:
            print(f"  scanned {mi}/{len(motifs)} motifs", flush=True)

    hits = pd.DataFrame(rows, columns=["window", "motif", "start", "end", "strand", "score"])
    sm = pd.DataFrame(summary).sort_values("hits", ascending=False)
    keep = set(sm[sm.hits >= args.min_sites].motif)
    hits = hits[hits.motif.isin(keep)]
    hits.to_csv(out_dir / "motif_hits.tsv", sep="\t", index=False)
    sm["kept"] = sm.motif.isin(keep)
    sm.to_csv(out_dir / "motif_scan_summary.csv", index=False)

    print(f"\ntotal hits: {len(hits):,} across {hits.motif.nunique()} motifs "
          f"(>= {args.min_sites} sites)")
    named = FOCUS + list(CONSENSUS_MOTIFS)
    focus = sm[sm.motif.isin(named)].copy()
    focus["label"] = focus.motif.map(lambda m: CONSENSUS_ALIAS.get(m, FOCUS_ALIAS.get(m, m)))
    print("\nmotifs named in the paper:")
    print(focus[["label", "width", "hits", "hits_per_window", "kept"]].to_string(index=False))
    print(f"\nwrote {out_dir/'motif_hits.tsv'}\nwrote {out_dir/'motif_scan_summary.csv'}")

if __name__ == "__main__":
    main()
