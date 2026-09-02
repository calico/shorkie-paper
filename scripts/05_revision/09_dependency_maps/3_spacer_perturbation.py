#!/usr/bin/env python3
"""Revision experiment 09, step 3 — an orthogonal test of regulatory syntax: does the model
care about the SPACING between two motifs?

Dependency maps (steps 1-2) ask whether two positions covary. This asks the sharper
question the usual definition of grammar actually names -- "relationships between
motifs such as their spacing, multiplicity, and arrangement" -- by intervening on spacing
directly rather than reading it off a correlation.

For each pair of motif occurrences in a locus, the spacer between them is lengthened or
shortened by 1-20 bp (insertions drawn from a dinucleotide-matched shuffle of the existing
spacer, so base composition is held fixed and only the geometry changes), and the model's
predicted distribution over the DOWNSTREAM motif is re-read. Two signatures are then
looked for:

  * monotonic decay -- the model's confidence in the second motif falls as the pair is
    pushed apart, i.e. it has learned that they belong together;
  * helical phasing  -- a ~10.5 bp periodicity in the response, the signature of two
    factors that must sit on the same face of the DNA helix. This is the strongest
    available evidence of learned syntax, because it cannot arise from motif recognition
    alone.

A flat response means the model treats the two motifs independently: motifs as words, not
grammar. That is a legitimate and reportable outcome.

Writes ``results/spacer_response.csv`` and ``results/spacer_response.png``.
GPU recommended: ~40 forward passes per motif pair.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam

from shorkie import config
from shorkie.models.ensemble import NUM_FEATURES, SCEREVISIAE_COL

SEQ_LEN = 16384
NT = "ACGT"
NT_IX = {c: i for i, c in enumerate(NT)}
SHIFTS = list(range(-20, 21))          # bp added to (or removed from) the spacer
HELICAL_PERIOD = 10.5

def parse_args():
    parser = argparse.ArgumentParser(description="Spacer-perturbation test of motif syntax.")
    parser.add_argument("--meme", default=None,
                        help="[default: <motif_db_dir>/merged_meme_high_conf.meme]")
    parser.add_argument("--max_pairs", type=int, default=6,
                        help="Motif pairs per locus (closest pairs first)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fasta", default=None, help="[default: config genome.fasta]")
    parser.add_argument("--lm_checkpoint", default=None,
                        help="[default: config models.shorkie_lm_checkpoint]")
    parser.add_argument("--lm_params", default=None,
                        help="[default: <models.shorkie_lm>/params.json]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    parser.add_argument("--pvalue", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_scanner(repo):
    path = repo / "scripts/05_revision/02_ism_motif_recovery/2_scan_motifs.py"
    spec = importlib.util.spec_from_file_location("scan_motifs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def encode_window(seq):
    x = np.zeros((SEQ_LEN, NUM_FEATURES), dtype="float32")
    for i, ch in enumerate(seq[:SEQ_LEN]):
        j = NT_IX.get(ch)
        if j is not None:
            x[i, j] = 1.0
    x[:, SCEREVISIAE_COL] = 1.0
    return x

def motif_logprob(probs, start, end, seq):
    """Mean log2 probability the model assigns to the observed bases of a motif."""
    total, n = 0.0, 0
    for i in range(start, min(end, len(seq))):
        j = NT_IX.get(seq[i])
        if j is not None:
            total += float(np.log2(max(probs[i, j], 1e-10)))
            n += 1
    return total / max(n, 1)

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    maps_dir = out_dir / "dep_maps"
    maps = sorted(maps_dir.glob("*.npz")) if maps_dir.exists() else []
    if not maps:
        sys.exit(f"error: no loci under {maps_dir} -- run step 1 first")

    fasta_path = Path(args.fasta) if args.fasta else config.path("genome.fasta")
    ckpt = Path(args.lm_checkpoint) if args.lm_checkpoint \
        else config.path("models.shorkie_lm_checkpoint")
    params = Path(args.lm_params) if args.lm_params \
        else config.path("models.shorkie_lm") / "params.json"
    for p, what in ((fasta_path, "genome FASTA"), (ckpt, "LM checkpoint"),
                    (params, "LM params.json")):
        if p is None or not Path(p).exists():
            sys.exit(f"error: {what} not found at {p}")

    from baskerville import seqnn
    prm = json.loads(Path(params).read_text())
    prm["model"]["num_features"] = NUM_FEATURES
    model = seqnn.SeqNN(prm["model"])
    model.restore(str(ckpt), trunk=False, by_name=False)

    scan = load_scanner(repo)
    meme = Path(args.meme) if args.meme else \
        Path(config.path("motif_db_dir")) / "merged_meme_high_conf.meme"
    motifs = scan.read_meme(meme)
    for key, cons in scan.CONSENSUS_MOTIFS.items():
        motifs[key] = scan.consensus_pwm(cons)

    fasta = pysam.Fastafile(str(fasta_path))
    refs = set(fasta.references)
    rng = np.random.default_rng(args.seed)
    rows = []

    for path in maps:
        z = np.load(path, allow_pickle=True)
        locus_seq = str(z["sequence"])
        chrom_raw, start, end = str(z["chrom"]), int(z["start"]), int(z["end"])
        chrom = chrom_raw if chrom_raw in refs else (
            chrom_raw[3:] if chrom_raw[3:] in refs else chrom_raw)
        centre = (start + end) // 2
        win_start = max(0, centre - SEQ_LEN // 2)
        window = fasta.fetch(chrom, win_start, win_start + SEQ_LEN).upper()
        window = window + "N" * max(0, SEQ_LEN - len(window))
        offset = start - win_start

        counts = np.zeros(4)
        for c in locus_seq:
            if c in NT_IX:
                counts[NT_IX[c]] += 1
        background = counts / max(counts.sum(), 1)
        codes = scan.encode(locus_seq)
        null_codes = [scan.encode(scan.dinuc_shuffle(locus_seq, rng)) for _ in range(3)]
        hits = []
        for name, pwm in motifs.items():
            lom = scan.log_odds(pwm, background)
            null = np.concatenate([scan.scan(c, lom)[0] for c in null_codes])
            null = null[np.isfinite(null)]
            if null.size == 0:
                continue
            thr = float(np.quantile(null, 1.0 - args.pvalue))
            best, _ = scan.scan(codes, lom)
            for pos in np.flatnonzero(best >= thr):
                hits.append((name, int(pos), int(pos + pwm.shape[0])))
        hits.sort(key=lambda h: h[1])

        pairs = [(a, b) for i, a in enumerate(hits) for b in hits[i + 1:]
                 if b[1] > a[2]]
        pairs.sort(key=lambda ab: ab[1][1] - ab[0][2])
        pairs = pairs[:args.max_pairs]
        if not pairs:
            print(f"SKIPPED: {path.stem} has no separated motif pair", file=sys.stderr)
            continue

        for (na, sa, ea), (nb, sb, eb) in pairs:
            # Identify the pair by POSITION, not by motif name: the same motif names can
            # occur at several positions in one locus, and keying on names alone lets one
            # pair overwrite another's deltas and merges them in the groupby downstream.
            pair_id = f"{na}@{sa}->{nb}@{sb}"
            spacer = locus_seq[ea:sb]
            variants, labels, loci_variants = [], [], []
            for shift in SHIFTS:
                if shift > 0:
                    filler = "".join(rng.choice(list(spacer or "ACGT"), size=shift))
                    mid = len(spacer) // 2
                    new_spacer = spacer[:mid] + filler + spacer[mid:]
                elif shift < 0:
                    d = -shift
                    if len(spacer) - d < 1:
                        continue
                    # Centred deletion of d bases. The previous form
                    # spacer[:cut+shift//2] + spacer[cut-shift//2+shift:] cancelled itself
                    # for even shifts and left the length change to a 3' truncation, so the
                    # negative arm was not the mirror of the positive one.
                    lc = len(spacer) // 2 - d // 2
                    new_spacer = spacer[:lc] + spacer[lc + d:]
                else:
                    new_spacer = spacer
                new_locus = locus_seq[:ea] + new_spacer + locus_seq[sb:]
                # Anchor the downstream flank in absolute coordinates: slice the window at
                # the ORIGINAL locus length so a length change is absorbed inside the locus
                # rather than silently duplicating or dropping flank.
                full = window[:offset] + new_locus + window[offset + len(locus_seq):]
                full = (full + "N" * SEQ_LEN)[:SEQ_LEN]
                variants.append(encode_window(full))
                nb_start = ea + len(new_spacer)
                labels.append((shift, nb_start, nb_start + (eb - sb)))
                loci_variants.append(new_locus)

            preds = []
            for i in range(0, len(variants), args.batch_size):
                out = model.model.predict(np.stack(variants[i:i + args.batch_size]), verbose=0)
                preds.append(np.asarray(out)[..., :4])
            preds = np.concatenate(preds, axis=0)

            pair_rows, base = [], None
            for k, (shift, nb_s, nb_e) in enumerate(labels):
                mutated = loci_variants[k]
                probs = preds[k][offset:offset + len(mutated)]
                # Score the bases of the MUTATED locus at the mutated coordinates. Reading
                # them from the native locus scores whatever happens to sit at the shifted
                # offset, which makes the whole response curve an artifact.
                lp = motif_logprob(probs, nb_s, nb_e, mutated)
                if shift == 0:
                    base = lp
                pair_rows.append(dict(locus=path.stem, pair_id=pair_id, motif_a=na,
                                      motif_b=nb, native_spacer=len(spacer), shift=shift,
                                      downstream_logprob=round(lp, 5)))
            if base is not None:
                for r in pair_rows:
                    r["delta_vs_native"] = round(r["downstream_logprob"] - base, 5)
            rows += pair_rows
            print(f"{path.stem}: {pair_id} spacer {len(spacer)} bp, "
                  f"{len(labels)} shifts", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("error: no motif pairs to perturb")
    df.to_csv(out_dir / "spacer_response.csv", index=False)

    # helical periodicity: power at ~10.5 bp in the shift response, per pair
    summary = []
    for (locus, pair_id), g in df.groupby(["locus", "pair_id"]):
        g = g.sort_values("shift")
        y = g["delta_vs_native"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 16:
            continue
        y = y - y.mean()
        freqs = np.fft.rfftfreq(len(y), d=1.0)
        power = np.abs(np.fft.rfft(y)) ** 2
        target = 1.0 / HELICAL_PERIOD
        k = int(np.argmin(np.abs(freqs - target)))
        summary.append(dict(locus=locus, pair_id=pair_id,
                            range_of_response=round(float(np.ptp(y)), 4),
                            helical_power_fraction=round(
                                float(power[k] / max(power[1:].sum(), 1e-12)), 4)))
    sdf = pd.DataFrame(summary)
    if not sdf.empty:
        sdf.to_csv(out_dir / "spacer_helical_summary.csv", index=False)
        print("\n" + sdf.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    for (locus, pair_id), g in df.groupby(["locus", "pair_id"]):
        g = g.sort_values("shift")
        ax.plot(g["shift"], g["delta_vs_native"], marker="o", ms=3, lw=1,
                label=f"{locus}: {pair_id}", alpha=0.85)
    ax.axvline(0, color="0.4", ls="--", lw=0.9)
    ax.set_xlabel("change in spacer length (bp)")
    ax.set_ylabel("Δ log2 P(downstream motif) vs native spacing")
    ax.set_title("Spacer-perturbation response — is the model sensitive to motif spacing?",
                 fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "spacer_response.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_dir/'spacer_response.csv'}")
    print(f"wrote {out_dir/'spacer_response.png'}")

if __name__ == "__main__":
    main()
