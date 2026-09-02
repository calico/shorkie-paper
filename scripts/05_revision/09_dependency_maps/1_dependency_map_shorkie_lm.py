#!/usr/bin/env python3
"""Revision experiment 09, step 1 — nucleotide dependency maps from Shorkie_LM
.

The problem with the manuscript's "captures conserved regulatory grammar" (L148): the evidence
behind it is recovery of known yeast motifs, which are words rather than grammar. Grammar means
relationships BETWEEN motifs -- spacing, multiplicity, arrangement. Either the claim is toned
down, or something tests those relationships directly. Nucleotide dependency maps over loci with
well-characterised regulatory syntax are one way to do it.

The repository already contains this method -- but applied to the WRONG model.
``scripts/04_analysis/shorkie_lm/lm_SMT3_viz/dependency_map/compute_and_visualize_dep_maps_RPL43B.py``
implements the Tomaz da Silva et al. (2025) recipe for **SpeciesLM**
(``johahi/specieslm-fungi-upstream-k1``), which is the comparator, not Shorkie_LM. This
script ports the same computation to Shorkie_LM, so the two can be run on the same loci.

Method, following that reference implementation:

  1. generate every single-nucleotide substitution of an L bp locus (3L mutants);
  2. embed each in the 16,384 x 170 Shorkie_LM input and take ONE forward pass -- because
     the model emits a nucleotide distribution at every position, a single pass yields the
     whole row of the dependency matrix, so an L x L map costs ~3L passes, not L x 3L;
  3. for query position i and variant (j, n), the dependency is the change in the model's
     log-odds at i:
         logit(p_alt[i]) - logit(p_ref[i])
  4. dep_map[i, j] = max over the 4 x 4 (variant base, query base) entries of |effect|,
     with the diagonal zeroed.

A caveat worth stating, because it is a real limitation rather than a footnote: Shorkie_LM is trained with 15% masking, and this recipe -- like the SpeciesLM
reference it ports -- reads the distribution from an UNMASKED input. That is the standard
formulation of the method, but it evaluates the model slightly off its training
distribution, and at an unmasked position the model can read the reference base directly,
so ``p_ref`` is near one-hot and the ``log2(1 - p)`` term is at its least numerically
stable. Masking the query position instead would cost a further factor of L (one full
variant sweep per query position) and is not implemented here.

Writes ``results/dep_maps/<locus>.npz`` per locus.
GPU required in practice: ~1,500 forward passes per 500 bp locus.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pysam

from shorkie import config
from shorkie.models.ensemble import NUM_FEATURES, SCEREVISIAE_COL

SEQ_LEN = 16384
NT = "ACGT"
NT_IX = {c: i for i, c in enumerate(NT)}
EPS = 1e-10

# Loci with well-characterised regulatory syntax -- chosen so a genuine grammar signal has
# something specific to be found in, and so each ties back to a figure.
#
# Coordinates are NOT hardcoded. Hardcoding them is how the first draft of this script ended
# up scanning the wrong chromosome for RPL28 and a window 21 kb from HSP12, so every locus is
# now resolved from the GTF by systematic ORF name and strand at run time. Use --list_loci to
# print the resolved windows and check them before spending GPU time.
#
#   kind="promoter"   window is `up` bp upstream to `down` bp downstream of the gene's TSS,
#                     on the correct strand
#   kind="intergenic" window is the span between two genes (a divergent UAS)
#   kind="interval"   an explicit window, for a feature that is not a promoter
DEFAULT_LOCI = [
    dict(name="GAL1_10_UAS", kind="intergenic", between=("YBR019C", "YBR020W"),
         note="divergent GAL10-GAL1 UAS: Gal4 sites with constrained spacing"),
    dict(name="PHO5_promoter", kind="promoter", gene="YBR093C", up=600, down=100,
         note="Pho4 UASp1 / UASp2 pair, a classic cooperative arrangement"),
    dict(name="CLN2_promoter", kind="promoter", gene="YPL256C", up=600, down=100,
         note="tandem SBF (SCB) sites"),
    dict(name="RPL28_promoter", kind="promoter", gene="YGL103W", up=600, down=100,
         note="ribosomal protein promoter: Rap1 with RRPE/PAC (ties to Figure 4)"),
    dict(name="HSP12_promoter", kind="promoter", gene="YFL014W", up=600, down=100,
         note="multiple STREs, the Msn2/Msn4 element (ties to Figure 5)"),
    # DTD1 = YDL219W. This one IS an explicit window because it is an intron rather than a
    # promoter, and it is the exact window the paper's Figure 4E shows.
    dict(name="DTD1_intron", kind="interval", chrom="chrIV", start=65_235, end=65_431,
         note="donor-branch-acceptor, the one syntax the paper already shows (Figure 4E)"),
]

def gene_records(gtf_path):
    """{systematic ORF: (chrom, start1, end, strand)} for every gene in the GTF."""
    out = {}
    for line in open(gtf_path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        m = re.search(r'gene_id "([^"]+)"', f[8])
        if m:
            # GTF is 1-based inclusive; store 0-based start so downstream fasta.fetch and
            # the 08_tsne_length_control loader agree on coordinates.
            out[m.group(1)] = (f[0], int(f[3]) - 1, int(f[4]), f[6])
    return out

def resolve_loci(loci, genes, chrom_prefix="chr"):
    """Turn the locus specs above into concrete (chrom, start, end) windows."""
    resolved = []
    for L in loci:
        spec = dict(L)
        kind = spec.get("kind", "interval")
        try:
            if kind == "promoter":
                chrom, s, e, strand = genes[spec["gene"]]
                tss = s if strand == "+" else e
                lo, hi = ((tss - spec["up"], tss + spec["down"]) if strand == "+"
                          else (tss - spec["down"], tss + spec["up"]))
            elif kind == "intergenic":
                a, b = (genes[g] for g in spec["between"])
                chrom = a[0]
                lo, hi = min(a[2], b[2]), max(a[1], b[1])   # between the two gene bodies
                if hi <= lo:
                    raise ValueError(f"{spec['between']} do not flank an intergenic gap")
            else:
                chrom = spec["chrom"]
                lo, hi = spec["start"], spec["end"]
        except KeyError as e:
            print(f"SKIPPED: {spec['name']} — {e} not in the GTF", file=sys.stderr)
            continue
        except ValueError as e:
            print(f"SKIPPED: {spec['name']} — {e}", file=sys.stderr)
            continue
        if not str(chrom).startswith(chrom_prefix):
            chrom = f"{chrom_prefix}{chrom}"
        spec.update(chrom=chrom, start=int(lo), end=int(hi))
        resolved.append(spec)
    return resolved

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nucleotide dependency maps from Shorkie_LM.")
    parser.add_argument("--loci", default=None,
                        help="Comma-separated name:chrom:start:end entries; default is the "
                             "characterised-syntax panel resolved from the GTF")
    parser.add_argument("--list_loci", action="store_true",
                        help="Print the GTF-resolved windows and exit (check these before "
                             "committing GPU time)")
    parser.add_argument("--gtf", default=None, help="[default: config genome.gtf]")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fasta", default=None, help="[default: config genome.fasta]")
    parser.add_argument("--lm_checkpoint", default=None,
                        help="[default: config models.shorkie_lm_checkpoint]")
    parser.add_argument("--lm_params", default=None,
                        help="[default: <models.shorkie_lm>/params.json]")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def parse_loci(spec):
    out = []
    for entry in spec.split(","):
        parts = entry.split(":")
        if len(parts) != 4:
            sys.exit(f"error: bad --loci entry '{entry}' (want name:chrom:start:end)")
        out.append(dict(name=parts[0], chrom=parts[1], start=int(parts[2]),
                        end=int(parts[3]), note="user-supplied"))
    return out

def one_hot_window(seq, locus_offset, locus_seq):
    """16,384 x 170 input with `locus_seq` written at `locus_offset` of the window."""
    x = np.zeros((SEQ_LEN, NUM_FEATURES), dtype="float32")
    full = seq[:locus_offset] + locus_seq + seq[locus_offset + len(locus_seq):]
    for i, ch in enumerate(full[:SEQ_LEN]):
        j = NT_IX.get(ch)
        if j is not None:
            x[i, j] = 1.0
    x[:, SCEREVISIAE_COL] = 1.0
    return x

def main():
    args = parse_args()
    config.load()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    (out_dir / "dep_maps").mkdir(parents=True, exist_ok=True)

    gtf = Path(args.gtf) if args.gtf else config.path("genome.gtf")
    if gtf is None or not Path(gtf).exists():
        sys.exit(f"error: --gtf not resolved ({gtf}). Fetch the genome with "
                 "`bash data/download.sh --genome -u PROJECT` or pass --gtf.")
    genes = gene_records(gtf)
    loci = (parse_loci(args.loci) if args.loci
            else resolve_loci(DEFAULT_LOCI, genes))
    print(f"gtf   : {gtf}  ({len(genes)} genes)", flush=True)
    print("resolved loci:", flush=True)
    for L in loci:
        src = (L.get("gene") or "+".join(L.get("between", ())) or "explicit")
        print(f"  {L['name']:16s} {L['chrom']}:{L['start']:,}-{L['end']:,} "
              f"({L['end']-L['start']} bp, from {src})", flush=True)
    if args.list_loci:
        return

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
    print(f"checkpoint : {ckpt}", flush=True)

    fasta = pysam.Fastafile(str(fasta_path))
    refs = set(fasta.references)

    for L in loci:
        try:
            chrom = L["chrom"] if L["chrom"] in refs else (
                L["chrom"][3:] if L["chrom"][3:] in refs else L["chrom"])
            locus_len = L["end"] - L["start"]
            centre = (L["start"] + L["end"]) // 2
            win_start = max(0, centre - SEQ_LEN // 2)
            window = fasta.fetch(chrom, win_start, win_start + SEQ_LEN).upper()
            if len(window) < SEQ_LEN:
                window = window + "N" * (SEQ_LEN - len(window))
            offset = L["start"] - win_start
            ref_locus = window[offset:offset + locus_len]
            if len(ref_locus) != locus_len:
                print(f"SKIPPED: {L['name']} window truncated", file=sys.stderr)
                continue

            # Enumerate the variants, but build the one-hot windows ONE BATCH AT A TIME.
            # Each window is 16384 x 170 float32 = 10.6 MiB, so materialising all 3L+1 of
            # them would need ~21 GiB for a 700 bp locus and OOM before the first predict.
            variants = [(-1, -1)]
            for pos in range(locus_len):
                for n, base in enumerate(NT):
                    if base != ref_locus[pos]:
                        variants.append((pos, n))
            print(f"{L['name']}: {locus_len} bp, {len(variants)} forward passes", flush=True)

            def mutant_locus(v):
                pos, n = v
                if pos < 0:
                    return ref_locus
                return ref_locus[:pos] + NT[n] + ref_locus[pos + 1:]

            probs = []
            for i in range(0, len(variants), args.batch_size):
                chunk = variants[i:i + args.batch_size]
                batch = np.stack([one_hot_window(window, offset, mutant_locus(v))
                                  for v in chunk])
                out = model.model.predict(batch, verbose=0)
                probs.append(np.asarray(out)[:, offset:offset + locus_len, :4])
                del batch
                if (i // args.batch_size) % 20 == 0:
                    print(f"  {min(i + args.batch_size, len(variants))}/{len(variants)}",
                          flush=True)
            probs = np.concatenate(probs, axis=0).astype(np.float64) + EPS
            probs /= probs.sum(axis=-1, keepdims=True)

            ref_probs = probs[0]
            # log-odds difference, exactly as the SpeciesLM reference implementation
            effect = np.zeros((locus_len, locus_len, 4, 4), dtype=np.float32)
            for k, (p, n) in enumerate(variants[1:], start=1):
                delta = (np.log2(probs[k]) - np.log2(1 - probs[k])
                         - np.log2(ref_probs) + np.log2(1 - ref_probs))
                effect[p, :, n, :] = delta
            dep_map = np.max(np.abs(effect), axis=(2, 3))
            np.fill_diagonal(dep_map, 0.0)

            np.savez_compressed(
                out_dir / "dep_maps" / f"{L['name']}.npz",
                dep_map=dep_map.astype(np.float32), ref_probs=ref_probs.astype(np.float32),
                sequence=ref_locus, chrom=L["chrom"], start=L["start"], end=L["end"],
                note=L["note"], model="Shorkie_LM")
            print(f"  saved {L['name']}.npz  dep_map {dep_map.shape} "
                  f"max {dep_map.max():.2f}", flush=True)
        except Exception as e:                                   # fail-soft per locus
            print(f"SKIPPED: {L['name']} ({e})", file=sys.stderr)

    print(f"\nwrote {out_dir/'dep_maps'}")

if __name__ == "__main__":
    main()
