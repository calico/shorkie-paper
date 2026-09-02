# 09 · Nucleotide dependency maps — words or grammar?

The paper claims Shorkie_LM "captures conserved regulatory grammar", but the evidence
behind it is recovery of known motifs — words, not grammar. Grammar means relationships *between*
motifs: spacing, multiplicity, arrangement. Either the claim is toned down, or something tests
those relationships directly.

Together with experiment 10, this addresses the second of the paper's two least-supported claims.

## Why this design

The distinction is precise: motif recovery demonstrates vocabulary, while the paper's sentence claims
syntax. There are only two honest outcomes — evidence of syntax, or a narrower sentence — and this
experiment is built so that either can be reported.

A useful accident of the repository made the design straightforward: the method is **already
implemented here, for the wrong model**.
`scripts/04_analysis/shorkie_lm/lm_SMT3_viz/dependency_map/compute_and_visualize_dep_maps_RPL43B.py`
runs the Tomaz da Silva recipe on **SpeciesLM** (`johahi/specieslm-fungi-upstream-k1`) — the
comparator model, not Shorkie_LM. So there is a reference implementation to port from and a
ready-made baseline to compare against, and step 1 is a faithful port rather than a fresh derivation.

**The computation is cheaper than it looks.** Because Shorkie_LM emits a nucleotide distribution at
every position, one forward pass per mutant yields a whole row of the dependency matrix — an L×L map
costs ~3L passes, not L×3L. A 500 bp locus is ~1,500 passes.

**Raw dependency is not evidence on its own.** Nearby positions covary simply because they share a
receptive field, so dependency falls off with distance regardless of any regulatory relationship.
Step 2 therefore tests dependency between two *motif* positions against a **distance-matched**
background of position pairs the same distance apart. Within-motif dependency is the sanity check
(any model that learned the motif should show it); **between-motif** dependency is the grammar test.

**Step 3 intervenes rather than correlates.** Dependency maps ask whether two positions covary;
spacing is the thing the definition actually names. So the spacer between two motifs is
lengthened or shortened by 1–20 bp — insertions drawn from a dinucleotide-matched shuffle of the
existing spacer, so composition is held fixed and only geometry changes — and the model's confidence
in the downstream motif is re-read. Two signatures are looked for:

- **monotonic decay** — confidence falls as the pair is pushed apart, i.e. the model has learned they
  belong together;
- **~10.5 bp periodicity** — helical phasing, the signature of two factors that must sit on the same
  face of the DNA helix. This is the strongest available evidence of learned syntax, because motif
  recognition alone cannot produce it.

A flat response means the model treats the two motifs independently — words, not grammar.

## Loci

Chosen so a genuine syntax signal has somewhere specific to be found, and so each ties back to a
figure:

| Locus | Resolved window | Syntax under test |
|---|---|---|
| GAL1-10 UAS | chrII:278,352–279,021 | Gal4 sites with constrained spacing in a divergent promoter |
| PHO5 promoter | chrII:430,851–431,551 | the Pho4 UASp1/UASp2 cooperative pair |
| CLN2 promoter | chrXVI:66,514–67,214 | tandem SBF (SCB) sites |
| RPL28 promoter | chrVII:310,367–311,067 | Rap1 with RRPE/PAC — ties to Figure 4 |
| HSP12 promoter | chrVI:106,656–107,356 | multiple STREs, the Msn2/Msn4 element — ties to Figure 5 |
| DTD1 intron | chrIV:65,235–65,431 | donor–branch–acceptor, the one syntax the paper already shows (Figure 4E) |

**Coordinates are resolved from the GTF at run time, never hardcoded.** Each promoter is defined by
systematic ORF name plus strand (600 bp upstream to 100 bp downstream of the TSS) and the GAL1-10 UAS
as the intergenic span between YBR019C and YBR020W. This is not a stylistic preference: an earlier
draft of this script hardcoded the windows and got RPL28 on the wrong chromosome (chrXII rather than
chrVII), HSP12 21 kb from the gene, and CLN2's window inside the gene body rather than its promoter.
Run `--list_loci` to print the resolved windows and check them before committing GPU time. DTD1 is
the one explicit window, because it is an intron rather than a promoter and it is the exact window
the paper's Figure 4E shows (DTD1 = YDL219W).

The DTD1 intron is a deliberate positive control: the paper already demonstrates the model
reconstructs all three splice signals, and their relative arrangement is genuine, well-defined syntax.
If the dependency analysis cannot detect structure *there*, that is diagnostic of the method rather
than of the model.

## How to report the outcome

| Result | Response |
|---|---|
| Between-motif dependency above the distance-matched background, and/or helical phasing | Cite it as direct evidence of syntax; the L148 and L216–218 claims stand with a supporting figure. |
| Within-motif dependency only, flat spacer response | Tone the claims down from "regulatory grammar" to "conserved motifs", and report this as the negative result that justified the change. |

## Steps

| Step | Script | Cost | Output |
|---|---|---|---|
| 1 | `1_dependency_map_shorkie_lm.py` | **GPU**, ~1,500 passes per 500 bp locus | `results/dep_maps/<locus>.npz` |
| 2 | `2_quantify_dependency.py` | CPU, minutes | `results/dependency_stats.csv`, `dependency_maps.png` |
| 3 | `3_spacer_perturbation.py` | **GPU**, ~40 passes per motif pair | `results/spacer_response.csv`, `spacer_helical_summary.csv`, `spacer_response.png` |

```bash
scripts/common/submit.sh --profile gpu 1_dependency_map_shorkie_lm.sh
bash 2_quantify_dependency.sh
scripts/common/submit.sh --profile gpu 3_spacer_perturbation.sh
```

For the SpeciesLM side-by-side, run the existing
`scripts/04_analysis/shorkie_lm/lm_SMT3_viz/dependency_map/compute_and_visualize_dep_maps_RPL43B.py`
on the same coordinates — it needs `torch` + `transformers` (the `pytorch_cuda` environment), not the
`yeast_ml` one.

## A caveat worth stating

Shorkie_LM is trained with 15% masking, and this recipe — like the SpeciesLM reference it ports —
reads the distribution from an **unmasked** input. That is the standard formulation of the method,
but it evaluates the model slightly off its training distribution, and at an unmasked position the
model can read the reference base directly, so `p_ref` is near one-hot and the `log2(1 − p)` term is
at its least numerically stable. Masking the query position instead would cost a further factor of L
— one full variant sweep per query position — and is not implemented here.

Motif occurrences are located with experiment 02's native PWM scanner, so both experiments agree on
what counts as a motif.
