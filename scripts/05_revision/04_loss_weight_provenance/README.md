# 04 · Provenance of the region-dependent loss weighting

The Results state that the loss was down-weighted by 0.1 at coding (72% of R64) and
repetitive (7.39%) positions. The Methods describe the down-weighting but not those percentages or
where the annotations come from, and the two passages use different words for the same thing.

## Why this design

The premise that this is absent from the Methods is *partly* mistaken and partly right, and it is
worth being precise about which is which.

The down-weighting **is** in the Methods — manuscript L582–584: *"We computed categorical
cross-entropy over masked positions only and reweighted the loss by genomic region. In order to
focus the model on regulatory sequences, we down-weighted exonic and repeat regions by a factor of
0.1"* — and it appears in Equation 1 as the position-specific weight `w`. What is missing, and what
made it unfindable, is everything that connects that sentence to the Results sentence: the word
"coding" versus "exonic", the two percentages, where the annotations come from, and the fact that
the supervised runs weight regions too.

So this experiment is a documentation job with two verifiable halves: recompute the percentages from
the actual annotation files, and read the weights out of the committed configs rather than restating
them from memory.

Because "coding" is ambiguous, step 1 computes **every** plausible definition side by side instead of
picking one — that is the honest way to report a number the paper states without defining.

## What it found

**The repetitive figure is exact.** 892,009 soft-masked bp of 12,071,326 = **7.389%**, matching the
published 7.39% to three decimals. The definition is the lowercase fraction of the RepeatMasker +
DUST soft-masked assembly the corpus build produces.

**The coding figure is approximate and needs its definition stated.** No standard definition lands
on 72%:

| Definition | % of R64 |
|---|---|
| exon, all gene biotypes | 73.83 |
| exon, all biotypes, 2 bp chew (what the training mask applies) | 73.60 |
| exon, protein-coding genes only | **71.22** ← closest to 72 |
| CDS | 71.07 |
| CDS, 2 bp chew | 70.86 |

72% is a fair round number but sits between the exon (73.8%) and CDS (71.1%) definitions. The
revision should state the definition and quote the exact value.

**A second, genuine Methods gap.** The supervised runs weight regions too, and this is documented
nowhere:

| Run | exon_loss_scale | non_exon | relative weight on exonic/repetitive positions |
|---|---|---|---|
| `shorkie_lm` (pretraining) | 0.1 | 1.0 | **0.10** |
| `shorkie_finetuned` | 1.0 | 4.0 | **0.25** |
| `shorkie_scratch` | 1.0 | 4.0 | **0.25** |

Note this is *not* an inversion — both stages down-weight exonic and repetitive positions relative to
the rest, concentrating the objective on regulatory sequence. The supervised stage is simply milder
(0.25 vs 0.10) and writes it as an up-weighting of the complement. It applies identically to the
fine-tuned and random-init arms, so it does not confound the headline comparison, but it belongs in
the Methods.

Both weights are applied at `external/baskerville-yeast/src/baskerville/trainer.py:845` and `:849`.

## Steps

| Step | Script | Output |
|---|---|---|
| 1 | `1_recompute_annotation_fractions.py` | `results/annotation_fractions.csv`, `results/verify_revision_04.csv` |
| 2 | `2_document_loss_weights.py` | `results/loss_weights.csv`, `results/loss_weights_methods.md` (Methods-ready paragraph) |

## Inputs and cost

CPU, well under a minute. Step 2 needs only committed files. Step 1 needs the R64 corpus tier
(`<datasets.lm_corpus_split_root>/data_r64_gtf`) for the soft-masked FASTA and the Ensembl GTF —
override with `--r64_tier_dir`. Both denominators are reported: the whole assembly, and the union of
the 16,384 bp windows the language model actually trains on (11,702,272 bp), since the loss weighting
only ever sees the latter.
