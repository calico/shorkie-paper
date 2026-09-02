# 06 · Shorkie_Random_Init on the MPRA benchmark

The MPRA result is a zero-shot transfer claim: Shorkie was never trained on reporter constructs, yet
its marginalised predictions correlate with MPRA expression. What carries that transfer — the
language-model prior, or simply having been trained on yeast RNA-seq at all?

## Why this design

Shorkie_Random_Init is the exact control for that question: same architecture, same supervised data,
same 8-fold split, no pretraining. A related but much larger question — whether self-supervised
pretraining would improve a dedicated MPRA model such as DREAM-RNN — is a different modelling
programme and is not attempted here.

So the design principle is that **only the initialisation may differ**. The 10 (plus-strand) and 12
(minus-strand) reporter genes, the eleven insertion sites, the 180 bp analysis context, the sequence
categories, `--rc`, `--stats`, the targets sheet and the genome are all held fixed, and the
correlation recipes are *imported* from the published Figure-6 loaders rather than rewritten, so no
difference in filtering can leak into the comparison.

## A blocker this had to clear first

The published MPRA runs pass `--ctx <gene>.tsv` — the per-gene insertion contexts. Those files are
not where the run scripts say they are (`experiments/{SUM_data_process/,}MPRA/genes/` are both
empty), which means the MPRA pipeline could not be re-run at all, **for any model**, until this was
fixed.

They survive one directory deeper, next to the `generate_tsv.py` that made them, and that generator
turned out to depend on a hand-maintained list of 22 literal TSS coordinates. Step 0 reproduces the
recipe from the R64 GTF instead — eleven sites 100–200 bp from the TSS in 10 bp steps, each row the
110 bp insert footprint centred on the site — and **`--validate_against` confirms all 22 files come
out byte-identical to the originals**. The pipeline is now regenerable from the annotation rather
than from a lost intermediate.

## What it will answer

- Does the LM prior carry the zero-shot MPRA transfer, or would any yeast-trained supervised model
  do as well? A **small or absent** gap is a publishable and useful negative result, and is what the
  response should report if that is what the data shows.
- Does the answer differ between the single-sequence categories (6D/6E — absolute expression) and the
  ref/alt paired categories (6F/6G/6H — variant effects)? Variant effects are the harder task and the
  one closer to the paper's headline claims, so a gap that appears only there would be informative.
- Does the high-vs-low classification (6B/6C), where Shorkie is near-ceiling at AUROC 0.995, separate
  the models at all — or is it too easy to be diagnostic?

## Steps

| Step | Script | Cost | Output |
|---|---|---|---|
| 0 | `0_build_context_tsv.py` | CPU, seconds | `results/ctx/{pos,neg}/<SYMBOL>.tsv` — validated byte-identical to the originals |
| 1 | `1_run_mpra_random_init.sh` | **GPU**, 4 array jobs | `results/MPRA_random_init/<category>/<gene>_<strand>/` |
| 2 | `2_postprocess_to_npz.py` | CPU | `results/npz/…` in the Figure-6 layout |
| 3 | `3_extend_fig6_series.py` | CPU, minutes | `results/fig6_with_random_init.csv`, `results/fig6_three_way.png` |

Step 1 is four SLURM array jobs, sized exactly as the published ones (genes × categories):

```bash
python 0_build_context_tsv.py                                  # once
scripts/common/submit.sh --profile gpu --array 0-49 1_run_mpra_random_init.sh single pos
scripts/common/submit.sh --profile gpu --array 0-29 1_run_mpra_random_init.sh dual   pos
scripts/common/submit.sh --profile gpu --array 0-59 1_run_mpra_random_init.sh single neg
scripts/common/submit.sh --profile gpu --array 0-35 1_run_mpra_random_init.sh dual   neg
python 2_postprocess_to_npz.py && python 3_extend_fig6_series.py
```

Add `--dry-run` to step 1 to print the resolved `hound_MPRA_folds.py` command without submitting.

## Inputs

Released weights (`data/download.sh --models random_init`), the R64 genome (`--genome`), the MPRA
sequence subsets under `datasets.mpra`, and the T0 track index TSV that
`scripts/04_analysis/shorkie/mpra/3_process_hdf5_logsed/0_get_T0_rnaseq_index.py` produces.

## On the "correlative, non-causal" phrasing

That sentence can be read as a hint that this experiment was tried and buried. It was not.
The phrase (manuscript L337, L438) is about *Shorkie's own* endogenous predictions possibly
reflecting correlation rather than causation — a caveat on the model, not a report of a failed
comparison. The response should say so plainly, and then give the number this experiment produces,
whichever way it falls.
