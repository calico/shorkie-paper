# 10 · Corpus-size-matched fungal control

The evolutionary "sweet spot" at 165 Saccharomycetales genomes is confounded with corpus
size, annotation quality and optimisation difficulty. The peak could be driven partly by the amount
of data rather than by phylogenetic scope, and the paper acknowledges the confounds without
measuring them.

This is one of the paper's two least-supported claims, and the most consequential experiment in the
set.

## Why this design

The control is easy to describe in one sentence, but that leaves two decisions open which change
what it actually tests.

**Which 165 genomes.** Sampling from the whole fungal corpus would re-include the ~170
Saccharomycetales genomes it already contains, so the "broad" corpus would partly *be* the narrow
one. Step 2 samples from the **1,191 non-Saccharomycetales** genomes only, so the two corpora differ
in phylogenetic scope rather than in membership.

**How to spread them.** A uniform draw of 165 from 1,191 is dominated by the best-represented orders
and is not really "the broader fungal kingdom". The default is a stratified draw across taxonomic
orders (`--strategy random` gives the literal uniform draw for comparison). The result spans **66
orders** against the reference tier's 1.

Everything else is held fixed so that only scope varies: same `unet_small` architecture (the one
Figure 1F/G used — *not* the released `unet_small_bert_drop`, or the new point would not land on that
figure), same `num_features = 4 + 165 + 1 = 170`, same optimiser and schedule, and the same held-out
split — valid chrXI/XIII/XV, test chrXII/XIV/XVI, drawn from *S. cerevisiae* R64 only. The window
subsampling deliberately touches the training split alone, because if valid/test moved, perplexity
would no longer be comparable to the published Figure 1G.

## What step 1 already establishes, with no GPU at all

Before any model is trained, the named confounds can be measured from the committed
species lists and the release manifest:

| Tier | Genomes | Assembly | Mean genome | Chromosome-level | Orders | Train windows | Assembly sampled |
|---|---|---|---|---|---|---|---|
| R64 | 1 | 0.01 Gb | 12.1 Mb | 100% | 1 | 1,201 | 40.8% |
| 80_strains | 80 | 1.00 Gb | 12.5 Mb | 100% | 1 | 102,315 | 42.0% |
| **165_Saccharomycetales** | 165 | 2.08 Gb | 12.6 Mb | 36.4% | 1 | **385,551** | **75.9%** |
| 1341_Fungus | 1,361 | 41.62 Gb | 30.6 Mb | 9.5% | 67 | **625,355** | **6.2%** |

The headline is already an answer to part of the question:

> **The broad fungal corpus has 1.62× MORE training windows than the Saccharomycetales corpus, not
> fewer.** Data volume alone cannot explain why it underperforms.

But the other confounds are real and large. Fungal genomes are 2.4× bigger, far more fragmented
(9.5% chromosome-level vs 36.4%), and only **6.2%** of each raw assembly survives repeat/homology
filtering into training windows, against **75.9%** for Saccharomycetales — so each fungal genome
contributes far less usable sequence, and the corpus is spread thinly across 1,361 species instead of
concentrated in 165. That is the confound the control isolates.

*(Window counts multiply by the 4,096 bp stride, not the 16,384 bp window length — the tiling overlaps
4×, so "sampled bp" would otherwise be inflated fourfold.)*

### The third confound: optimisation

Three confounds are named above; corpus size and annotation quality are only the first two.
`1_training_dynamics.py` parses the four tiers' `train.out` logs — no GPU needed — and finds
something the paper does not report:

| Tier | epochs run | `train_epochs_max` | `patience` | best valid loss | best epoch | still descending? |
|---|---|---|---|---|---|---|
| R64 | 117 | **500** | **50** | 0.4181 | 65 | no |
| 80_strains | 170 | **500** | **50** | 0.4154 | 118 | no |
| 165_Saccharomycetales | 5,001 | **10,000** | **1,000** | **0.4018** | 4,727 | no |
| 1341_Fungus | 3,991 | **10,000** | **1,000** | 0.4055 | 2,989 | no |

**The four tiers were not trained on a matched schedule.** R64 and 80_strains were given a 500-epoch
ceiling and patience 50; Saccharomycetales and 1341_Fungus were given 10,000 and 1,000 — a 20×
difference. Each run converged *within its own budget* (none was still descending at the end), so no
tier is straightforwardly under-trained. But "converged within its own budget" is not "trained
comparably", and any comparison across the two schedule groups carries this confound.

This is the "optimisation difficulties" confound, it is real, and it should be disclosed rather than
argued away. It also changes the design of step 7 below, which matches the schedule as well as the
window count.

## Reading the result

| Outcome | Interpretation |
|---|---|
| control ≈ 1341_Fungus | The broad corpus underperforms because of its **scope**; window count was never the explanation, and the sweet-spot claim stands as written. |
| control ≈ 165_Saccharomycetales | Matching data volume closes the gap; the claim should be weakened from "optimal evolutionary scope" to a statement about effective corpus size. |
| control in between | Both contribute; report the decomposition rather than a single cause. |

All three are reportable. The experiment is not designed so that only one answer is publishable.

## Steps

| Step | Script | Cost | Output |
|---|---|---|---|
| 1 | `1_characterize_corpora.py` | CPU, seconds | `results/corpus_characteristics.csv`, `corpus_taxonomy.csv`, `verify_revision_10.csv` |
| 1 | `1_training_dynamics.py` | CPU, seconds | `results/training_dynamics.csv`, `training_dynamics.png` |
| 2 | `2_sample_matched_corpus.py` | CPU, seconds | `results/species_fungi165_matched_gtf.cleaned.csv` (committed-list schema), `matched_corpus_comparison.csv` |
| 3 | `3_build_corpus.sh` (+ `3_subsample_windows.py`) | CPU + network, hours | TFRecords for the new tier, training windows subsampled to exactly 385,551 |
| 4 | `4_train_matched_lm.sh` | **GPU, days** | `results/lm_fungi165_matched_unet_small/train/model_best.h5` |
| 5 | `5_eval_perplexity.sh` | **GPU, hours** | held-out perplexity on the R64 test split |
| 6 | `6_finetune_and_eval.sh` | **GPU, days** — optional | 8-fold supervised fine-tune from the control LM |
| 7 | `7_volume_matched_baseline.sh` | **GPU, hours–days** | Saccharomycetales downsampled to the 80_strains window count and schedule |

```bash
bash 1_characterize_corpora.sh
python 2_sample_matched_corpus.py --write_species_list
bash 3_build_corpus.sh --dry-run && bash 3_build_corpus.sh
scripts/common/submit.sh --profile gpu 4_train_matched_lm.sh
scripts/common/submit.sh --profile gpu 5_eval_perplexity.sh
```

Step 6 is optional and expensive. Perplexity answers the scope question directly; the fine-tune
answers whether the corpus choice changes *downstream* prediction rather than just the language-model
objective. Run it only if step 5 leaves the question open. Score it with experiment 01's machinery
(`1_collect_fold_metrics.py --supervised_root <results dir>`) so it arrives with cross-fold intervals.

## Step 7 · The other half of the sweet spot

The "sweet spot" is a **peak**, and the data-volume confound runs in opposite directions on its two
sides:

| Side | Comparison | Window counts | Does volume explain it? |
|---|---|---|---|
| upper | Saccharomycetales **>** 1341_Fungus | 385,551 vs 625,355 | **No** — the broad corpus has 1.62× *more* |
| lower | Saccharomycetales **>** 80_strains | 385,551 vs 102,315 | **Possibly entirely** — 3.77× *more*, untested |

The corpus-matched control (steps 2–5) tests only the upper side. The lower side is where the
claim is actually vulnerable — and it is *cheaper* to test, because it needs no new genomes and no
corpus build: reuse the existing Saccharomycetales tier, subsample its training windows to 102,315,
rebuild only the TFRecords, retrain.

Step 7 also **matches the training schedule to 80_strains**, because of the finding above: matching
only the window count would leave the 20× patience difference standing and the arm would answer
nothing. Set `MATCH_SCHEDULE=0` to isolate volume alone. The source tier is ~50 GB and almost all of
it is read-only FASTA/GTF, so those are symlinked and only the small per-split BEDs are duplicated.

Read-out: if the downsampled Saccharomycetales LM still beats 80_strains, phylogenetic scope is doing
the work on *both* sides of the peak and the sweet-spot claim is safe. If it ties, the lower half of
the peak is a data-volume/schedule artifact and the claim must be restated.

## Notes on the build

`3_build_corpus.sh` drives the four corpus stages directly through their uniform
`--save_suffix` / `--out_dir` interface rather than editing the hardcoded tier table in
`scripts/01_data_build/lm_corpus/run_pipeline.sh`, so no existing pipeline file is modified. The
corpus build downloads from EnsemblFungi release-59 and needs network access; stage 2 fetches
Ensembl's pre-soft-masked (`dna_sm`) assemblies rather than running RepeatModeler.

`params_unet_small.json` is committed here — it is the exact config the published
165_Saccharomycetales `unet_small` run used, so the control is self-contained and cannot drift from
the comparison it is meant to join.
