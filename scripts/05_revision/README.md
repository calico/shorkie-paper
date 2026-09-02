# 05_revision — additional analyses and controls

> **Advanced / analysis-only.** Nothing here is needed to *run* Shorkie — download released
> artifacts with `data/download.sh` and use `examples/` + `minimal_example/`. This stage holds
> follow-up analyses and controls for the claims in the Shorkie paper.

Eleven experiments, each answering one question the paper raises but does not fully settle: how
stable the headline numbers are across folds, whether the ISM motif-recovery claim survives
quantification, what the pretraining-corpus comparison is confounded with, and whether the model has
learned regulatory syntax or only motifs.

[`EXPERIMENTS.md`](EXPERIMENTS.md) is the design document: why each experiment is built the way it
is, what question it answers, and how to read each possible outcome. Each leaf `README.md` carries
the question it addresses, the per-step recipe, and — where the experiment has been run — what it
found.

## Index

| # | Directory | Question | Cost | Status |
|---|---|---|---|---|
| 01 | [`01_headline_uncertainty/`](01_headline_uncertainty) | How stable is the 0.67→0.78 gap across the eight test folds? | CPU | **run** |
| 02 | [`02_ism_motif_recovery/`](02_ism_motif_recovery) | Does the ISM motif-recovery claim survive quantification? | CPU + GPU | steps 1–4 **run** |
| 03 | [`03_rnaseq_atlas_accounting/`](03_rnaseq_atlas_accounting) | How do 8 TFs plus a gene panel make 3,053 tracks? | CPU | **run** |
| 04 | [`04_loss_weight_provenance/`](04_loss_weight_provenance) | Where do the 72% / 7.39% loss-weighting figures come from? | CPU | **run** |
| 05 | [`05_figure_and_caption_fixes/`](05_figure_and_caption_fixes) | Three presentation defects in Figures 4, 5 and 6 | CPU | **run** |
| 06 | [`06_mpra_random_init/`](06_mpra_random_init) | Is Shorkie better than Shorkie_Random_Init on MPRA? | GPU | scripted |
| 07 | [`07_gene_restricted_ism/`](07_gene_restricted_ism) | Should ISM sum only the target gene's output bins? | GPU | scripted |
| 08 | [`08_tsne_length_control/`](08_tsne_length_control) | Is the Figure 2E separation driven by element length? | CPU + GPU | step 1 **run** |
| 09 | [`09_dependency_maps/`](09_dependency_maps) | Words or grammar — has the LM learned motif *syntax*? | GPU | scripted |
| 10 | [`10_corpus_size_control/`](10_corpus_size_control) | Is the corpus "sweet spot" a data-volume artifact? | CPU + GPU | steps 1–2 **run** |
| 11 | [`11_layerwise_reinit/`](11_layerwise_reinit) | Which pretrained layer group carries the transfer? | GPU | scripted |

**run** = executed, with results and numbers recorded in that directory's README.
**scripted** = ready to submit; needs GPU nodes.

Two experiments carry an extra step added after an audit: **02 step 5** runs Shorkie_Random_Init ISM
over the RRB promoters, so RRPE can be settled at Figure 4B–C rather than left unresolvable; and
**10 step 7** downsamples Saccharomycetales to the 80_strains window count and schedule, closing the
side of the "sweet spot" that the corpus-matched control does not test. `10 step 1` also gained a
training-dynamics companion, which found that the four corpus tiers were **not trained on a matched
schedule**.

## Conventions

Same as the rest of `scripts/`: numbered steps inside each leaf, every working `*.py` paired with a
`*.sh` runner that accepts `--dry-run`, paths resolved through `shorkie.config` with no machine
literals, and submission through `scripts/common/submit.sh --profile {cpu,gpu}`. Numeric claims are
checked with `reproduction/common/compare.{Check, write_verdicts}` into a `verify_revision_NN.csv`.

```bash
bash scripts/05_revision/01_headline_uncertainty/1_collect_fold_metrics.sh
scripts/common/submit.sh --profile gpu scripts/05_revision/07_gene_restricted_ism/1_gene_restricted_ism.sh
```

Each leaf writes to its own `results/`. The derived tables, verification CSVs and figures are
committed; two large regenerable intermediates are not — `01_headline_uncertainty/results/fold_metrics.csv`
(26 MB) and `02_ism_motif_recovery/results/saliency_cache.npz` (3 MB), both rebuilt by step 1 of their
own experiment.

## Relationship to `reproduction/`

`reproduction/` reproduces what was **published** and must keep doing so; nothing in this stage
modifies it. Revised panels are written under `scripts/05_revision/**/results/` instead. Where an
experiment here needs a published recipe, it **imports** it from `reproduction/figure_NN/recheck/`
rather than reimplementing it — so a difference in a comparison can never be a difference in method.
