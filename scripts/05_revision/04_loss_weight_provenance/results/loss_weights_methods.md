# Region-dependent loss weighting — provenance

Extracted from the committed training configs by `scripts/05_revision/04_loss_weight_provenance/2_document_loss_weights.py`.

| Run | task / loss | exon_loss_scale | non_exon | repeat_loss_scale | non_repeat | exon rel. weight | repeat rel. weight |
|---|---|---|---|---|---|---|---|
| `shorkie_lm` | self-supervised / mlm | 0.1 | 1.0 | 0.1 | 1.0 | 0.1 | 0.1 |
| `shorkie_finetuned` | fine-tune / poisson_mn | 1.0 | 4.0 | 1.0 | 4.0 | 0.25 | 0.25 |
| `shorkie_scratch` | supervised / poisson_mn | 1.0 | 4.0 | 1.0 | 4.0 | 0.25 | 0.25 |

## Where the weights are applied

- `external/baskerville-yeast/src/baskerville/trainer.py:845` — `sw = exon_mask * self.exon_loss_scale + (1 - exon_mask) * self.non_exon_loss_scale`
- `external/baskerville-yeast/src/baskerville/trainer.py:849` — `repeat_sw = repeat_mask * self.repeat_loss_scale + (1 - repeat_mask) * self.non_repeat_loss_scale`

## Methods text this supports

**Pretraining.** Masked-token cross-entropy is weighted per position by genomic region. Exonic and repetitive positions receive a weight of 0.1 against 1.0 elsewhere, concentrating the objective on non-coding, non-repetitive sequence. Repetitive positions are those soft-masked by RepeatMasker + DUST in the corpus build; exonic positions come from the Ensembl Fungi release-59 annotation, with 2 bp chewed from each internal exon edge (`shorkie.data.bed_helper.get_exon_mask`). See step 1 for the exact genome fractions these definitions imply.

**Supervised training.** The same mechanism is used, expressed the other way round: exonic and repetitive positions receive 1.0 against 4.0 elsewhere, i.e. a relative weight of 0.25 on those regions in the Poisson-multinomial objective. The direction matches pretraining — the objective is concentrated on non-coding, non-repetitive sequence — but the magnitude is milder (0.25 rather than 0.10). This applies identically to the fine-tuned and random-initialised runs, so it does not confound the comparison between them, but it is currently absent from the Methods and should be stated.
