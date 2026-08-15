# lm_corpus — the masked-LM pretraining corpora

> **Advanced / reproduction-only.** The finished corpora are released; fetch them with
> `data/download.sh --lm-corpus <tier> -u PROJECT` rather than rebuilding.

Numbered stages, run in order (`run_pipeline.sh` chains them and `cd`s into each):

| Stage | Does |
|-------|------|
| `1_data_download/` | Fetch assemblies + GTFs for a tier's species list. |
| `2_repeat_region_masking/` | RepeatMasker / DUST soft-masking. |
| `3_data_filtering/` | Drop unusable contigs; build the sequence BEDs. |
| `4_tf_data_generation/` | Write 16,384 bp ZLIB TFRecords with the train/valid/test split. |

```bash
bash run_pipeline.sh --help
bash run_pipeline.sh --verify     # diff statistics.json against the expected counts
```

Which species go into each tier is fixed by the committed CSVs in
[`../../../data/species_lists/`](../../../data/species_lists), so the build is reproducible.

All four tiers share one held-out split, drawn from *S. cerevisiae* R64 only and split by whole
chromosome — **valid:** chrXI, chrXIII, chrXV; **test:** chrXII, chrXIV, chrXVI — with chrXI–XVI
excluded from training everywhere. Preserve that split if you want numbers comparable to the paper.

The phylogenetic-tree helper used for Figure 1B lives in
[`../../04_analysis/others/phylogenetic_tree/`](../../04_analysis/others/phylogenetic_tree).
