# 01_data_build — building the training data

> **Advanced / reproduction-only.** You do not need this to *use* Shorkie: the corpora and supervised
> tracks are already released. Fetch them with `data/download.sh` — see [`../../data/README.md`](../../data/README.md).

| Subdirectory | Builds |
|--------------|--------|
| `lm_corpus/` | The four masked-LM pretraining corpora (R64, 80_strains, 165_Saccharomycetales, 1341_Fungus): download assemblies → repeat-mask → filter → 16,384 bp ZLIB TFRecords. |
| `supervised_tracks/` | The 5,215-track supervised dataset: FASTQ → BAM → BigWig → peaks → TFRecords. |

Both stages resolve every path through `config/paths.yaml`. The species lists that pin each corpus tier
(with NCBI accessions) are committed in [`../../data/species_lists/`](../../data/species_lists).
