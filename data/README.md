# data/

Only **small, committed** reference files live here. Large artifacts (genome
FASTAs, TFRecords, ~93 GB of BigWigs, trained `model_best.h5` weights) are
hosted on GCS and fetched on demand.

| Path | What |
|------|------|
| `R64_annotations/` | committed S. cerevisiae R64 refs: `chrom.sizes`, `assembly_gaps.bed`, `mask_rossi.bed` |
| `species_lists/` | per-tier species CSVs for the LM corpus (R64 / 80_strains / 165_Saccharomycetales / 1341_Fungus) — committed so the corpus build is reproducible |
| `manifest.json` | every large artifact: `gs://` URI, size, MD5, and tier mapping *(added in Phase 4)* |
| `download.sh` | fetch + checksum-verify artifacts into `data_root` *(added in Phase 4)* |

Released artifacts (see top-level `README.md`):
- LM weights: `gs://seqnn-share/shorkie_lm/`
- Shorkie weights (8 folds): `gs://seqnn-share/shorkie/f0..f7/`
- Datasets: `gs://shorkie-paper/data/{unsupervised,supervised}/`

Point `data_root` in `config/paths.yaml` at where `download.sh` places these.
