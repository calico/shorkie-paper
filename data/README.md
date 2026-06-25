# data/

Only **small, committed** reference files live here. Large artifacts (genome
FASTAs, TFRecords, ~93 GB of BigWigs, trained `model_best.h5` weights) are hosted
on GCS, catalogued in `manifest.json`, and fetched on demand by `download.sh`.

| Path | What |
|------|------|
| `R64_annotations/` | committed *S. cerevisiae* R64 refs: `chrom.sizes`, `assembly_gaps.bed`, `mask_rossi.bed` |
| `species_lists/` | per-tier LM-corpus species CSVs (R64 / 80_strains / 165_Saccharomycetales / 1341_Fungus) — committed so the corpus build is reproducible. See `species_lists/README.md`. |
| `manifest.json` | every released artifact: `gs://` URI, size, MD5, local layout, tier mapping |
| `download.sh` | manifest-driven fetch + checksum verify |

## Released artifacts

- **Model weights — public** (`gs://seqnn-share`): **live now** — LM `shorkie_lm/`, fine-tuned 8-fold
  `shorkie/f0..f7/`. **Catalogued, publication pending** — the random-init ablation 8-fold
  `shorkie_random_init/f0..f7/` (lr 5e-4). Downloadable with gsutil or plain https.
- **Datasets — requester-pays** (`gs://shorkie-paper`): **live now** — LM corpora
  `data/unsupervised/{genome,processed}/<tier>/`, supervised `data/supervised/{bigwigs,processed}/`.
  **Catalogued, publication pending** — the benchmark scores `eqtl/{scores,dream_eval}/` (Figure 7) +
  `mpra/{ground_truth,test_subset_ids,scores,dream}/` (Figure 6). Need gsutil + a billing project (`-u`).

The `eqtl`/`mpra` entries are the reproduction-minimal subsets (per-SNP scores + ground truth) that
let Figures 6–7 reproduce on CPU. The `shorkie_random_init` + `eqtl` + `mpra` artifacts are prepared
and catalogued in `manifest.json` (`pending_upload: true`) but **not yet on the buckets** — the
maintainer publishes them with `scripts/00_setup/upload_release.sh` (then `verify_release.py` confirms).
The large raw third-party inputs (1011-genomes GVCF, full DREAM sequences, DREAM-RNN/PrixFixe weights)
are documented in `manifest.json` (`external_raw`) but not re-hosted. `models.shorkie_scratch` (an older
lr-1e-4 random-init run) is a **deprecated alias** of `models.shorkie_random_init`.

## download.sh

```bash
# Minimal example: 8 fine-tuned folds -> ./my_shorkie (exactly what minimal_example expects)
data/download.sh --minimal

# All model weights (LM + fine-tuned) -> <dest>/models/...
data/download.sh --models all

# LM corpus tier(s) and supervised tracks (requester-pays -> pass -u PROJECT)
data/download.sh --lm-corpus 165_Saccharomycetales -u my-gcp-project
data/download.sh --supervised bigwigs            -u my-gcp-project
```

`--dest` sets the output base (default: `release_root` from `config/paths.yaml`).
Model files are MD5-verified against `manifest.json`; large dataset prefixes rely
on gsutil's built-in transfer integrity check. Add `--dry-run` to preview.

Point `data_root` / `release_root` in `config/paths.yaml` at where these land.
