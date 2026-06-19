# LM genome-corpus build

Builds the multi-species fungal corpus the Shorkie **LM** is pretrained on:
EnsemblFungi release-59 genomes → soft-masked → filtered, chromosome-holdout
split → one-hot ZLIB TFRecords. Four tiers (R64 / 80_strains /
165_Saccharomycetales / 1341_Fungus) are defined by the committed species lists
in [`data/species_lists/`](../../../data/species_lists/) — see that README for
the tier table and the 1341/1361/1385 naming note.

> **Practical entry point:** download the prebuilt TFRecords instead of building
> from scratch — `data/download.sh --lm-corpus <tier>`. Stage 2 (de-novo
> RepeatModeler/RepeatMasker) is heavy; the orchestrator defaults to downloading
> Ensembl's pre-masked assemblies.

## Orchestrator

```bash
# Build the R64 tier (smallest; 1 species) — dry run prints every command:
scripts/01_data_build/lm_corpus/run_pipeline.sh --tier r64 --dry-run

# Check a built tier against the expected released statistics:
scripts/01_data_build/lm_corpus/run_pipeline.sh --tier r64 --verify
```

`run_pipeline.sh` resolves the build root from `datasets.lm_corpus_split_root`
in `config/paths.yaml`, seeds the committed species CSV into it, then runs each
stage with `--save_suffix <tier> --out_dir <root>`.

## Stages

| Stage | Scripts | Tools | What |
|---|---|---|---|
| **1 · download** | `1_data_download/{1_download_fasta,2_download_gtf,3_clean_fasta,4_split_gtf}.py` | `curl`, `pysam` | Fetch FASTA + GTF from EnsemblFungi r59 FTP; clean FASTA (keep `chromosome`-level, ≥32,768 bp, normalize chrom names); split GTF by biotype (protein_coding / rRNA / tRNA). |
| **2 · masking** | `2_repeat_region_masking/{0_rerun_repeatModeler.sh,1_repeatMasker.sh,2_rerun_dust.sh,3_download_masked_fasta.py}` | RepeatModeler, RepeatMasker (`-xsmall -e rmblast`), DUST | Soft-mask repeats. De-novo path is heavy; `3_download_masked_fasta.py` pulls Ensembl's pre-soft-masked (`dna_sm`) assemblies instead. |
| **3 · filtering** | `3_data_filtering/1_generate_sequences_bed.py` (+ `shorkie.data.bed_helper.generate_beds`) | `pysam`, `pybedtools` | Tile the genome into 16,384 bp windows; apply the chromosome-holdout split + biotype/repeat thresholds; write `sequences_{train,valid,test}.bed` and `statistics.json`. |
| **4 · tfrecord** | `4_tf_data_generation/{1_write_data_multi.py,write_data.py,write_data_with_gtf.py}` | TensorFlow, `baskerville.dna_io` | One-hot encode and write `tfrecords/{label}-*.tfr` (ZLIB). `_multi` reads `statistics.json` to shard into parts. |

`phylogentic_tree/` is an auxiliary step (NCBI taxonomy → species tree) used for
figures, not required for the corpus build.

## Chromosome-holdout split

Source: `3_data_filtering/1_generate_sequences_bed.py:59-73`. The same held-out
chromosomes are used for **every** tier, and only *S. cerevisiae* R64
(`GCA_000146045_2`) contributes valid/test windows — which is why all tiers share
`test_seqs = 528` and `valid_seqs = 518`:

- **train**: all species, **excluding** R64 `chrXI … chrXVI` from every species.
- **valid**: R64 `chrXI`, `chrXIII`, `chrXV`.
- **test**: R64 `chrXII`, `chrXIV`, `chrXVI`.

> ⚠️ Naming caveat: the corpus directory is
> `test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI`, which labels the
> chromosomes the **opposite** way round from the code above. The **code is
> authoritative** (valid = chrXI/XIII/XV, test = chrXII/XIV/XVI); the directory
> name is just a fixed string.

## Filtering thresholds

Source: `1_generate_sequences_bed.py:37-42`.

| Parameter | Value |
|---|---|
| `seq_length` | 16,384 bp |
| `seq_stride` | 4,096 bp |
| `record_size` | 32 (windows padded to a multiple) |
| `max_rm_frac` | 0.001 (≤ ~16 masked bp per train window; valid/test exempt) |
| `contig_padding` | 512 bp from contig edges |
| exon-overlap caps | `''` 0.925 · `.protein_coding` 0.875 · `.rRNA` 0.03125 · `.tRNA` 0.03125 |
| shuffle seed | 42 |

## Expected per-tier statistics

`run_pipeline.sh --tier <t> --verify` checks the built `statistics.json` against
these (all: `seq_length 16384`, `test_seqs 528`, `valid_seqs 518`):

| Tier | `num_species` | `train_seqs` |
|---|---|---|
| r64 | 1 | 1,201 |
| strains | 80 | 102,315 |
| saccharomycetales | 165 | 385,551 |
| fungi | 1,361 | 625,355 |

## Known quirks

- `1_write_data_multi.py:68` ships with the label loop set to `['test']` only
  (the `['train','valid','test']` line is commented). Edit it to build all
  splits.
- The de-novo masking scripts (`0_…`, `1_…`, `2_…`) reference site-specific
  RepeatMasker library paths; prefer `3_download_masked_fasta.py`.
