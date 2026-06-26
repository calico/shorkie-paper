# Species lists — LM corpus tiers

The Shorkie LM is a masked DNA language model pretrained on a multi-species
fungal genome corpus. The paper releases **four corpus tiers** spanning
increasing phylogenetic breadth. Each tier is defined by the exact set of
genome assemblies listed in the CSV files committed here, so the corpus build
(`scripts/01_data_build/lm_corpus/`) is reproducible from this repo.

| Tier (public name) | Species CSV (this dir) | Internal dir | # species | LM-corpus `train_seqs` |
|---|---|---|---|---|
| **R64** | `species_r64_gtf.cleaned.csv` | `data_r64_gtf` | 1 | 1,201 |
| **80_strains** | `species_strains_gtf.cleaned.csv` | `data_strains_gtf` | 80 | 102,315 |
| **165_Saccharomycetales** ⭐ | `species_saccharomycetales_gtf.cleaned.csv` | `data_saccharomycetales_gtf` | 165 | 385,551 |
| **1341_Fungus** | `species_fungi_1385_gtf.cleaned.csv` | `data_fungi_1385_gtf` | 1,361 | 625,355 |

⭐ The **released Shorkie LM** (`gs://seqnn-share/shorkie_lm/`) was pretrained on
the **165_Saccharomycetales** tier (model dir `lm_saccharomycetales_gtf`). The
other three tiers were used for the LM ablation variants in the paper.

All four tiers share the same held-out **test (528 seqs)** and **valid (518 seqs)**
splits (both drawn from *S. cerevisiae* R64 only — see the chromosome-holdout
split under `scripts/01_data_build/lm_corpus/`); only `train_seqs` grows
with the tier. `seq_length = 16384` throughout.

## A note on the "1341 / 1361 / 1385" numbers

The broadest tier carries three different counts in different places — all refer
to the same corpus:

- **1385** — the *initial* EnsemblFungi r59 download list (internal dir name
  `data_fungi_1385_gtf`, CSV stem `species_fungi_1385_gtf`).
- **1361** — the *cleaned* species count after filtering assemblies that lack a
  usable FASTA/GTF (this is the authoritative count: the CSV here has 1,361 data
  rows and the tier's `statistics.json` reports `num_species: 1361`).
- **1341** — the public label used in the top-level `README.md` and the GCS path
  `gs://shorkie-paper/data/unsupervised/{genome,processed}/1341_Fungus/`.

Use `1341_Fungus` when referring to the released GCS artifact; the actual number
of assemblies in the corpus is **1,361**.

## CSV schema

Comma-separated, one row per assembly (plus a header). Columns:

| Column | Meaning |
|---|---|
| `Unnamed: 0` | row index (dropped on load) |
| `Name` | species / strain name |
| `Classification` | order (e.g. `Saccharomycetales`) |
| `Taxon ID` | NCBI taxonomy ID |
| `Assembly` | assembly name (e.g. `R64-1-1`) |
| `Accession` | INSDC/GCA accession (e.g. `GCA_000146045.2`) — the genome key |
| `Species_str` | Ensembl species slug (lowercased, for FTP URLs) |
| `assembly` | assembly string used in the FASTA/GTF filename |
| `Core_db` | Ensembl core DB (also yields the collection sub-path) |
| `assembly_level` | `chromosome` or `scaffold` |
| `n_chroms` | number of top-level sequences |
| `total_length` | total assembly length (bp) |

The download stage (`1_data_download/{1_download_fasta,2_download_gtf}.py`) turns
`Accession` / `Species_str` / `assembly` / `Core_db` into EnsemblFungi r59 FTP
URLs. See the build scripts under `scripts/01_data_build/lm_corpus/`.
