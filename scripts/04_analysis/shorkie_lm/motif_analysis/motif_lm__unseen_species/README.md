# motif_lm__unseen_species — cross-species generalization of LM motifs

Motif discovery from **Shorkie_LM** on species that were **held out of LM
training**, to test whether the learned regulatory grammar generalizes across
evolutionary distance. Same machinery as the in-distribution `motif_lm/`
pipeline (per-base LM contribution scores → TF-MoDISco → annotation/visualization),
run separately for each taxonomic tier.

## Taxonomic tiers (unseen species)

Each tier is a distinct held-out clade, ordered by increasing distance from the
*Saccharomycetales* the model was trained on:

| Tier (`dataset` arg) | Group |
|---|---|
| `saccharomycetales_select` | held-out *Saccharomycetales* species |
| `strains_select` | held-out *S. cerevisiae* strains |
| `schizosaccharomycetales` | fission yeasts |
| `orbiliales` | early-diverging Ascomycota |
| `ascomycota` | broad Ascomycota |

Each tier is scored with the 3 LM replicates
(`unet_small_bert_drop`, `…_retry_1`, `…_retry_2`); the `avg_*` scripts pool the
replicates before MoDISco (the cross-replicate / "cross-fold" averaging step).

## Pipeline (numbered steps)

1. `1_search_motif.py` — extract per-base LM contribution scores for a
   `(model, dataset)` pair (the `.sh` fans out over models × tiers via a SLURM array).
2. `2_modisco_script.sh` — TF-MoDISco motif discovery on the scored sequences.
3. `3_modisco_report.sh` / `3_modisco_report_high_conf.sh` — annotate discovered
   motifs against the external motif DB (`config: motif_db_dir`).
4. `4_viz_motif.py` — DNA-logo visualizations per species/model.
5. `5_map_motif_2_genome.py` — map discovered motifs to genome coordinates and
   write per-motif `.meme` files.
6. `6_fimo_scan_genome.sh` — FIMO scan of the genome with the discovered motifs.
7. `7_viz_motif_on_chrom.py` — chromosome-ideogram view of motif hits.

Cross-replicate variants: `avg_1_search_motif.py` → `avg_2_modisco_script.sh` →
`avg_3_modisco_report.sh` (average the 3 LM replicates, then MoDISco/report).

Nested follow-on analyses:
- `1_get_motif_pos_from_modisco/` — extract seqlet → genome positions from the
  MoDISco `.h5`.
- `2_motif_to_tss_dist/` — motif-to-TSS distance distributions.
- `3_motif_cluster/` — motif clustering + co-occurrence.

## Paths & external tools

All machine paths resolve from `config/paths.yaml` — `work_root` (LM scores /
genomes / corpus) and `motif_db_dir` (the MEME/YeTFaSCo annotation DB). No
absolute paths are baked in. Heavy external tools required on `PATH`:
`modisco`/`tfmodisco` and `fimo` (MEME suite). GPU is not needed here (the LM
predictions are precomputed by the `03_eval`/LM-scoring steps); these are
CPU/`bigmem` jobs.

> Generated outputs (`*_viz_seq/`, `archive/`, FIMO results) are not committed —
> only the reproducible code is. Run via `sbatch <step>.sh` on the paper's
> cluster, or adapt through `scripts/common/submit.sh` elsewhere.
