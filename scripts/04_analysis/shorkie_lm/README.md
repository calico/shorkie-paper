# 04_analysis/shorkie_lm — language-model interpretability

> **Advanced / reproduction-only.** You don't need this to *run* Shorkie — use `examples/` +
> `minimal_example/` on the released artifacts (`data/download.sh`). These analyses produce the
> gated intermediates behind the figure notebooks (`notebooks/fig0N_*.ipynb`).

Stage-04 analyses of **Shorkie_LM**, the masked DNA language model: what
regulatory grammar it learned, where it attends, how it organizes promoters in
embedding space, and a single-locus case study (SMT3) with nucleotide dependency
maps. These steps consume precomputed LM scores/embeddings; most are CPU
(`bigmem`) jobs — only the SMT3 inference and embedding-prediction steps touch a
GPU. Working `*.py` scripts keep a paired `*.sh` SLURM runner.

| Subdirectory | Purpose |
|---|---|
| `motif_analysis/` | Per-base LM contribution scores → TF-MoDISco motif discovery → annotation/visualization. See per-variant sub-pipelines below. |
| `attention_map/` | `1_visualize_attention.py` — extract and plot transformer attention maps for a locus across LM/supervised model variants. |
| `umap_cluster_promoter/` | Promoter embedding clustering: `1_predict_seqs_LM` (compute LM embeddings) → `2_viz_clusters_LM` (t-SNE/UMAP visualization) → `3_viz_clusters_LM_clustering` (K-Means cluster assignment); `2_check_embedding.py` is a sanity check. |
| `lm_SMT3_viz/` | SMT3 case study: `1`–`2` DNA-logo PWMs (SpeciesLM vs Shorkie_LM), `3_inference_smt3_unmasked` (run the LM over SMT3 windows), `4_viz_smt3_logo_unmasked` (logo of predictions), `5_verify_alignment` (coordinate check). Nested `dependency_map/` computes nucleotide dependency maps (separate PyTorch/HuggingFace stack — see *Paths & external tools* below). |

## `motif_analysis/` variants

The motif pipeline is run several times over different sequence sets; each shares
the same machinery (search → MoDISco → report → visualize, with `avg_*` variants
pooling the 3 LM replicates):

| Variant | Sequence set |
|---|---|
| `motif_lm/` | In-distribution *Saccharomycetales* motifs (full pipeline: search, MoDISco, reports, TSS-distance, clustering, enrichment, ideograms). |
| `motif_lm__processed_genome/` | Same pipeline on the repeat-masked/processed genome; adds `5_map_motif_2_genome.py`. |
| `motif_lm__RP_TSS/` | Motifs restricted to ribosomal-protein (RP) and TSS gene sets (starts with `0_create_tfrecords.sh`; maps MoDISco patterns to the MEME DB). |
| `motif_lm__unseen_species/` | Cross-species generalization on held-out clades — the same search → MoDISco → report → visualize machinery run per held-out clade. |
| `motif_db_viz/` | Reference-DB motif logos (MEME DB, YeTFaSCo PFMs, ambiguous-string logos) for comparison. |

## Paths & external tools

All machine paths resolve from `config/paths.yaml` via `shorkie.config` — the keys
used here are `work_root` (LM scores, embeddings, genomes, model `.h5`) and
`motif_db_dir` (the MEME/YeTFaSCo annotation DB). No absolute paths are baked in.
Heavy external tools required on `PATH`: `modisco`/`tfmodisco` (motif discovery)
and `fimo` (MEME suite, for genome scans); logos use `logomaker`.

> The `dependency_map/` sub-analysis uses a **separate PyTorch/HuggingFace stack**
> (not the `yeast_ml` TF env) and adapts a third-party nucleotide-dependency-map method.
> Generated outputs are not committed; run via `sbatch <step>.sh`, or adapt through
> `scripts/common/submit.sh`.
