# 02 · Quantifying ISM motif recovery, Shorkie vs Shorkie_Random_Init

The paper states that Shorkie's ISM maps preserved regulatory motif signatures acquired
during pretraining whereas Shorkie_Random_Init failed to recover key motifs. That is a sweeping
negative supported only by side-by-side logos, and there are known counter-examples such as RRPE.
This replaces it with a measurement.

*(The RRPE panels are Figure 4B–C — FUN12 and KRE33 — not 3C; the substantive point stands.)*

## Why this design

The published claim is a sweeping negative ("failed to recover") supported only by side-by-side
logos. Two things have to be true for a quantitative replacement to be fair.

**First, the two models must be put on a common scale.** Their raw ISM magnitudes differ by ~2.5×
overall and by more than 10× on individual windows — Shorkie_Random_Init's attribution maps are
simply *louder*. Comparing raw attribution would measure that, not motif recovery. So saliency is
standardised within each window before anything else.

**Second, the comparison must be paired and the counter-examples must survive.** Both models are
scored at exactly the same motif occurrences in exactly the same windows, so the contrast is a
paired difference. Results are reported per motif with intervals rather than pooled, precisely so
that motifs where Shorkie_Random_Init does better remain visible; pooling would hide them.

The measurement per occurrence is

```
recovery = mean(standardised |saliency| inside the motif)
         − mean(standardised |saliency| in its ±50 bp flanks)
```

with flank positions covered by *any* other motif hit excluded, so a neighbouring site cannot
inflate the background and mask a real enrichment.

Two implementation choices worth noting. Motif scanning is **implemented natively** rather than
calling FIMO — the MEME suite is an optional dependency here (Figure 4 panel H caches its TomTom
output for exactly this reason), and significance is calibrated against a **dinucleotide**-shuffled
null so the poly-A tracts that dominate yeast promoters cannot masquerade as signal. And the
manuscript's own consensus strings are scanned alongside the database PWMs, because the database
Stb3 matrix finds fewer RRPE sites in these windows than the p=1e-4 threshold expects by chance —
without the consensus, RRPE itself would be untestable.

## What it found

**The claim as written is not supported.**

Across 397 paired promoter windows (137 RP + 260 TSS) and 121 motifs with ≥15 sites, only **2
motifs are significant after Holm correction**, and both are core-promoter/splicing elements rather
than sequence-specific TF sites:

| Motif | n sites | Shorkie | Random_Init | contrast [95% CI] | Holm p |
|---|---|---|---|---|---|
| TATA box (TATAAA) | 335 | +0.216 | −0.095 | **+0.431** [+0.335, +0.496] | 7×10⁻¹⁹ |
| 5' splice donor (GTATGT) | 56 | +0.247 | −0.094 | **+0.280** [+0.106, +0.547] | 0.047 |
| branch point (TACTAAC) | 27 | +0.369 | −0.211 | +0.523 [+0.154, +0.909] | 0.16 (n.s.) |

For the sequence-specific TF motifs the paper discusses, **none reaches significance**, and several
point estimates favour Shorkie_Random_Init:

| Motif | contrast | favours |
|---|---|---|
| Ume6 | −0.240 | Random_Init |
| Tbf1 | −0.214 | Random_Init |
| Fhl1 | −0.207 | Random_Init |
| Cbf1 | −0.160 | Random_Init |
| Abf1 | −0.133 | Random_Init |
| Rap1 | +0.123 | Shorkie (n.s.) |
| Reb1 | −0.009 | Random_Init (n.s.) |
| **RRPE consensus** | **+0.018** [−0.192, +0.358] | indistinguishable |

On RRPE specifically the honest answer is neither confirmation nor refutation: at n=48 sites the
interval spans zero comfortably. RRPE is not resolvable on this measure, which is itself a reason
not to have made a directional claim about it.

**Recommended revision.** Replace the sweeping sentence with the specific supported one: Shorkie
places significantly more attribution than Shorkie_Random_Init on **core promoter and splicing
elements** (TATA box, 5' splice donor, and by point estimate the branch point), while for individual
transcription-factor binding motifs the two models are statistically indistinguishable on these
promoter windows, with several point estimates favouring the random-init baseline.

## A second, separate problem with the published sentence: its scope

Beyond the statistics, "Across **all** ISM analyses" is broader than the data ever allowed. The
released ISM cache is extremely lopsided — **482 parts for Shorkie against 16 for
Shorkie_Random_Init**:

| Tree | Subsets with cached ISM |
|---|---|
| Shorkie | RP, TSS, TSS_select, TSS_select_targets, **SS** (splice sites), **RRB_targets**, and MET4 / MSN2 / MSN4 / RPN4 / SWI4 target sets |
| Shorkie_Random_Init | RP, TSS, TSS_select (+ two one-off panels) |

So the sentence covers splice-site, RRB and five TF-target analyses in which
Shorkie_Random_Init **was never run at all** — the comparison could not have been made there, by any
method. This is worth stating independently of the effect sizes, because it explains
*why* the claim reads as broader than the evidence: it was generalising from the promoter windows to
analyses that had no baseline. Step 5 closes the RRB half; the remainder should simply be narrowed to
the analyses where a paired comparison exists.

## Limitations

- The cached ISM covers **fold f0 only**, so this is a single-fold measurement.
- The paired windows are the RP and TSS promoter sets — see step 5 below, which closes this gap.
- Three of the 550 cached `scores.h5` files are truncated and are skipped; the scripts pair windows by
  genomic coordinate rather than by index so this cannot silently misalign the comparison.

## Step 5 · Settling RRPE at the panels where it is the headline motif

The released ISM cache has **16 RRB-target parts for Shorkie and zero for Shorkie_Random_Init**, so
steps 1–4 can only pair RP and TSS windows. RRPE and PAC are the headline motifs of **Figure 4B–C**
(FUN12, KRE33), which are RRB promoters — exactly where RRPE matters most. On the current
data the honest answer is "unresolvable at n=48 RRPE sites in the RP/TSS windows", which is true but
does not settle RRPE at those panels.

`5_run_random_init_rrb_ism.sh` closes that gap with one GPU array job (0–15). It mirrors the published
Shorkie invocation
(`04_analysis/shorkie/ism_motif/motif_shorkie__RP_TSS/ism_run/motif_shorkie_targets.sh` with
`exp_data="RRB_targets"`) exactly — same FASTA, same 500 bp windows, same `--rc`, same
`--stats logSED`, same targets sheet — changing only the model. Output lands in the canonical ISM tree
beside the Shorkie data, and `1_extract_ism_saliency.py` already lists
`gene_exp_motif_test_RRB_targets` in `SUBSETS` and pairs by genomic coordinate, so re-running steps
1–4 afterwards picks the RRB windows up with no arguments and RRPE/PAC appear with paired sites.

```bash
scripts/common/submit.sh --profile gpu --array 0-15 5_run_random_init_rrb_ism.sh
bash 1_extract_ism_saliency.sh && bash 2_scan_motifs.sh && bash 3_recovery_scores.sh && bash 4_plot_recovery.sh
```

## Steps

| Step | Script | Output |
|---|---|---|
| 1 | `1_extract_ism_saliency.py` | `results/saliency_cache.npz`, `results/windows.csv` |
| 2 | `2_scan_motifs.py` | `results/motif_hits.tsv`, `results/motif_scan_summary.csv` |
| 3 | `3_recovery_scores.py` | `results/motif_recovery.csv`, `results/motif_recovery_occurrences.csv` |
| 4 | `4_plot_recovery.py` | `results/motif_recovery.png` |

## Inputs and cost

CPU only, roughly ten minutes end to end and ~1 GB of transient memory. Needs the cached ISM scores
under `results.ism_scores`, the targets sheet, and a MEME-format yeast motif file under
`motif_db_dir` (`merged_meme_high_conf.meme` by default). No model weights, no GPU, and no MEME suite
installation.
