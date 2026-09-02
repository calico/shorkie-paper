# 08 · Is the Figure 2E t-SNE separation driven by element length?

The five element classes in Figure 2E have very different length distributions, but are
zero-padded to a common input length and mean-pooled across the full padded sequence. The fraction
of real sequence entering each embedding therefore differs by class — which could produce the
observed separation independently of any learned regulatory feature.

## Why this design

The mechanism is visible directly in the code, and that is worth
verifying before designing anything: `umap_cluster_promoter/1_predict_seqs_LM.py` centre-pads each
interval with `N` to 16,384 bp (lines 117–119) and mean-pools the layer activations across the
**full padded axis** (line 173). Every embedding is therefore scaled by roughly
(real length / 16,384), and that factor is a property of the class rather than of its regulatory
content.

So the question is not whether the artifact exists — it does — but how much of the published
separation it accounts for. That needs a **baseline**, not just an alternative pooling scheme: if a
one-dimensional length feature already separates the classes, then separation of a 16,384-long
mean-pooled embedding is not by itself evidence of learned structure.

Step 1 establishes that baseline with no GPU at all. Step 2 then recomputes the embeddings four
ways, each differing from the published scheme in exactly one respect, so step 3 can attribute what
changes:

| Scheme | What it changes | What it isolates |
|---|---|---|
| `published` | nothing — the baseline to reproduce | — |
| `masked_pool` | pools only over real (unpadded) positions | removes the pooling artifact, keeps genuine length differences |
| `length_matched` | every interval trimmed to a fixed 500 bp | removes both |
| `residualised` | regresses each dimension on log(length) | removes the linear length component post hoc |

`masked_pool` and `length_matched` are deliberately both present: the first tests whether the
artifact is purely a pooling bug, the second whether anything survives when the classes contribute
identical amounts of sequence.

Step 3 measures separation in the **full embedding space** (5-fold-CV k-NN accuracy and macro-F1),
not on the t-SNE coordinates — t-SNE is not a faithful metric space, and the published silhouette was
computed on the projection. The 2-D silhouette and a k-means ARI are reported alongside so the
published number stays comparable.

## What step 1 already found

**Element length alone classifies the five classes at 80.9% accuracy (chance 20%), macro-F1 0.805**
— on a class-balanced sample of **91 intervals per class**, the cap set by the 91 transposable
elements.

And the correspondence with the published separation is close to monotonic — the two best-separated
classes are precisely the two length extremes:

| Class | n | Median length | % of the 16,384 bp window | Published 2-D silhouette |
|---|---|---|---|---|
| tRNA | 299 | **73 bp** | 0.45% | **0.850** |
| Transposable element | 91 | **3,468 bp** | 21.2% | **0.636** |
| Promoter | 6,600 | 500 bp (fixed by construction) | 3.05% | 0.188 |
| Protein-coding gene | 6,600 | 1,070 bp | 6.53% | 0.049 |
| Intergenic region | 6,353 | 347 bp | 2.12% | −0.095 |

tRNAs contribute under half a percent of the pooled window and are the tightest cluster in the
published figure; transposable elements contribute 21% and are the second tightest. That is what the
confound predicts.

**This experiment is set up to report a negative result if the data say so.** If the controlled embeddings
do not beat the length-only baseline, the honest revision is to state that Figure 2E's separation is
substantially length-driven and to restrict the claim accordingly — and the panel should either be
recomputed under masked pooling or presented with the caveat attached.

## Steps

| Step | Script | Cost | Output |
|---|---|---|---|
| 1 | `1_element_length_distributions.py` | CPU, seconds | `results/element_lengths.csv`, `length_only_baseline.csv`, `element_lengths.png` |
| 2 | `2_embed_variants.py` | **GPU**, ~2,000 forward passes at the default 400 intervals/class | `results/embeddings_<scheme>.npz`, `embedding_meta.csv` |
| 3 | `3_quantify_separation.py` | CPU, minutes (t-SNE dominates) | `results/separation_metrics.csv`, `tsne_panel.png` |

```bash
bash 1_element_length_distributions.sh --gtf /path/GCA_000146045_2.59.gtf
scripts/common/submit.sh --profile gpu 2_embed_variants.sh
bash 3_quantify_separation.sh
```

## Inputs

Step 1 needs only the R64 GTF. Step 2 additionally needs the FASTA and the released Shorkie_LM
checkpoint (`data/download.sh --models lm`); it taps the first self-attention layer
(`multihead_attention`), the layer the published panel uses — override with `--layer`.

The published silhouette values quoted above come from
`reproduction/figure_02/recheck/fig2E_separation.csv`, so the comparison is against the reproduction's
own record of the published panel rather than a re-derivation.
