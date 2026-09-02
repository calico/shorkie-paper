# Additional analyses and controls — design document

How each experiment is set up, **why** it is set up that way, and what question it can answer. The
per-experiment `README.md` files carry the run recipes; this document is the argument that connects
them.

---

## What these are for

Two claims in the paper are broader than the evidence behind them, and both are addressed here: the
**optimal evolutionary scope** of the pretraining corpus (experiment 10) and the **transfer of
regulatory grammar** (experiments 09 and 11). Those are the ones worth the most compute. The rest
are missing controls, missing quantification, and presentation defects.

Three design principles run through all of them.

**Reuse the published recipe rather than reimplementing it.** Where an experiment compares against a
published number, it *imports* the code that produced that number — `mpra_common.py` for the MPRA
correlations, `build_3C_violin.py`'s panel recipes for the Figure-3 statistics, `fig4_common.py`'s
saliency definition for ISM. If the comparison were recomputed independently, any difference could be
a difference in method rather than a difference in model, and the experiment would answer nothing.

**Design so that either outcome is reportable.** Several of these can go against the paper. Each
README states in advance how to report the result if it does. An experiment that can only come out
one way is not a control.

**Put a number on the confound before spending GPU time on it.** Four experiments (08, 10, and the
provenance halves of 03 and 04) turned out to be answerable, at least in part, from artifacts already
on disk. Those ran first, and in two cases they changed what the expensive follow-up needs to show.

---

## Group A — questions answerable from existing artifacts

### 01 · Cross-fold confidence intervals

**Setup.** Every trained fold already wrote per-track accuracy tables, so no model is re-run. Three
intervals are computed — a t interval on the per-fold statistic (n=8), a bootstrap over tracks, and a
hierarchical bootstrap over both — plus a **paired** per-fold contrast, since Shorkie and the
baselines are evaluated on the same folds.

**Why paired.** Marginal intervals overlap; the paired interval does not. Comparing two independent
intervals would throw away the pairing and understate the evidence.

**What it answered.** 0.776 [0.715, 0.841] vs 0.703 [0.664, 0.767] against the LR-optimised baseline;
paired Δ +0.063 [+0.028, +0.098], Wilcoxon p = 0.016. Shorkie wins in **7 of 8** folds against that
baseline (fold 2 is the exception, by −0.004) and in **all 8** against the un-tuned one, where
p = 0.0078 — the smallest value a signed-rank test can return at n=8. The harder comparison is
significant but not saturated, and should be reported that way.

**An inconsistency it surfaced.** The manuscript's `0.67` is the *un-tuned* baseline
(`supervised_unet_small_bert_drop`, lr 1e-4); Figure 3C plots the *LR-optimised* one (0.703). Both
numbers are correct for their own baseline, but quoting the weaker one in the headline understates
the strongest available control.

### 02 · Quantifying the ISM motif-recovery claim

**Setup.** 550 cached ISM `scores.h5` files exist across both model trees — though very unevenly:
**482 for Shorkie against 16 for Shorkie_Random_Init** (see the scope note below). Per-position
saliency is reconstructed with the published Equations 18–20, motif occurrences are located with a
native PWM scan, and recovery is scored as (saliency inside the motif − saliency in its flanks).

**Two choices that decide whether the comparison means anything.** First, saliency is **standardised
within each window**: the models' raw magnitudes differ by ~2.5× overall and >10× in places, so a raw
comparison would measure loudness. Second, flank positions covered by *any* other motif hit are
excluded, so a neighbouring site cannot inflate the background.

**Why the scanner is native rather than FIMO.** MEME is an optional dependency here (Figure 4 panel H
caches its TomTom output for exactly this reason), and significance is calibrated against a
**dinucleotide**-shuffled null so the poly-A tracts that dominate yeast promoters cannot masquerade as
signal. The manuscript's own consensus strings are scanned alongside the database PWMs — without them
RRPE would be untestable, because the database Stb3 matrix finds fewer sites in these windows than the
p=1e-4 threshold expects by chance.

**What it answered.** Of 121 motifs, only **two** survive Holm correction, and both are core
promoter/splicing elements: TATA box (+0.431, p=7×10⁻¹⁹) and the 5' splice donor (+0.280, p=0.047).
For sequence-specific TF motifs **none** is significant and several point estimates favour
Shorkie_Random_Init (Ume6 −0.240, Tbf1 −0.214, Fhl1 −0.207). RRPE is unresolvable at n=48. The
sweeping sentence should become the specific supported one.

**The claim's scope is wrong too.** "Across *all* ISM analyses" covers splice-site, RRB and five
TF-target window sets in which Shorkie_Random_Init was **never run at all** — the comparison could not
have been made there, by any method. Step 5 closes the RRB half; the remainder should be narrowed to
the analyses where a paired comparison exists.

### 03 · RNA-seq atlas accounting

**Setup.** Not an experiment — a parsing job, which is the correct answer to a question about
arithmetic. The committed targets sheet encodes each track's design in its identifier
(`<GENE>_T<minutes>_S<sample>`), so the partition is **derived from the sampling schedules** and then
cross-checked against the 8 TFs the Methods name, rather than assumed.

**What it answered.** 3,053 = 580 (8 TFs, schedule 0/5/10/15/30/45/60/90, 4–12 replicates per point)
+ 2,473 (329 further genes, schedule 0/5/10/20/40/70/120/180, mostly one replicate). Exactly eight
genes use the microarray-matched schedule and they are exactly the eight the Methods name — a strong
internal check. The Methods say 460 genes were attempted; **329** yielded released tracks, so 131 did
not survive QC, and the paper currently implies all 460 contributed.

### 04 · Loss-weight provenance

**Setup.** The down-weighting **is** in the Methods (manuscript L582–584) and in Equation 1; what is
missing is everything connecting that sentence to the Results — the word "coding" versus "exonic", the
percentages, and the annotation provenance. Because "coding" is ambiguous, **every** plausible
definition is computed side by side rather than one being asserted.

**What it answered.** The repetitive figure is exact: 892,009 soft-masked bp of 12,071,326 = **7.389%**
against the published 7.39%. The coding figure is bracketed 70.9–73.8% and no standard definition
lands on 72% — closest is protein-coding exon at 71.2%. And a second, genuine gap surfaced: the
supervised configs weight regions too (relative weight 0.25 on exonic/repetitive positions against
0.10 in pretraining) and this appears nowhere in the Methods. It applies identically to both
supervised arms, so it does not confound the headline comparison — but it belongs in the paper.

### 05 · Figure and caption corrections

**Setup.** Three presentation defects, each fixed in a way that *demonstrates* the fix rather than
asserting it.

- **6B/6C colours.** The published aggregates are dark green, dark red and black — three colours that
  differ little in hue and almost not at all in lightness. Worst-case CIE76 separation under normal
  vision and three simulated colour-vision deficiencies goes **4.3 → 31.1** (a just-noticeable
  difference is ~2.3, so the published palette's worst pair was barely two colours at all). Marker
  shape and line style become redundant encodings, and the AUROC/AUPRC values are asserted unchanged
  against the committed `fig6_BC.csv`.
- **4D schematic.** Rather than nudging labels by eye, the geometry is **measured from the annotation**:
  321 introns, median length 116 nt, median branch point 40 nt from the 3' splice site. The branch
  point belongs near the 3' end, not mid-intron.
- **Figure 5 caption.** Narrower than it looks: **only the caption is wrong**. The Results text
  (L274/L276) and the figure agree with each other; the caption alone has D and E swapped, and the
  swap propagates to I/J through "analogous to (A–E)".

---

## Group B — new measurements requiring GPU

### 06 · Shorkie_Random_Init on MPRA

**Question.** The MPRA result is a zero-shot transfer claim. Is it the language-model prior that
carries the transfer, or would any yeast-trained supervised model do as well?

**Setup.** Only the initialisation may differ. The reporter genes, insertion sites, analysis context,
sequence categories, flags, targets sheet and genome are all held fixed, and the correlation recipes
are imported from the published Figure-6 loaders.

**A blocker it had to clear first.** The published runs pass `--ctx <gene>.tsv`, and those files are
not where the run scripts say they are — so the MPRA pipeline could not be re-run **for any model**.
They survive one directory deeper, next to a generator that depended on a hand-maintained list of 22
literal TSS coordinates. Step 0 regenerates them from the GTF instead, and validates that all 22 come
out **byte-identical** to the originals.

### 07 · Gene-restricted ISM

**Question.** Equation 17 sums over all 896 output bins (~14.3 kb), so a promoter mutation is scored
against neighbouring genes too. Does restricting to the target gene change any conclusion?

**Setup.** All three bin scopes — all bins, gene body, TSS ±1 kb — are computed **over the same
mutations in the same forward passes**, so the comparison is exact. The TSS window is there because
`gene_body` inherits whatever the annotation says the gene is, which for a short gene is a handful of
noisy bins; it separates "restricting helps" from "restricting to *this* interval helps".

**What makes it worth measuring rather than conceding.** The paper is internally inconsistent: its
*variant* scoring already restricts to gene-body bins (Methods L1175). The machinery and the intent
both exist; the ISM pipeline simply does not use them.

**The time-course half.** Neighbouring-gene coverage could also affect the comparison across
induction time points — Figure 5C, the 8×8 distance heatmap between per-timepoint ISM logos — and
correlating static saliency would not test that. So the ISM is computed per bin scope **and per
induction time point**, reusing the published track partition (`fig05_lib.tp_tracks`), and step 2
rebuilds that distance matrix under each scope with the published recipe.

**Reading it.** Not whether the numbers change — they must, the denominators differ — but whether the
*conclusions* do. Hence rank agreement, top-25 position overlap (the concrete question for a logo
panel), the time-course distance structure, and contamination: how much reference coverage falls
outside the gene at all.

### 08 · The Figure 2E length confound

**Question.** Does class separation survive once element length is controlled?

**Setup.** Four embedding schemes, each differing from the published one in exactly one respect —
masked pooling, length-matched intervals, and length residualisation — so the attribution is clean.
`masked_pool` and `length_matched` are both present deliberately: the first tests whether the artifact
is purely a pooling bug, the second whether anything survives when classes contribute identical
sequence. Separation is measured in the **full embedding space**, not on t-SNE coordinates, which are
not a faithful metric space.

**Why a baseline was needed before any GPU work.** If length alone separates the classes, then
separation of a mean-pooled 16,384-long embedding is not evidence of learned structure. Step 1 built
that baseline: **length alone classifies the five classes at 80.9%** (chance 20%, on a balanced sample
of 91 per class), and the two best-separated classes in the published panel are exactly the length
extremes — tRNA (median 73 bp, 0.45% of the window, silhouette 0.850) and transposable elements
(3,468 bp, 21%, 0.636). The mechanism is confirmed; what remains is how much of the separation it
explains.

### 09 · Dependency maps — words or grammar?

**Question.** The paper claims syntax; motif recovery demonstrates vocabulary. Is there evidence of
syntax?

**Setup.** The method is already implemented in this repository — for **SpeciesLM**, the comparator,
not Shorkie_LM. Step 1 is a faithful port, and cheaper than it looks: because the model emits a
distribution at every position, one forward pass per mutant yields a whole row of the matrix, so an
L×L map costs ~3L passes.

**Why raw dependency is not evidence.** Nearby positions covary because they share a receptive field.
Step 2 therefore tests **between-motif** dependency against a **distance-matched** background;
within-motif dependency is only the sanity check.

**Why step 3 intervenes instead of correlating.** Spacing is what the definition of grammar actually
names, so the spacer between two motifs is lengthened and shortened with composition held fixed, and
two signatures are sought: monotonic decay, and ~10.5 bp helical phasing — which motif recognition
alone cannot produce. The DTD1 intron is included as a positive control: the paper already shows the
model reconstructs all three splice signals, and their arrangement is genuine syntax. If the method
cannot find structure there, that is diagnostic of the method.

**Reading it.** A flat response means words, not grammar, and the honest revision is to tone L148 and
L216–218 down.

### 10 · Corpus-size-matched fungal control

**Question.** Is the 165-genome "sweet spot" a phylogenetic-scope effect, or a data-volume artifact?

**Setup — two decisions a one-sentence description leaves open.** Sampling from the whole fungal
corpus would re-include its ~170 Saccharomycetales genomes, so the "broad" corpus would partly *be*
the narrow one; the draw is therefore from the **1,191 non-Saccharomycetales** genomes only. And a
uniform draw would be dominated by the best-represented orders, so the default is **stratified** across
taxonomic orders — the result spans 66 orders against the reference tier's 1. Everything else is held
fixed, including the `unet_small` architecture (the one Figure 1F/G used, *not* the released
`unet_small_bert_drop`, or the new point would not land on that figure) and the held-out split.

**What the CPU companion already established.** The confounds are quantifiable with no GPU at all, and
one of them is already an answer:

> **The broad fungal corpus has 1.62× MORE training windows than the Saccharomycetales corpus, not
> fewer.** Data volume alone cannot explain why it underperforms.

But the other confounds are large: fungal genomes are 2.4× bigger, far more fragmented (9.5%
chromosome-level vs 36.4%), and only **6.2%** of each raw assembly survives filtering into training
windows against **75.9%** — so each genome contributes far less usable sequence, spread across 1,361
species instead of 165.

**The peak has two sides, and only one was being tested.** The corpus-matched control addresses the
upper side (vs 1341_Fungus), where the broad corpus has 1.62× *more* windows so volume cannot be the
explanation. The lower side (vs 80_strains) runs the other way — Saccharomycetales has **3.77× more**
windows, so volume could explain that half entirely. Step 7 closes it, and is the cheaper of the two
arms: no new genomes, no corpus build, just subsample the existing tier and retrain.

**A third confound, found by looking.** The training logs settle the "optimisation difficulty"
question — in the uncomfortable direction. The four tiers were **not trained on a matched schedule**:
R64 and 80_strains had `train_epochs_max=500` / `patience=50`, while Saccharomycetales and
1341_Fungus had `10000` / `1000`. Each run converged within its own budget, so none is
straightforwardly under-trained, but a 20× difference means "converged" is not "trained comparably".
This is why step 7 matches the schedule as well as the window count.

**Reading it.** *Upper side:* control ≈ 1341_Fungus means scope is the explanation and the claim
stands; control ≈ 165_Saccharomycetales means volume was; in between, report the decomposition.
*Lower side:* if the downsampled model still beats 80_strains, scope is doing the work on both sides;
if it ties, the lower half of the peak is a volume/schedule artifact and the claim must be restated.

### 11 · Layer-group selective re-initialization

**Question.** Where does the transferred information live? The ablation cannot say *what* was
transferred, but if the answer is "everywhere", that settles whether a specific attribution is
warranted.

**Setup.** The trunk is three stages, so there are **three** arms, not two — the U-Net decoder is free
to add and separates local sequence features from output reconstruction. Groups are derived **from the
model graph** (first to last `MultiHeadAttention`), not from generated layer names, and the partition
is printable before anything is written. Re-initialisation copies pretrained weights into a freshly
built network for every layer outside the target group, so retained layers are bit-exact and reset
layers start exactly where a from-scratch model would. Two of the five arms are free: full-LM is
Shorkie and all-random is Shorkie_Random_Init, both released.

**The trap.** The groups are not the same size — the transformer holds ~60% of trunk parameters
against ~16% for the convolutional tower. A bigger drop from resetting it is expected on capacity
grounds alone, so the interesting quantity is the drop **relative to the fraction discarded**, and the
parameter table is printed next to the metrics for exactly that reason. A second limit: re-initialising
a group and then fine-tuning measures the *value of a pretrained initialisation* for that group, not
its information content — fine-tuning can relearn a reset group from the supervised data.

**Reading it.** Distributed loss is a real result, not a failure.

---

## What is deliberately not done

**Pretraining an MPRA model such as DREAM-RNN.** It is a different modelling programme rather than a
missing control: a different architecture and framework, MLM pretraining of a model that was never
designed for it, and retraining on a 6.7M-sequence library. Experiment 06 covers the half that is in
scope — whether the LM prior, rather than supervised training alone, is what carries the zero-shot
MPRA transfer.

## Cost summary

| Tier | Experiments | Cost |
|---|---|---|
| No new compute | 01, 02, 03, 04, 05, 08 step 1, 10 steps 1–2 (incl. training dynamics) | minutes on CPU |
| GPU inference | 02 step 5, 06, 07, 08 step 2, 09 | hours to a day per experiment |
| GPU training | 10 steps 4–7, 11 step 2 | days per arm; 10 steps 4–6 need a corpus build first, step 7 does not |

The first tier is complete. The second and third are scripted and dry-run verified.
