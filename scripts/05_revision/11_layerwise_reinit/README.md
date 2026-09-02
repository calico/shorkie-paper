# 11 · Which pretrained layer group carries the transfer?

The paper attributes the downstream gain to transfer of specific content — "exon-intron
structure and regulatory grammar". That attribution is hard to support: the improvement could equally
come from motif representations or from general inductive biases acquired during pretraining.

## Why this design

The paper attributes the downstream gain to specific transferred content ("exon-intron structure and
regulatory grammar"). A layer-group ablation cannot confirm *what* was transferred, but
it can say *where* it lives in the network — and if the answer is "everywhere", that is itself the
answer to whether a specific attribution is warranted.

The architecture makes the split clean. From `scripts/02_train/shorkie_lm/params.json` the trunk is
exactly three stages — `conv_dna + res_tower ×7` → `transformer_tower ×8` → `unet_conv ×7` — so there
are **three** re-initialisable groups, not two. The U-Net decoder is a third arm that is free to add
and separates "local sequence features" from "output reconstruction".

Two implementation choices matter:

**Groups are derived from the model graph, not from layer names.** Keras layer names depend on
construction order and would silently mis-assign if anything changed. Instead the transformer group
is every layer from the first `MultiHeadAttention` to the last, the convolutional tower is everything
before, the decoder everything after. `--list_layers` prints the partition so it can be checked
before any checkpoint is written. Validated on the released checkpoint:

| Group | Layers | Parameters | Share |
|---|---|---|---|
| conv tower | 74 | 2,232,864 | **16.3%** |
| transformer | 78 | 8,152,448 | **59.7%** |
| U-Net decoder | 74 | 3,280,516 | **24.0%** |
| total | 226 | 13,665,828 | 100% |

**Re-initialisation uses the architecture's own initialisers.** A second `SeqNN` is built fresh from
the same config and the pretrained weights are copied in for every layer *outside* the target group.
So retained layers are bit-exact Shorkie_LM and reset layers start exactly where a from-scratch model
would — not from noise of some other scale.

## Five arms, two of them free

| Arm | LM weights kept | Status |
|---|---|---|
| full LM | all | **Shorkie** — already released, do not retrain |
| `conv_reset` | transformer + decoder | to train |
| `transformer_reset` | conv tower + decoder | to train |
| `decoder_reset` | conv tower + transformer | to train |
| all random | none | **Shorkie_Random_Init** — already released, do not retrain |

Every arm uses `scripts/02_train/shorkie_finetuned/params.json` unchanged and the same 8-fold
`westminster_train_folds.py` invocation as the published Shorkie, so the *only* difference between
arms is which part of the checkpoint was reset. Step 3 scores them with experiment 01's machinery,
which puts them on the same scale as the published numbers with the same cross-fold intervals and
paired-fold statistics.

**A limit on what this can show, worth stating up front.** Re-initialising a group and then
fine-tuning measures **the value of a pretrained initialisation for that group** — not the
information content of the group. Fine-tuning has 8 folds of supervised data to relearn a reset group
from, so a small drop means "the supervised task can recover this", not "the LM had nothing here".
That is a different limitation from information being distributed across the model, and both should
be reported.

## How to read it — and the trap to avoid

**The three groups are not the same size.** The transformer holds ~60% of the trunk parameters
against ~16% for the convolutional tower, so resetting it discards nearly four times as much of the
model. A larger drop is therefore expected on capacity grounds alone, and the interesting quantity is
the drop *relative to the fraction of the model discarded* — which is why step 3 prints the parameter
table next to the metrics.

| Outcome | Interpretation |
|---|---|
| conv reset costs far more per parameter than transformer reset | the transfer is carried by local sequence features — motif detectors — supporting "motif representations" over "grammar" |
| transformer reset costs far more per parameter | the transfer is carried by long-range context, closer to the paper's claim |
| all arms lose similarly per parameter | the information is **distributed**; no specific attribution is warranted and lines 216–218 should be weakened |

The third outcome is a real result, not a failure, and the response should say so plainly if that is
what the data show.

## Steps

| Step | Script | Cost | Output |
|---|---|---|---|
| 1 | `1_make_reinit_checkpoints.py` | CPU, minutes | `results/checkpoints/<arm>.h5`, `layer_partition.csv`, `layer_assignment.csv` |
| 2 | `2_train_arms.sh <arm>` | **GPU, days per arm** | `results/arms/<arm>/train/f{0..7}c0/` |
| 3 | `3_eval_arms.py` | CPU, minutes | `results/arm_metrics.csv`, `arm_comparison.png` |

```bash
python 1_make_reinit_checkpoints.py --list_layers      # check the partition first
python 1_make_reinit_checkpoints.py
for arm in conv_reset transformer_reset decoder_reset; do
  scripts/common/submit.sh --profile gpu 2_train_arms.sh "$arm"
done
bash 3_eval_arms.sh
```

Step 2 is the expensive part: three full 8-fold supervised fine-tunes, each costing what training
Shorkie itself cost. Step 3 needs experiment 01 to have been run first, since it reads the released
anchors from `01_headline_uncertainty/results/headline_ci.csv`.

## Inputs

The released Shorkie_LM checkpoint (`data/download.sh --models lm`) and the supervised TFRecords
(`datasets.supervised_data`). Step 1 needs no GPU.
