# Nucleotide dependency maps (DNA-LM)

`compute_and_visualize_dep_maps_RPL43B.py` (+ the companion notebook) computes and
visualizes **nucleotide dependency maps** for a yeast locus (RPL43B) from a
masked DNA language model — the in-silico "mutate position *i*, read the change at
position *j*" analysis that exposes epistatic / structural dependencies between
bases.

## Method provenance (third-party)

The dependency-map method and its reference implementation are **not ours**. They
come from the Gagneur lab:

- **Repository:** <https://github.com/gagneurlab/dependencies_DNALM> (pinned commit
  `0a12361`)
- **License:** MIT © 2024 Gagneur lab
- **Manuscript:** *"Nucleotide dependency analysis of DNA language models reveals
  genomic functional elements."*

This directory contains **only the Shorkie-specific wrapper** that applies the
method to our locus of interest. It is **self-contained** — it does not import the
upstream package — so to reproduce it you do not need to clone that repo; the
single commented breadcrumb `# from utils import plot_weights` points at the
upstream helper it was adapted from. We deliberately do **not** vendor or
submodule the upstream code (per the release plan); install it separately if you
want the generic, model-agnostic implementation and the original figures.

## Environment

This is a **PyTorch / HuggingFace** analysis (`torch`, `transformers`,
`flash-attn`, `datasets`, `biopython`) applied to a BERT-style DNA-LM — a
**different stack** from the rest of Shorkie (TensorFlow / `baskerville`). It will
not run in the `yeast_ml` conda env; create a separate environment matching the
upstream `requirements.txt` (torch 2.1, transformers 4.26, flash-attn 2.0.x, …)
and an NVIDIA GPU. Only `config` (path resolution) is shared with the rest of
this repo (`from shorkie import config`).
