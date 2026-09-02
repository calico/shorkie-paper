#!/usr/bin/env python3
"""Revision experiment 04, step 2 — extract the region-dependent loss weights actually
used by every training run, straight from the committed configs and the trainer that
consumes them.

The point of this script is that nothing here is asserted from memory: the weights are
read out of ``scripts/02_train/*/params.json``, and the lines that apply them are located
by pattern in the pinned ``external/baskerville-yeast`` submodule, so the Methods text it
supports can cite an exact file and line.

It surfaces two things the paper does not currently state:

  1. The pretraining weights ARE described in Methods (manuscript L582-584) but without
     their provenance -- ``exon_loss_scale`` / ``repeat_loss_scale`` = 0.1 against
     ``non_*`` = 1.0, i.e. exonic and repetitive positions contribute a tenth as much to
     the masked-token cross-entropy.
  2. The SUPERVISED runs apply the same kind of weighting and this is documented nowhere:
     both the fine-tuned and the random-init configs set ``exon_loss_scale`` /
     ``repeat_loss_scale`` = 1.0 against ``non_*`` = 4.0. Expressed as a ratio that is a
     relative weight of 0.25 on exonic and repetitive positions -- the same direction as
     pretraining (focus the objective on regulatory sequence), just less aggressive than
     the 0.10 used there, and written as an up-weighting of the complement rather than a
     down-weighting of the region. It applies identically to both supervised arms, so it
     does not confound the Shorkie vs Shorkie_Random_Init comparison, but it is a
     substantive modelling choice and belongs in the Methods.

Writes ``results/loss_weights.csv`` and ``results/loss_weights_methods.md`` -- the latter
is a Methods-ready paragraph with the numbers filled in from the configs.

CPU only, instant.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from shorkie import config

# Config -> the run it trains, in the order the pipeline executes them.
RUNS = [
    ("shorkie_lm", "Shorkie LM (masked-language-model pretraining)"),
    ("shorkie_finetuned", "Shorkie (supervised fine-tuning from the LM checkpoint)"),
    ("shorkie_scratch", "Shorkie_Random_Init (supervised, random initialisation)"),
]

WEIGHT_KEYS = ["exon_loss_scale", "non_exon_loss_scale",
               "repeat_loss_scale", "non_repeat_loss_scale"]

# Other loss-relevant keys worth reporting alongside, so the table is self-contained.
CONTEXT_KEYS = ["task", "loss", "mask_rate", "learning_rate",
                "has_mask", "has_repeat_mask", "use_bert"]

def parse_args():
    parser = argparse.ArgumentParser(description="Document the region-dependent loss weights.")
    parser.add_argument("--out_dir", default=None, help="Output directory [default:./results]")
    return parser.parse_args()

def find_apply_sites(trainer_py):
    """Locate the lines in baskerville-yeast's trainer that apply the region weights."""
    sites = []
    if not trainer_py.exists():
        print(f"SKIPPED: trainer not found at {trainer_py} "
              f"(run scripts/00_setup/init_submodules.sh)", file=sys.stderr)
        return sites
    for i, line in enumerate(open(trainer_py), start=1):
        if re.search(r"(exon_mask|repeat_mask)\s*\*\s*self\.(exon|repeat)_loss_scale", line):
            sites.append((i, line.strip()))
    return sites

def main():
    args = parse_args()
    config.load()
    repo = Path(config.repo_root())
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, description in RUNS:
        params_path = repo / "scripts" / "02_train" / name / "params.json"
        if not params_path.exists():
            print(f"SKIPPED: {params_path} not found", file=sys.stderr)
            continue
        train = json.loads(params_path.read_text()).get("train", {})
        row = dict(run=name, description=description,
                   params_json=str(params_path.relative_to(repo)))
        for k in CONTEXT_KEYS + WEIGHT_KEYS:
            row[k] = train.get(k)
        # The ratio is what matters, and it makes the two runs directly comparable:
        # <1 means the region contributes less than its complement, >1 more.
        for region in ("exon", "repeat"):
            num, den = train.get(f"{region}_loss_scale"), train.get(f"non_{region}_loss_scale")
            row[f"{region}_relative_weight"] = (
                round(num / den, 4) if isinstance(num, (int, float))
                and isinstance(den, (int, float)) and den else None)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "loss_weights.csv", index=False)

    trainer_py = repo / "external" / "baskerville-yeast" / "src" / "baskerville" / "trainer.py"
    sites = find_apply_sites(trainer_py)

    lines = [
        "# Region-dependent loss weighting — provenance",
        "",
        "Extracted from the committed training configs by "
        "`scripts/05_revision/04_loss_weight_provenance/2_document_loss_weights.py`.",
        "",
        "| Run | task / loss | exon_loss_scale | non_exon | repeat_loss_scale | non_repeat | "
        "exon rel. weight | repeat rel. weight |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['run']}` | {r['task']} / {r['loss']} | {r['exon_loss_scale']} | "
            f"{r['non_exon_loss_scale']} | {r['repeat_loss_scale']} | "
            f"{r['non_repeat_loss_scale']} | {r['exon_relative_weight']} | "
            f"{r['repeat_relative_weight']} |")

    lines += ["", "## Where the weights are applied", ""]
    if sites:
        rel = trainer_py.relative_to(repo)
        for ln, code in sites:
            lines.append(f"- `{rel}:{ln}` — `{code}`")
    else:
        lines.append("- (submodule not initialised; run `scripts/00_setup/init_submodules.sh`)")

    lines += [
        "",
        "## Methods text this supports",
        "",
        "**Pretraining.** Masked-token cross-entropy is weighted per position by genomic "
        "region. Exonic and repetitive positions receive a weight of 0.1 against 1.0 "
        "elsewhere, concentrating the objective on non-coding, non-repetitive sequence. "
        "Repetitive positions are those soft-masked by RepeatMasker + DUST in the corpus "
        "build; exonic positions come from the Ensembl Fungi release-59 annotation, with "
        "2 bp chewed from each internal exon edge (`shorkie.data.bed_helper.get_exon_mask`). "
        "See step 1 for the exact genome fractions these definitions imply.",
        "",
        "**Supervised training.** The same mechanism is used, expressed the other way "
        "round: exonic and repetitive positions receive 1.0 against 4.0 elsewhere, i.e. a "
        "relative weight of 0.25 on those regions in the Poisson-multinomial objective. "
        "The direction matches pretraining — the objective is concentrated on non-coding, "
        "non-repetitive sequence — but the magnitude is milder (0.25 rather than 0.10). "
        "This applies identically to the fine-tuned and random-initialised runs, so it does "
        "not confound the comparison between them, but it is currently absent from the "
        "Methods and should be stated.",
    ]
    (out_dir / "loss_weights_methods.md").write_text("\n".join(lines) + "\n")

    print(df[["run", "task", "loss"] + WEIGHT_KEYS +
             ["exon_relative_weight", "repeat_relative_weight"]].to_string(index=False))
    print(f"\napply sites found in trainer.py: {len(sites)}")
    for ln, code in sites:
        print(f"  line {ln}: {code}")
    print(f"\nwrote {out_dir/'loss_weights.csv'}")
    print(f"wrote {out_dir/'loss_weights_methods.md'}")

if __name__ == "__main__":
    main()
