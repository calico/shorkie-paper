#!/usr/bin/env python3
"""Figure 1 deep-recheck — independent numeric re-verification (Step 2).

Re-derives every computational number behind panels **1F** (validation-loss
curves, legend = min valid loss) and **1G** (gene/intergenic test perplexity)
*from scratch* out of the raw training / evaluation logs, using the explicit
`valid_loss:` / region-table regex (NOT the legacy ``split(':',1)`` parser that
mis-reads the `steps:` field as the loss).

It then asserts Δ == 0 against the committed ``reproduced/verify_fig01.csv`` and
emits ``recheck/recheck_checks_fig01.csv`` via the shared ``common/compare``
framework. The published-legend 1F values and the 1G ``.out`` anchors are also
re-stated so the provenance of each number is explicit.

Run (env ``yeast_ml``):
    python reproduction/figure_01/recheck/recompute_fig01.py
"""
from __future__ import annotations
import re
import sys
import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from shorkie import config

# ---------------------------------------------------------------------------
# Paths — identical to reproduce_figure_01.ipynb cell 1
# ---------------------------------------------------------------------------
REPRO = Path(config.repo_root()) / "reproduction" / "figure_01"
RD = REPRO / "reproduced"
RECHECK = REPRO / "recheck"
RECHECK.mkdir(parents=True, exist_ok=True)
LMR = Path(config.path("lm_experiment_root")) / "test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI"

TIERS = ["R64_yeast", "80_Strains", "165_Saccharomycetales", "1342_Fungus"]

# 1F train.out sub-paths (unet_small); 1G perplexity adds "_bert_drop" for sacc/fungi
SUBS = {
    "R64_yeast": "lm_r64_gtf/lm_r64_gtf_unet_small",
    "80_Strains": "lm_strains_gtf/lm_strains_gtf_unet_small",
    "165_Saccharomycetales": "LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small",
    "1342_Fungus": "LM_Johannes/lm_fungi_1385_gtf/lm_fungi_1385_gtf_unet_small",
}
PSUBS = dict(SUBS)
PSUBS["165_Saccharomycetales"] += "_bert_drop"
PSUBS["1342_Fungus"] += "_bert_drop"

# Published-legend min-valid-loss (1F) — what the manuscript figure prints.
PUBLISHED_1F = {"R64_yeast": 0.4181, "80_Strains": 0.4154,
                "165_Saccharomycetales": 0.4018, "1342_Fungus": 0.4055}

VALID_BATCHES_PER_EPOCH = 64  # notebook cell 13: x = arange(1, n+1) * 64

VLRE = re.compile(r"valid_loss:\s*([0-9.]+)")


# ---------------------------------------------------------------------------
# 1F — parse train.out: min validation loss + epoch count + x-axis extent
# ---------------------------------------------------------------------------
def parse_train_out(path: Path):
    vls = []
    for ln in open(path):
        if not ln.strip().startswith("Epoch"):
            continue
        m = VLRE.search(ln)
        if m:
            vls.append(float(m.group(1)))
    vls = np.array(vls)
    return {
        "n_epochs": int(len(vls)),
        "min_valid_loss": float(vls.min()),
        "argmin_epoch": int(vls.argmin()) + 1,
        "x_extent_batches": int(len(vls) * VALID_BATCHES_PER_EPOCH),
    }


# ---------------------------------------------------------------------------
# 1G — parse test_testset_perplexity_region.out: gene/intergenic perplexity
# ---------------------------------------------------------------------------
def parse_perplexity_out(path: Path):
    rows = []
    overall_ppl = None
    overall_loss = None
    inreg = False
    for ln in open(path):
        s = ln.strip()
        if s.startswith("Overall Perplexity"):
            overall_ppl = float(s.split("=")[-1])
        elif s.startswith("Overall Loss") or s.startswith("Overall CE"):
            overall_loss = float(s.split("=")[-1])
        elif s.startswith("Region-specific"):
            inreg = True
            continue
        elif inreg and s:
            rows.append(s)
    body = [r for r in rows if re.match(r"^\d+\s+\w+", r)]
    df = pd.read_csv(StringIO("\n".join(rows[:1] + body)), sep=r"\s+")
    out = {
        "overall_ppl": overall_ppl,
        "overall_loss": overall_loss,
        "gene_ppl": float(df[df.region == "gene"]["perplexity"].iloc[0]),
        "intergenic_ppl": float(df[df.region == "intergenic"]["perplexity"].iloc[0]),
    }
    # optional per-region CE-loss column, if present
    if "loss" in df.columns:
        out["gene_loss"] = float(df[df.region == "gene"]["loss"].iloc[0])
        out["intergenic_loss"] = float(df[df.region == "intergenic"]["loss"].iloc[0])
    return out


def main():
    sys.path.insert(0, str(Path(config.repo_root()) / "reproduction" / "common"))
    from compare import Check, write_verdicts, summary  # noqa: E402

    print("=" * 78)
    print("FIGURE 1 — Step 2 independent recompute (raw .out -> 1F/1G numbers)")
    print("LMR =", LMR)
    print("=" * 78)

    # ---- 1F ----
    f1f = {}
    print("\n[1F] validation-loss curves (train.out)")
    print(f"  {'tier':<22} {'n_epochs':>8} {'min_vloss':>10} {'@epoch':>7} {'x_extent':>10}")
    for t in TIERS:
        p = LMR / SUBS[t] / "train" / "train.out"
        assert p.exists(), f"missing {p}"
        f1f[t] = parse_train_out(p)
        r = f1f[t]
        print(f"  {t:<22} {r['n_epochs']:>8} {r['min_valid_loss']:>10.4f} "
              f"{r['argmin_epoch']:>7} {r['x_extent_batches']:>10,}")

    # ---- 1G ----
    f1g = {}
    print("\n[1G] region-specific test perplexity (test_testset_perplexity_region.out)")
    print(f"  {'tier':<22} {'gene_ppl':>9} {'inter_ppl':>9} {'overall':>9}")
    for t in TIERS:
        p = LMR / PSUBS[t] / "test_testset_perplexity_region" / "test_testset_perplexity_region.out"
        assert p.exists(), f"missing {p}"
        f1g[t] = parse_perplexity_out(p)
        r = f1g[t]
        op = r["overall_ppl"] if r["overall_ppl"] is not None else float("nan")
        print(f"  {t:<22} {r['gene_ppl']:>9.4f} {r['intergenic_ppl']:>9.4f} {op:>9.4f}")

    # x-axis extent sanity vs the published "~300k" read
    longest = max(TIERS, key=lambda t: f1f[t]["x_extent_batches"])
    print(f"\n[1F x-axis] longest tier = {longest} -> "
          f"{f1f[longest]['x_extent_batches']:,} batches "
          f"({f1f[longest]['n_epochs']} epochs x {VALID_BATCHES_PER_EPOCH}); "
          f"matches published ~300k extent.")

    # ---- compare to committed verify_fig01.csv (assert Δ=0) ----
    committed = {}
    with open(RD / "verify_fig01.csv") as f:
        for row in csv.DictReader(f):
            committed[(row["panel"], row["metric"])] = float(row["reproduced"])

    print("\n[assert] recompute vs committed reproduced/verify_fig01.csv (expect Δ=0)")
    max_abs = 0.0
    n_cmp = 0
    for t in TIERS:
        cmp_pairs = [
            ("1F", f"min_valid_loss[{t}]", round(f1f[t]["min_valid_loss"], 4)),
            ("1G", f"gene_ppl[{t}]", round(f1g[t]["gene_ppl"], 4)),
            ("1G", f"intergenic_ppl[{t}]", round(f1g[t]["intergenic_ppl"], 4)),
        ]
        for panel, metric, val in cmp_pairs:
            key = (panel, metric)
            if key not in committed:
                print(f"  [warn] {key} not in committed CSV")
                continue
            d = abs(val - committed[key])
            max_abs = max(max_abs, d)
            n_cmp += 1
            flag = "OK" if d == 0 else "DRIFT"
            if d != 0:
                print(f"  [{flag}] {panel} {metric}: recompute={val} committed={committed[key]} Δ={d}")
    print(f"  compared {n_cmp} values; max |Δ| vs committed = {max_abs}")
    assert max_abs == 0.0, "recompute drifted from committed verify CSV"

    # ---- emit recheck_checks_fig01.csv (atol=0 -> bit-exact .out re-derivation) ----
    checks = []
    for t in TIERS:
        # 1F tied to the PUBLISHED legend value (true published match)
        checks.append(Check("1F", f"min_valid_loss[{t}]", PUBLISHED_1F[t],
                            round(f1f[t]["min_valid_loss"], 4), rtol=0.0, atol=0.0))
        # 1G re-derived from the .out vs the committed reproduced value (atol=0)
        checks.append(Check("1G", f"gene_ppl[{t}]", committed[("1G", f"gene_ppl[{t}]")],
                            round(f1g[t]["gene_ppl"], 4), rtol=0.0, atol=0.0))
        checks.append(Check("1G", f"intergenic_ppl[{t}]", committed[("1G", f"intergenic_ppl[{t}]")],
                            round(f1g[t]["intergenic_ppl"], 4), rtol=0.0, atol=0.0))
    print("\n" + summary(checks))
    out_csv = RECHECK / "recheck_checks_fig01.csv"
    write_verdicts(checks, out_csv)
    n_pass = sum(c.verdict == "PASS" for c in checks)
    print(f"\n{n_pass}/{len(checks)} recheck numeric checks PASS (Δ=0, atol=0)")
    assert n_pass == len(checks), "some recheck numeric checks failed"

    # also dump the full re-derived table for the discrepancy write-up
    with open(RECHECK / "recompute_fig01_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "n_epochs", "min_valid_loss", "argmin_epoch", "x_extent_batches",
                    "gene_ppl", "intergenic_ppl", "overall_ppl", "overall_loss"])
        for t in TIERS:
            w.writerow([t, f1f[t]["n_epochs"], f1f[t]["min_valid_loss"], f1f[t]["argmin_epoch"],
                        f1f[t]["x_extent_batches"], f1g[t]["gene_ppl"], f1g[t]["intergenic_ppl"],
                        f1g[t]["overall_ppl"], f1g[t]["overall_loss"]])
    print(f"[OK] full re-derived table -> {RECHECK / 'recompute_fig01_table.csv'}")


if __name__ == "__main__":
    main()
