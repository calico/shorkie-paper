#!/usr/bin/env python3
"""Numeric verification helpers for the figure reproduction audit.

A panel's reproduction is "verified" when the value we recompute matches the
value reported in the manuscript (text/caption/table) within tolerance. These
helpers build a tidy reproduced-vs-reported table and emit a per-figure verdict
CSV (``figure_NN/reproduced/verify_figNN.csv``).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import csv
import math


@dataclass
class Check:
    panel: str          # e.g. "1G"
    metric: str         # e.g. "perplexity_gene_165_Saccharomycetales"
    reported: float     # value stated in the manuscript
    reproduced: float   # value we recomputed
    rtol: float = 0.02  # relative tolerance (2% default)
    atol: float = 0.0   # absolute tolerance

    @property
    def delta(self) -> float:
        return self.reproduced - self.reported

    @property
    def verdict(self) -> str:
        if self.reported is None or self.reproduced is None:
            return "MISSING"
        if any(map(lambda v: v is not None and math.isnan(v), [self.reported, self.reproduced])):
            return "MISSING"
        tol = self.atol + self.rtol * abs(self.reported)
        return "PASS" if abs(self.delta) <= tol else "FAIL"


def write_verdicts(checks: list[Check], out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in checks:
        d = asdict(c)
        d["delta"] = c.delta
        d["verdict"] = c.verdict
        rows.append(d)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "metric", "reported",
                                          "reproduced", "delta", "rtol", "atol", "verdict"])
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(1 for c in checks if c.verdict == "PASS")
    print(f"[verify] {n_pass}/{len(checks)} PASS -> {out_csv}")
    return out_csv


def summary(checks: list[Check]) -> str:
    lines = [f"{c.panel:>4}  {c.metric:<48}  reported={c.reported:<10}  "
             f"reproduced={c.reproduced:<10}  Δ={c.delta:+.4g}  {c.verdict}"
             for c in checks]
    return "\n".join(lines)
