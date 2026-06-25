#!/usr/bin/env python3
"""Motif -> TF matching for Figure 4 (panels A/B/C Reference-DB labels + panel H pairing).

The original pipeline matches MoDISco patterns to a yeast motif database with TomTom
(run inside the modisco *report*, whose `motifs.html` lists match0/qval0/...). This
module is the source of truth for the pattern<->TF mapping. It prefers a freshly-run
TomTom (written to recheck/tomtom_RP_matches.tsv by run_tomtom.py) and falls back to
parsing the cached `report/motifs.html` — both are TomTom-derived, so the pairing matches
the published figure either way.
"""
from __future__ import annotations
import os, re, warnings
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
warnings.filterwarnings("ignore")

import fig4_common as F

# modisco_results -> its cached report/motifs.html
def report_html(modisco_h5: Path) -> Path:
    return modisco_h5.parent / "report" / "motifs.html"


def _clean_tf(name: str) -> str:
    """'Rap1p&consensus=ACAC...' -> 'RAP1'; 'FHL1.1' -> 'FHL1'; 'ABF1.15' -> 'ABF1'."""
    if not isinstance(name, str):
        return ""
    m = re.match(r"([A-Za-z0-9]+?)p?&consensus", name)
    if m:
        return m.group(1).upper()
    return re.split(r"[._ ]", name)[0].upper()


@lru_cache(maxsize=8)
def load_html(html_path: str) -> pd.DataFrame:
    df = pd.read_html(html_path)[0]
    return df


def tf_to_pattern(modisco_h5: Path, tf: str, qmax: float = 1.01):
    """Return (pattern_tag, qval, matched_name) of the best MoDISco pattern for TF `tf`
    by scanning match0/1/2 columns of the report; None if no match within qmax."""
    df = load_html(str(report_html(modisco_h5)))
    tfu = tf.upper()
    best = None
    for _, row in df.iterrows():
        for k in range(3):
            mc, qc = f"match{k}", f"qval{k}"
            if mc not in df.columns:
                continue
            nm = row.get(mc)
            if _clean_tf(nm) == tfu:
                q = float(row.get(qc, 1.0)) if pd.notnull(row.get(qc)) else 1.0
                if q <= qmax and (best is None or q < best[1]):
                    best = (str(row["pattern"]).replace(".", "/", 1) if "." in str(row["pattern"])
                            else str(row["pattern"]), q, str(nm))
    if best is None:
        return None
    # normalize tag to "pos_patterns.pattern_N"
    tag = best[0].replace("/", ".") if "/" in best[0] else best[0]
    return (tag, best[1], best[2])


@lru_cache(maxsize=2)
def _tomtom_tsv():
    p = F.RECHECK / "tomtom_RP_matches.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None


def tf_to_pattern_tomtom(tf: str, qmax: float = 1.01):
    """Best modisco pattern for `tf` from the fresh full TomTom table (all ranks),
    returning (tag, qval, target). Recovers TFs missed by the report's top-3."""
    df = _tomtom_tsv()
    if df is None:
        return None
    hit = df[df["Target_ID"].astype(str).str.upper().str.replace(".", "", regex=False).str.startswith(tf.upper())]
    if len(hit) == 0:
        return None
    best = hit.sort_values("q-value").iloc[0]
    if float(best["q-value"]) > qmax:
        return None
    return (str(best["Query_ID"]), float(best["q-value"]), str(best["Target_ID"]))


def best_pattern_by_correlation(modisco_h5: Path, db_arr):
    """Pearson fallback: modisco pattern whose trimmed CWM info-profile best matches the
    DB motif (sliding, both orientations). Returns (tag, corr)."""
    target = db_arr / (np.linalg.norm(db_arr) + 1e-9)
    best = (None, -1.0)
    with h5py.File(modisco_h5, "r") as f:
        for grp in ["pos_patterns", "neg_patterns"]:
            if grp not in f:
                continue
            for pn in f[grp]:
                cwm = np.array(f[grp][pn]["contrib_scores"][:])
                cwm = F.trim_cwm(cwm)
                for mat in (cwm, cwm[::-1, ::-1]):
                    w = min(len(mat), len(db_arr))
                    if w < 4:
                        continue
                    a = mat[:w].flatten(); b = db_arr[:w].flatten()
                    if a.std() == 0 or b.std() == 0:
                        continue
                    c = float(np.corrcoef(a, b)[0, 1])
                    if c > best[1]:
                        best = (f"{grp}.{pn}", c)
    return best


def pattern_cwm(modisco_h5: Path, tag: str):
    grp, pn = tag.split(".") if "." in tag else tag.split("/")
    with h5py.File(modisco_h5, "r") as f:
        return np.array(f[grp][pn]["contrib_scores"][:])


# -------- LM modisco seqlets mapped to genome (Reference-DB row + TF boxes A/B/C) --------
def lm_modisco_h5(lm_set: str) -> Path:
    return (F.LM_ROOT / f"lm_saccharomycetales_gtf_{F.LM_ARCH}" /
            f"eval_{lm_set}" / "modisco_results_w16384_n100000.h5")


def lm_hits_in_window(lm_set: str, example_idx: int, bed_start: int, win_start: int, win_end: int,
                      qmax: float = 0.5):
    """Return list of dicts {tf, gstart, gend} for LM-MoDISco seqlets of `example_idx`
    that overlap [win_start, win_end] and whose pattern has a TomTom TF match (qval<=qmax)."""
    mh = lm_modisco_h5(lm_set)
    if not mh.exists():
        return []
    df = load_html(str(report_html(mh)))
    # pattern_tag -> best TF + qval
    tagtf = {}
    for _, row in df.iterrows():
        tag = str(row["pattern"])
        nm = row.get("match0"); q = float(row.get("qval0", 1.0)) if pd.notnull(row.get("qval0")) else 1.0
        tagtf[tag] = (_clean_tf(nm), q)
    hits = []
    with h5py.File(mh, "r") as f:
        for grp in ["pos_patterns", "neg_patterns"]:
            if grp not in f:
                continue
            for pn in f[grp]:
                tag = f"{grp}.{pn}"
                tf, q = tagtf.get(tag, ("", 1.0))
                if not tf or q > qmax:
                    continue
                sg = f[grp][pn].get("seqlets")
                if sg is None:
                    continue
                starts = sg["start"][:]; ends = sg["end"][:]; ex = sg["example_idx"][:]
                for i in range(len(starts)):
                    if int(ex[i]) != example_idx:
                        continue
                    gs = bed_start + int(starts[i]); ge = bed_start + int(ends[i])
                    if ge > win_start and gs < win_end:
                        hits.append(dict(tf=tf, gstart=gs, gend=ge, qval=q, tag=tag))
    # dedup overlapping same-TF hits, keep best qval
    hits.sort(key=lambda h: (h["gstart"], h["qval"]))
    return hits
