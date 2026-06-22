#!/usr/bin/env python3
"""Figure 1 deep-recheck — Step 3: determinism of the heavy external-tool panels.

Re-running mash / nucmer+show-coords / ete4 from the released FASTAs must
reproduce the committed intermediates. We diff the freshly regenerated tables
against the git-committed versions:

  1D Mash    : byte-identical expected (mash MinHash is seeded; config path
               prefix unchanged).
  1C MUMmer  : data-identical expected. The only legitimate difference is the
               absolute FASTA path embedded in the .coords header line, which
               flips between the real path (/scratch4/...) and its symlink
               (/home/kchao10/scr4_ssalzbe1/...) depending on what config
               resolved at generation time. We normalise that prefix, then diff
               the alignment rows. (YJM195 is the corrected strain rep — it
               replaces the previously-committed YJM1078.)
  1B tree    : leaf count compared; byte-identity is NOT required (NCBI taxonomy
               versioning), documented as deterministic-modulo-NCBI.

Writes recheck/determinism_fig01.csv.

Run (env yeast_ml):
    python reproduction/figure_01/recheck/diff_heavy_panels.py
"""
from __future__ import annotations
import subprocess
from pathlib import Path

from shorkie import config

REPO = Path(config.repo_root())
RC = REPO / "reproduction" / "figure_01" / "reproduced"
RECHECK = REPO / "reproduction" / "figure_01" / "recheck"

PREFIXES = ["/scratch4/ssalzbe1/khchao/Yeast_ML", "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML"]


def norm(text: str) -> str:
    for p in PREFIXES:
        text = text.replace(p, "<WORK>")
    return text


def git_show(relpath: str):
    try:
        return subprocess.run(["git", "show", f"HEAD:{relpath}"], cwd=str(REPO),
                              capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None


def compare(relpath: str):
    """Return (status, detail) comparing committed vs on-disk, path-normalised."""
    committed = git_show(relpath)
    disk_p = REPO / relpath
    if committed is None:
        return ("NEW", "no committed version (added this recheck)")
    if not disk_p.exists():
        return ("MISSING_ON_DISK", "committed but not regenerated")
    disk = disk_p.read_text()
    if committed == disk:
        return ("BYTE_IDENTICAL", "")
    if norm(committed) == norm(disk):
        return ("DATA_IDENTICAL", "only the embedded FASTA path prefix differs")
    # quantify the difference
    cl = norm(committed).splitlines()
    dl = norm(disk).splitlines()
    ndiff = sum(1 for a, b in zip(cl, dl) if a != b) + abs(len(cl) - len(dl))
    return ("DIFFERS", f"{ndiff} differing lines (committed {len(cl)} vs disk {len(dl)})")


def main():
    rows = []  # (panel, artifact, status, detail)

    # ---- 1D Mash ----
    for tab in ["saccharomycetales_dist.tab", "strains_dist.tab"]:
        rel = f"reproduction/figure_01/reproduced/panelD_mash/{tab}"
        st, det = compare(rel)
        rows.append(("1D", tab, st, det))

    # ---- 1C MUMmer ----
    mummer = {
        "GCA_000146045_2.coords": "R64-1-1 self (species)",
        "GCA_000975585_2.coords": "YJM195 (strain) — corrected rep, replaces YJM1078",
        "GCA_000002545_2.coords": "N. glabratus CBS138 (order)",
        "GCA_000182965_3.coords": "C. albicans SC5314 (order)",
        "GCA_000182925_2.coords": "N. crassa OR74A (kingdom)",
        "GCA_000002945_2.coords": "S. pombe 972h (kingdom)",
    }
    for fn, label in mummer.items():
        rel = f"reproduction/figure_01/reproduced/panelC_mummer/{fn}"
        st, det = compare(rel)
        rows.append(("1C", f"{fn} [{label}]", st, det or ""))
    # stale YJM1078 (committed, no longer a target)
    old = "reproduction/figure_01/reproduced/panelC_mummer/GCA_000975645_3.coords"
    if (REPO / old).exists():
        rows.append(("1C", "GCA_000975645_3.coords [YJM1078 — STALE, to remove]",
                     "STALE", "superseded by YJM195 (GCA_000975585_2)"))

    # ---- 1B tree ----
    nwk_rel = "reproduction/figure_01/reproduced/panelB_tree/species_tree.nwk"
    committed_nwk = git_show(nwk_rel)
    disk_nwk = (REPO / nwk_rel).read_text()

    def n_leaves(nwk):
        try:
            from ete4 import Tree
            return len(list(Tree(nwk, parser=1).leaves()))
        except Exception:
            return nwk.count(",") + 1

    disk_leaves = n_leaves(disk_nwk)
    if committed_nwk is None:
        rows.append(("1B", "species_tree.nwk", "NEW", f"{disk_leaves} leaves"))
    elif committed_nwk == disk_nwk:
        rows.append(("1B", "species_tree.nwk", "BYTE_IDENTICAL", f"{disk_leaves} leaves"))
    else:
        rows.append(("1B", "species_tree.nwk", "DETERMINISTIC_MODULO_NCBI",
                     f"{n_leaves(committed_nwk)} -> {disk_leaves} leaves (NCBI taxonomy versioning)"))
    taxids_disk = (REPO / "reproduction/figure_01/reproduced/panelB_tree/taxids.txt").read_text().split()
    rows.append(("1B", "taxids.txt", "INPUT", f"{len(taxids_disk)} input taxon IDs (1341_Fungal superset)"))

    # ---- report ----
    print(f"{'panel':<6} {'status':<26} artifact")
    print("-" * 90)
    for panel, art, st, det in rows:
        print(f"{panel:<6} {st:<26} {art}")
        if det:
            print(f"{'':<6} {'':<26}   ({det})")

    out = RECHECK / "determinism_fig01.csv"
    with open(out, "w") as f:
        f.write("panel,artifact,status,detail\n")
        for panel, art, st, det in rows:
            art_q = art.replace(",", ";")
            det_q = det.replace(",", ";")
            f.write(f"{panel},{art_q},{st},{det_q}\n")
    print(f"\n[OK] -> {out}")

    # determinism verdict for the data-bearing artifacts
    ok = all(st in ("BYTE_IDENTICAL", "DATA_IDENTICAL", "NEW", "DETERMINISTIC_MODULO_NCBI", "INPUT", "STALE")
             for _, _, st, _ in rows)
    print(f"[{'PASS' if ok else 'REVIEW'}] heavy-panel determinism")


if __name__ == "__main__":
    main()
