#!/bin/bash
# Figure 1B — phylogeny of the 4 datasets. Full recompute of the NCBI-taxonomy
# species tree from the committed taxon IDs, via ete3/ete4 ncbiquery.
# Faithful to scripts/04_analysis/others/phylogenetic_tree/{1_get_taxo_id.py, 2_plot_tree.sh}.
# The published circular tree is additionally styled in iTOL (web) — this step
# reproduces the underlying newick topology; the notebook renders + highlights it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root
OUT="reproduction/figure_01/reproduced/panelB_tree"
mkdir -p "$OUT"

# Taxon IDs: the 1341_Fungal list is the superset (contains all 4 datasets).
SPECIES_CSV="data/species_lists/species_fungi_1385_gtf.cleaned.csv"

# Extract the "Taxon ID" column -> taxids.txt (python handles the CSV header robustly).
python - "$SPECIES_CSV" "$OUT/taxids.txt" <<'PY'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
ids=[]
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        t = row.get("Taxon ID") or row.get("taxon_id")
        if t and t.strip().isdigit():
            ids.append(t.strip())
with open(dst, "w") as o:
    o.write("\n".join(ids) + "\n")
print(f"[OK] {len(ids)} taxon IDs -> {dst}")
PY

NWK="$OUT/species_tree.nwk"
echo "[tree] building NCBI-taxonomy topology for $(wc -l < "$OUT/taxids.txt") taxa"
if command -v ete4 >/dev/null 2>&1; then
  cat "$OUT/taxids.txt" | ete4 ncbiquery --tree > "$NWK" && echo "[OK] ete4 -> $NWK"
elif command -v ete3 >/dev/null 2>&1; then
  cat "$OUT/taxids.txt" | ete3 ncbiquery --tree > "$NWK" 2>/dev/null && echo "[OK] ete3 -> $NWK" || echo "[warn] ete3 CLI failed; notebook builds via NCBITaxa API"
else
  echo "[warn] no ete CLI; the notebook builds the tree via the ete3.NCBITaxa Python API"
fi
echo "[done] panel B in $OUT (newick built if ete + NCBI taxa DB available)"
