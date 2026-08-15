#!/usr/bin/env python3
"""Create (or update) the Zenodo deposition mirroring the Shorkie release.

Why this exists: the data bucket `gs://shorkie-paper` is requester-pays, which needs
a billing-enabled Google Cloud project. That is impractical in some regions, so the
models and pretraining corpora are also published on Zenodo — reachable anywhere,
permanent, and citable by DOI.

    export ZENODO_TOKEN=...                      # never pass the token as an argument
    python scripts/00_setup/zenodo_upload.py --dir <packages/> --dry-run
    python scripts/00_setup/zenodo_upload.py --dir <packages/>

**This never publishes.** It leaves the deposition as a draft for you to review in the
browser and publish yourself — a published Zenodo DOI is permanent and cannot be
retracted, only superseded.

Uploads are resumable: files already present on the deposition with a matching
checksum are skipped, so re-running after an interruption costs nothing.

Get a token at https://zenodo.org/account/settings/applications/tokens/new/
with the `deposit:write` and `deposit:actions` scopes. Use --sandbox to rehearse
against https://sandbox.zenodo.org first (separate account and token).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("error: this script needs `requests` (pip install requests)")

PAPER_DOI = "10.1101/2025.09.19.677475"
REPO_URL = "https://github.com/calico/shorkie-paper"
DOCS_URL = "https://khchao.com/shorkie/"

CREATORS = [
    {"name": "Chao, Kuan-Hao"},
    {"name": "Magzoub, Majed M."},
    {"name": "Stoops, Emily H."},
    {"name": "Hackett, Sean R."},
    {"name": "Linder, Johannes"},
    {"name": "Kelley, David R."},
]

DESCRIPTION = f"""
<p>Model weights and pretraining data for <strong>Shorkie</strong>, a sequence-to-expression
model for budding yeast built on a fungal DNA language model.</p>

<p>Companion to the preprint
<a href="https://doi.org/{PAPER_DOI}">Predicting dynamic expression patterns in budding yeast
with a fungal DNA language model</a>.
Code: <a href="{REPO_URL}">{REPO_URL}</a> &middot;
Documentation: <a href="{DOCS_URL}">{DOCS_URL}</a></p>

<p>The canonical copies live in Google Cloud Storage, but the data bucket is requester-pays and
therefore needs a billing-enabled Google Cloud project, which is impractical in some regions.
This deposition mirrors that release so it is usable without Google Cloud. All files are
byte-identical to the bucket originals and verified against the MD5 checksums pinned in
<code>manifest.json</code>.</p>

<h3>Contents</h3>
<ul>
  <li><strong>Models</strong> &mdash; Shorkie_LM (masked DNA language model), Shorkie (8-fold
      supervised ensemble fine-tuned from it), and Shorkie_Random_Init (the from-scratch ablation).</li>
  <li><strong>Pretraining corpora</strong> &mdash; four tiers of increasing phylogenetic breadth
      (R64, 80_strains, 165_Saccharomycetales, 1341_Fungus), each as genome assemblies and matched
      16,384&nbsp;bp ZLIB TFRecords. <strong>Shorkie_LM was pretrained on the
      165_Saccharomycetales tier.</strong></li>
  <li><strong>Reference genome</strong> &mdash; the S.&nbsp;cerevisiae R64 FASTA and Ensembl Fungi
      release-59 GTF that the models and examples expect.</li>
  <li><strong>Species tables</strong> &mdash; per-tier assembly accessions, for provenance.</li>
</ul>

<p>All four corpus tiers share one held-out split, drawn from S.&nbsp;cerevisiae R64 only and split by
whole chromosome (valid: chrXI, chrXIII, chrXV; test: chrXII, chrXIV, chrXVI), with chrXI&ndash;XVI
excluded from training everywhere. Preserving that split matters if you use these corpora as a
baseline.</p>

<p><em>Not included:</em> the 39&nbsp;GB of raw 1341_Fungus genome assemblies. Those are public NCBI
data; every accession is listed in the included species tables, and the code repository rebuilds the
corpus from them. Their derived TFRecords <em>are</em> included.</p>

<p>The genome assemblies are third-party public data (Ensembl Fungi release 59 / NCBI GenBank),
redistributed here for reproducibility. Please cite the preprint and the original data sources; see
the included README for the full attribution list.</p>
""".strip()

METADATA = {
    "title": "Shorkie: model weights and fungal pretraining corpora for yeast expression prediction",
    "upload_type": "dataset",
    "description": DESCRIPTION,
    "creators": CREATORS,
    "license": "cc-by-4.0",
    "access_right": "open",
    "keywords": [
        "genomics", "yeast", "Saccharomyces cerevisiae", "DNA language model",
        "gene expression", "RNA-seq", "variant effect prediction", "deep learning",
        "regulatory genomics", "eQTL", "MPRA",
    ],
    "related_identifiers": [
        {"identifier": PAPER_DOI, "relation": "isSupplementTo", "scheme": "doi"},
        {"identifier": REPO_URL, "relation": "isSupplementTo", "scheme": "url"},
        {"identifier": DOCS_URL, "relation": "isDocumentedBy", "scheme": "url"},
    ],
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of files to upload")
    ap.add_argument("--deposition", help="existing deposition id (resume instead of creating)")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, then stop")
    ap.add_argument("--exclude", nargs="*", default=["benchmarks_eqtl.tar.gz", "benchmarks_mpra.tar.gz"],
                    help="filenames to skip (default: the benchmark tarballs, which are Drive-only)")
    args = ap.parse_args()

    src = Path(args.dir)
    if not src.is_dir():
        sys.exit(f"error: not a directory: {src}")

    files = sorted(p for p in src.iterdir()
                   if p.is_file() and p.name not in args.exclude)
    if not files:
        sys.exit(f"error: no files to upload in {src}")

    total = sum(p.stat().st_size for p in files)
    print(f"== {len(files)} files, {human(total)} total ==")
    for p in files:
        print(f"  {human(p.stat().st_size):>10}  {p.name}")
    if total > 50 * 1024**3:
        print("\nwarning: over Zenodo's default 50 GB per-record limit; request a quota increase.",
              file=sys.stderr)

    if args.dry_run:
        print("\n(dry run — nothing uploaded)")
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("error: set ZENODO_TOKEN in the environment (never pass it as an argument)")

    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    api = f"{base}/api/deposit/depositions"
    auth = {"Authorization": f"Bearer {token}"}

    # --- deposition -------------------------------------------------------
    if args.deposition:
        r = requests.get(f"{api}/{args.deposition}", headers=auth, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"\n== resuming deposition {dep['id']} ==")
    else:
        r = requests.post(api, headers=auth, json={}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"\n== created deposition {dep['id']} ==")

    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]

    # --- metadata ---------------------------------------------------------
    r = requests.put(f"{api}/{dep_id}", headers=auth,
                     json={"metadata": METADATA}, timeout=60)
    if not r.ok:
        sys.exit(f"error: metadata rejected ({r.status_code}): {r.text}")
    print("  metadata set")

    # --- existing files, for resume --------------------------------------
    r = requests.get(f"{api}/{dep_id}/files", headers=auth, timeout=60)
    r.raise_for_status()
    already = {f["filename"]: f.get("checksum", "").replace("md5:", "") for f in r.json()}

    # --- upload -----------------------------------------------------------
    for i, p in enumerate(files, 1):
        digest = md5(p)
        if already.get(p.name) == digest:
            print(f"  [{i}/{len(files)}] {p.name} — already uploaded, skipping")
            continue
        print(f"  [{i}/{len(files)}] {p.name} ({human(p.stat().st_size)}) …", flush=True)
        with open(p, "rb") as fh:
            r = requests.put(f"{bucket}/{p.name}", data=fh, headers=auth, timeout=None)
        if not r.ok:
            sys.exit(f"error: upload failed for {p.name} ({r.status_code}): {r.text}")
        got = r.json().get("checksum", "").replace("md5:", "")
        if got and got != digest:
            sys.exit(f"error: checksum mismatch for {p.name}: local {digest}, Zenodo {got}")
        print(f"      ok, md5 {digest}")

    url = f"{base}/deposit/{dep_id}"
    print(f"\n== draft ready ==\n{url}")
    print("\nReview it there, then press Publish yourself. Publishing mints a permanent DOI that")
    print("cannot be retracted, so this script deliberately stops here.")
    print(f"\nTo resume after an interruption:\n  python {sys.argv[0]} --dir {src} --deposition {dep_id}")


if __name__ == "__main__":
    main()
