#!/usr/bin/env python3
"""Verify that everything data/manifest.json catalogs is actually present on the
GCS buckets, with matching sizes (and md5 where pinned).

Run BEFORE upload to see the gaps, and AFTER `upload_release.sh` to confirm the
release is complete. Model bucket (gs://seqnn-share) is public; the data bucket
(gs://shorkie-paper) is requester-pays — pass `-u PROJECT`.

    python scripts/00_setup/verify_release.py [-u PROJECT] [--what models|data|all] [--strict]

Artifacts flagged `pending_upload` in the manifest (catalogued but not yet uploaded — e.g.
shorkie_random_init / eqtl / mpra before the maintainer runs upload_release.sh) are reported as
`[PEND]`, not failures. Exit code 0 iff every *live* artifact is present (size/md5 match where
pinned); `--strict` additionally requires the pending ones to be uploaded (run it post-upload).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifest.json"
GSUTIL = shutil.which("gsutil") or "/home/kchao10/data_ssalzbe1/khchao/google-cloud-sdk/bin/gsutil"


def gsutil(args, project=None):
    cmd = [GSUTIL] + (["-u", project] if project else []) + args
    return subprocess.run(cmd, capture_output=True, text=True)


def stat_object(uri, project=None):
    """Return (exists, size_bytes, md5_hex_or_None) for a single gs:// object."""
    r = gsutil(["stat", uri], project=project)
    if r.returncode != 0:
        return False, None, None
    size = md5 = None
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("Content-Length:"):
            size = int(s.split()[-1])
        elif s.startswith("Hash (md5):"):
            import base64
            md5 = base64.b64decode(s.split()[-1]).hex()
    return True, size, md5


def prefix_nonempty(uri, project=None):
    r = gsutil(["ls", uri.rstrip("/") + "/**"], project=project)
    n = len([x for x in r.stdout.splitlines() if x.strip()])
    return r.returncode == 0 and n > 0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--project", default=None, help="billing project for the requester-pays data bucket")
    ap.add_argument("--what", choices=["models", "data", "all"], default="all")
    ap.add_argument("--strict", action="store_true",
                    help="also fail if pending_upload artifacts are not yet on the bucket "
                         "(use after running upload_release.sh)")
    opts = ap.parse_args()

    man = json.load(open(MANIFEST))
    fails, pending = [], []   # pending = missing items belonging to a pending_upload entry (expected pre-upload)
    print(f"== verifying manifest against buckets (gsutil={GSUTIL}) ==")

    def record(uri, ok, is_pending):
        if ok:
            return "OK "
        (pending if is_pending else fails).append(uri)
        return "PEND" if is_pending else "FAIL"

    if opts.what in ("models", "all"):
        for name, m in man["models"].items():
            if not m.get("files"):
                print(f"[skip] models.{name}: no files (e.g. deprecated alias)"); continue
            pend = bool(m.get("pending_upload"))
            for f in m["files"]:
                ok, size, md5 = stat_object(f["gs_uri"])
                exp_sz, exp_md5 = f.get("size_bytes"), f.get("md5")
                good = ok and not (exp_sz and size != exp_sz) and not (exp_md5 and md5 and md5 != exp_md5)
                tag = record(f["gs_uri"], good, pend)
                print(f"  [{tag}] {f['gs_uri']}  size={size}{'' if not exp_sz else f'/{exp_sz}'}"
                      f"{'' if not exp_md5 else f'  md5={md5}'}")

    if opts.what in ("data", "all"):
        if not opts.project:
            print("[warn] data bucket is requester-pays; pass -u PROJECT to check it", file=sys.stderr)
        for name, d in man["datasets"].items():
            pend = bool(d.get("pending_upload"))
            uris = [v["gs_uri"] for v in d.values() if isinstance(v, dict) and "gs_uri" in v]
            uris += [t[k] for t in d.get("tiers", {}).values() for k in ("genome", "tfrecords") if k in t]
            for uri in uris:
                if not opts.project:
                    print(f"  [skip] {uri} (need -u PROJECT)"); continue
                ok, n = prefix_nonempty(uri, project=opts.project) if uri.endswith("/") \
                    else (stat_object(uri, project=opts.project)[0], 1)
                tag = record(uri, ok, pend)
                print(f"  [{tag}] datasets.{name}: {uri}" + (f"  ({n} objs)" if uri.endswith('/') else ""))

    print(f"\n== live: {'ALL PRESENT' if not fails else f'{len(fails)} MISSING/MISMATCH'}"
          f"{'' if not pending else f' | pending upload (not yet on bucket): {len(pending)}'} ==")
    for u in fails:
        print(f"  MISSING (live): {u}")
    for u in pending:
        print(f"  pending: {u}  (publish with scripts/00_setup/upload_release.sh)")
    # exit 0 if all *live* artifacts present; --strict also requires the pending ones uploaded
    sys.exit(1 if fails or (opts.strict and pending) else 0)


if __name__ == "__main__":
    main()
