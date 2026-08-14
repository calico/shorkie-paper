"""Release-integrity tests — the manifest, the download script, the training configs,
and the docs links. Fast, offline, no model weights and no GPU.

    pytest -q tests/

Every test here pins a defect that actually shipped at some point, so they are
regression guards rather than generic sanity checks. The one test that needs
network/credentials is marked `bucket` and skipped unless you opt in:

    pytest -q -m bucket --run-bucket -u-less   # see test_released_params_match_committed
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO / "data" / "manifest.json").read_text())

# Either a bucket root (buckets.*.gcs) or a bucket + object/prefix path.
GS_URI_RE = re.compile(r"^gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9](/.*)?$")


def _iter_gs_uris(node):
    """Yield every gs_uri / gs:// string anywhere in the manifest tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and v.startswith("gs://"):
                yield k, v
            else:
                yield from _iter_gs_uris(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_gs_uris(v)


# ── manifest ────────────────────────────────────────────────────────────────

def test_manifest_is_valid_and_has_expected_sections():
    assert MANIFEST["schema_version"] == 1
    assert {"models", "datasets", "buckets"} <= set(MANIFEST)
    assert {"shorkie_lm", "shorkie_finetuned", "shorkie_random_init"} <= set(MANIFEST["models"])
    # the genome is what examples/ and minimal_example/ need in order to run at all
    assert "genome" in MANIFEST["datasets"], "datasets.genome missing — examples cannot run"


def test_every_gs_uri_is_well_formed():
    uris = list(_iter_gs_uris(MANIFEST))
    assert uris, "no gs:// URIs found in the manifest"
    bad = [(k, v) for k, v in uris if not GS_URI_RE.match(v)]
    assert not bad, f"malformed gs:// URIs: {bad}"


def test_model_uris_live_under_the_shorkie_models_prefix():
    """The model bucket was reorganised under shorkie_models/; nothing may point above it."""
    for name, m in MANIFEST["models"].items():
        for f in m.get("files", []):
            assert "/shorkie_models/" in f["gs_uri"], f"{name}: stale path {f['gs_uri']}"
            if "https_uri" in f:
                assert "/shorkie_models/" in f["https_uri"], f"{name}: stale {f['https_uri']}"


def test_released_models_are_not_marked_pending():
    for name, m in MANIFEST["models"].items():
        if m.get("files"):
            assert not m.get("pending_upload"), f"{name} has files but is marked pending_upload"


def test_genome_entries_are_complete():
    g = MANIFEST["datasets"]["genome"]
    for key in ("fasta", "gtf"):
        entry = g[key]
        assert entry["gs_uri"].startswith("gs://")
        assert entry["local_name"], f"{key} needs a local_name for download.sh to save under"
        assert entry["size_bytes"] > 0
        assert re.fullmatch(r"[0-9a-f]{32}", entry["md5"]), f"{key} md5 not pinned"
    # download.sh saves the FASTA under the name config genome.fasta expects
    assert g["fasta"]["local_name"] == "GCA_000146045_2.cleaned.fasta"
    assert g["gtf"]["local_name"] == "GCA_000146045_2.59.gtf"


# ── download.sh ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("args", [
    ["--models", "all"], ["--models", "lm"], ["--models", "finetuned"],
    ["--models", "random_init"], ["--minimal"], ["--genome"],
    ["--eqtl"], ["--mpra", "all"], ["--supervised", "all"],
    ["--lm-corpus", "165_Saccharomycetales"],
])
def test_download_dry_run_emits_manifest_uris(args):
    """Every mode must dry-run cleanly and only reference URIs the manifest declares."""
    r = subprocess.run(["bash", str(REPO / "data" / "download.sh"), *args, "--dry-run"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"{args} failed:\n{r.stdout}\n{r.stderr}"
    known = {v for _, v in _iter_gs_uris(MANIFEST)}
    emitted = re.findall(r"gs://\S+", r.stdout)
    assert emitted, f"{args} emitted no gs:// URIs"
    for uri in emitted:
        # dataset prefixes are copied recursively, so a declared prefix covers its children
        assert any(uri == k or uri.startswith(k) or k.startswith(uri) for k in known), \
            f"{args} emitted undeclared URI {uri}"


def test_download_genome_saves_under_config_expected_names():
    r = subprocess.run(["bash", str(REPO / "data" / "download.sh"), "--genome", "--dry-run"],
                       capture_output=True, text=True, cwd=REPO)
    assert "GCA_000146045_2.cleaned.fasta" in r.stdout
    assert "GCA_000146045_2.59.gtf" in r.stdout


def test_download_rejects_unknown_mode():
    r = subprocess.run(["bash", str(REPO / "data" / "download.sh"), "--nonsense"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0


# ── training configs ────────────────────────────────────────────────────────
# The repo once shipped shorkie_scratch/params.json at lr 1e-4 while the RELEASED
# Shorkie_Random_Init is lr 5e-4 — i.e. the committed recipe did not reproduce the
# published ablation. These pin the published hyperparameters.

def _params(rel):
    return json.loads((REPO / "scripts" / "02_train" / rel / "params.json").read_text())


def test_committed_training_configs_match_published_recipes():
    lm, ft, ri = _params("shorkie_lm"), _params("shorkie_finetuned"), _params("shorkie_scratch")
    assert lm["train"]["task"] == "self-supervised" and lm["train"]["loss"] == "mlm"
    assert lm["train"]["learning_rate"] == 1e-4
    assert ft["train"]["task"] == "fine-tune"
    assert ft["train"]["learning_rate"] == 2e-5
    assert ri["train"]["task"] == "supervised"
    assert ri["train"]["learning_rate"] == 5e-4, (
        "Shorkie_Random_Init is released at lr 5e-4; a different value here means the "
        "committed config no longer reproduces the published ablation")


def test_finetuned_and_random_init_differ_only_in_task_and_lr():
    ft, ri = _params("shorkie_finetuned"), _params("shorkie_scratch")
    assert ft["model"] == ri["model"], "the ablation must hold the architecture fixed"
    differing = {k for k in set(ft["train"]) | set(ri["train"])
                 if ft["train"].get(k) != ri["train"].get(k)}
    assert differing == {"task", "learning_rate"}, f"unexpected train-block drift: {differing}"


@pytest.mark.bucket
def test_released_params_match_committed():
    """Opt-in: compare the committed configs against the live bucket copies.

    Needs gsutil/gcloud credentials, so it is deselected by default
    (`-m "not bucket"` in pyproject). Run explicitly with `pytest -m bucket`.
    """
    import shutil
    gs = shutil.which("gcloud")
    if not gs:
        pytest.skip("gcloud not available")
    pairs = [("shorkie_lm", "shorkie_lm"), ("shorkie", "shorkie_finetuned"),
             ("shorkie_random_init", "shorkie_scratch")]
    for remote, local in pairs:
        r = subprocess.run(
            [gs, "storage", "cat", f"gs://seqnn-share/shorkie_models/{remote}/params.json"],
            capture_output=True, text=True)
        assert r.returncode == 0, f"could not read released params for {remote}: {r.stderr}"
        assert json.loads(r.stdout) == _params(local), \
            f"committed scripts/02_train/{local}/params.json != released {remote}/params.json"


# ── config <-> manifest agreement ───────────────────────────────────────────
# models.shorkie_finetuned used to point at an author work-dir run whose weights
# are NOT the released ones (f0 md5 5ca26080... vs released 23e79b73...), so the
# example config silently resolved to a different model than users download.

def test_config_model_dirs_match_download_layout():
    from shorkie import config
    cfg = config.load(REPO / "config" / "paths.example.yaml")
    release_root = str(cfg.path("release_root"))
    for key, manifest_key in [("models.shorkie_lm", "shorkie_lm"),
                              ("models.shorkie_finetuned", "shorkie_finetuned"),
                              ("models.shorkie_random_init", "shorkie_random_init")]:
        resolved = str(cfg.path(key))
        assert resolved.startswith(release_root), (
            f"{key} resolves to {resolved!r}, outside release_root {release_root!r} — a fresh "
            f"clone would not get the released weights that data/download.sh fetches")
        # the directory download.sh writes into, derived from the manifest local_path
        local = MANIFEST["models"][manifest_key]["files"][0]["local_path"]  # models/<name>/...
        expected_dir = local.split("/")[1]
        assert Path(resolved).name == expected_dir, (
            f"{key} -> {Path(resolved).name!r} but download.sh writes to models/{expected_dir}/")


# ── minimal_example CLI (the defaults that used to resolve to '/params.json') ──

def test_minimal_example_defaults_resolve_to_real_files():
    code = (
        "import importlib.util,sys,os,json;"
        f"spec=importlib.util.spec_from_file_location('rsv',r'{REPO}/minimal_example/run_shorkie_variant.py');"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "print(json.dumps({'params':m.DEFAULT_PARAMS,'targets':m.DEFAULT_TARGETS}))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout.strip().splitlines()[-1])
    for key, val in d.items():
        assert val and not val.startswith("/params.json") and not val.startswith("/sheet.txt"), \
            f"{key} default is the old placeholder-derived path: {val!r}"
        assert Path(val).exists(), f"{key} default does not exist: {val}"


def test_minimal_example_ships_its_default_resources():
    for name in ("params.json", "sheet.txt"):
        assert (REPO / "minimal_example" / name).exists()


# ── documentation links ─────────────────────────────────────────────────────

def _tracked_files():
    r = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=REPO)
    return set(r.stdout.split())


LINK_RE = re.compile(r"\[[^\]]*\]\((?!https?:|mailto:|#)([^)#]+)")


def test_markdown_links_point_at_git_tracked_paths():
    """A link that exists on disk but is gitignored still 404s on GitHub.

    The 7 figure READMEs used to link into the gitignored paper/ directory, which
    resolved locally and 404'd for everyone else — so existence on disk is not enough.
    """
    tracked = _tracked_files()
    broken = []
    for md in sorted(REPO.glob("**/*.md")):
        rel = md.relative_to(REPO)
        if str(rel).startswith(("external/", "data_local/", "my_shorkie/", "paper/")):
            continue
        if str(rel) not in tracked:
            continue
        for target in LINK_RE.findall(md.read_text()):
            target = target.strip()
            if not target or target.startswith("<"):
                continue
            resolved = (md.parent / target).resolve()
            try:
                rel_target = resolved.relative_to(REPO)
            except ValueError:
                continue                      # points outside the repo; not our problem
            if not resolved.exists():
                broken.append(f"{rel} -> {target} (missing)")
            elif resolved.is_file() and str(rel_target) not in tracked:
                broken.append(f"{rel} -> {target} (exists locally but is NOT tracked; 404s on GitHub)")
    assert not broken, "broken markdown links:\n  " + "\n  ".join(broken)


def test_no_script_invokes_a_python_file_that_does_not_exist():
    """Several shell scripts used to `python some_file.py` a file that exists nowhere.

    Scripts may legitimately `cd` into a sibling directory first (run_pipeline.sh does),
    so a name is only broken if it exists nowhere in the repo. The one helper that is
    genuinely absent from this release is allowed *if and only if* its caller guards for
    it and exits with an explanatory message rather than failing cryptically.
    """
    tracked = _tracked_files()
    basenames = {Path(p).name for p in tracked}
    missing = []
    for sh in sorted(REPO.glob("scripts/**/*.sh")):
        rel = sh.relative_to(REPO)
        if str(rel) not in tracked:
            continue
        text = sh.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "echo " in stripped.split("python")[0]:
                continue
            m = re.search(r"(?<![\w/])python3?\s+([\w.\-]+\.py)\b", stripped)
            if not m:
                continue
            name = m.group(1)
            if name in basenames:
                continue
            guarded = re.search(rf'if \[ ! -f "{re.escape(name)}" \]', text)
            if not guarded:
                missing.append(f"{rel}: {name}")
    assert not missing, ("scripts invoking .py files that exist nowhere (and are not guarded):\n  "
                         + "\n  ".join(missing))
