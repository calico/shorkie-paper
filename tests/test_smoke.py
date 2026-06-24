"""Smoke tests for the installable `shorkie` package — fast, no model weights, no GPU.

    pytest -q tests/            # or: scripts/00_setup/verify_install.sh

Checks that the package imports, config loads, the variant-effect math is correct, and the
(16384, 170) model input is built correctly from a tiny synthetic FASTA. Does NOT touch the
released checkpoints (those are exercised by examples/ + minimal_example/).
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def test_import_package():
    import shorkie
    from shorkie import config
    from shorkie.models import ensemble  # noqa: F401
    assert hasattr(config, "load") and hasattr(config, "path") and hasattr(config, "repo_root")


def test_config_loads_and_resolves_models():
    from shorkie import config
    config.load()                       # loads paths.yaml or paths.example.yaml
    assert str(config.path("work_root"))
    # the released model keys must all resolve (incl. the lr=5e-4 Random_Init added at release)
    for key in ("models.shorkie_lm", "models.shorkie_finetuned", "models.shorkie_random_init"):
        assert str(config.path(key)), f"{key} did not resolve"


def test_ensemble_constants_and_logsed_math():
    from shorkie.models import ensemble as E
    assert E.NUM_FEATURES == 170 and E.N_DNA == 4 and E.SCEREVISIAE_COL == 114
    # logSED = log2(Σ_alt+1) − log2(Σ_ref+1) over gene-body bins, averaged over tracks.
    bins, tracks = 10, 3
    y_ref = np.ones((1, 1, bins, tracks), dtype="float32")
    y_alt = np.ones((1, 1, bins, tracks), dtype="float32") * 2.0
    gene_slice = slice(0, bins)
    s = E.logSED(y_ref, y_alt, gene_slice)
    exp = float(np.log2(2.0 * bins + 1) - np.log2(1.0 * bins + 1))
    assert abs(s - exp) < 1e-6
    per = E.logSED_per_track(y_ref, y_alt, gene_slice)
    assert per.shape == (tracks,) and np.allclose(per, exp)


def test_make_input_shape_from_synthetic_fasta(tmp_path):
    pysam = pytest.importorskip("pysam")
    from shorkie.models.ensemble import make_input, NUM_FEATURES, SCEREVISIAE_COL, N_DNA
    fa = tmp_path / "tiny.fa"
    fa.write_text(">chrI\n" + "ACGT" * 100 + "\n")
    pysam.faidx(str(fa))
    fo = pysam.Fastafile(str(fa))
    x = make_input(fo, "chrI", 100, 300, seq_len=16384, mask_pos=8192)
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    assert arr.shape == (16384, NUM_FEATURES)
    assert np.allclose(arr[:, SCEREVISIAE_COL], 1.0)          # species channel set
    assert np.allclose(arr[8192, :N_DNA], 0.0)                # masked position zeroed


def test_minimal_example_cli_help():
    """The minimal_example CLI imports + parses args (no model load)."""
    r = subprocess.run([sys.executable, "minimal_example/run_shorkie_variant.py", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "logSED" in (r.stdout + r.stderr)
