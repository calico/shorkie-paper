# tests/

Fast, offline checks. No model weights, no GPU, no network.

```bash
pytest -q                 # everything below except the bucket tests
pytest -q -m bucket -o addopts=""   # additionally hit GCS (needs credentials)
```

| File | Covers |
|------|--------|
| `test_smoke.py` | The package imports, config resolves, `logSED` math is numerically right, and `make_input` builds a correct `(16384, 170)` tensor from a synthetic FASTA. |
| `test_release.py` | Release integrity: manifest coherence, `download.sh` dry-runs for every mode, committed-vs-released training configs, `minimal_example` CLI defaults, submodule URLs, CITATION.cff consistency, and documentation links. |

Most of `test_release.py` exists to pin a defect that actually shipped once, so a failure usually means
a real regression rather than a flaky check. Two examples: the markdown-link test requires targets be
**git-tracked** (existing on disk is not enough — that is how links into the gitignored `paper/` came to
404 on GitHub), and the training-config test pins Shorkie_Random_Init at lr 5e-4 because the repo once
shipped 1e-4, which did not reproduce the released model.

`bucket`-marked tests are deselected by default via `addopts` in `pyproject.toml`.
