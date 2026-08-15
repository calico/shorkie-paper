# src/shorkie/

The installable helper package (`pip install -e .`). It is a thin layer over
`external/baskerville-yeast`, and exists so model loading, sequence preparation and variant scoring have
exactly **one** implementation — the same one used by `examples/`, `minimal_example/`, and the paper's
own eQTL/MPRA scorers.

| Module | Provides |
|--------|----------|
| `config` | Path resolution (`load`, `get`, `path`, `repo_root`). See [`../../config/README.md`](../../config/README.md). |
| `models.ensemble` | `load_ensemble`, `make_input`, `ensemble_predict`, `logSED`, `logSED_per_track`, and the input-layout constants. |
| `helpers.yeast_helpers` | Sequence/coverage utilities, ISM and gradient attribution, sequence-logo and coverage plotting. |
| `viz.load_cov` | bigWig/BED/HDF5 coverage I/O (`CovFace`, `read_coverage`, `seq_norm`). |
| `data` | BED helpers and job-submission utilities used by the data-build stage. |

Submodules are not auto-imported — import them explicitly:

```python
from shorkie.models.ensemble import load_ensemble, make_input, logSED
```

**Input layout:** models take `(16384, 170)` — channels 0–3 are DNA one-hot, 4–169 are species identity
(column 114 = *S. cerevisiae*). `make_input` builds this for you; hand-rolling a plain 4-channel
one-hot is the most common cause of load/prediction failures.

Full API reference: <https://khchao.com/shorkie/content/api.html>
