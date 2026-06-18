"""shorkie — shared utilities and path configuration for the Shorkie paper repo.

This installable package replaces the previously copy-pasted helper modules
(``yeast_helpers.py`` x9, ``util.py`` x3, ``bed_helper.py`` x2, ``load_cov.py``)
and the hardcoded filesystem paths. Install with ``pip install -e .`` from the
repository root.

Submodules:
  - ``shorkie.config``         central path/config loader (see config/paths.yaml)
  - ``shorkie.helpers``        sequence / coverage / plotting helpers
  - ``shorkie.data``           dataset-build helpers (BED, parallel I/O)
  - ``shorkie.viz``            visualization helpers
  - ``shorkie.models``         model ensemble loading + scoring
"""
from . import config  # noqa: F401

__all__ = ["config"]
__version__ = "0.1.0"
