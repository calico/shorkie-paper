"""Central path / configuration for shorkie-paper.

Loads a YAML config (default ``config/paths.yaml``, falling back to the
committed ``config/paths.example.yaml``) and exposes resolved paths. This is the
single source of truth that replaces the ~131 hardcoded ``Yeast_ML`` paths that
used to be scattered across the analysis scripts.

Resolution order for the config file:
  1. ``$SHORKIE_CONFIG``  (explicit path)
  2. ``<repo>/config/paths.yaml``      (user copy, git-ignored)
  3. ``<repo>/config/paths.example.yaml``  (committed template)

Inside the YAML, ``${token}`` is interpolated by, in order: an environment
variable named ``token``; otherwise a dotted key path within the same config
(e.g. ``${data_root}`` or ``${genome.fasta}``). Unresolved tokens are left
verbatim so missing optional values fail loudly at use-time, not load-time.

Typical use::

    from shorkie import config
    fasta = config.path("genome.fasta")          # -> pathlib.Path
    nfolds = config.get("models.num_folds", 8)
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import yaml

# src/shorkie/config.py  ->  <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_TOKEN = re.compile(r"\$\{([^}]+)\}")


def _config_path() -> Path:
    explicit = os.environ.get("SHORKIE_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    user = _CONFIG_DIR / "paths.yaml"
    return user if user.exists() else _CONFIG_DIR / "paths.example.yaml"


def _get_dotted(data, dotted):
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _interpolate(value, root):
    if isinstance(value, dict):
        return {k: _interpolate(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, root) for v in value]
    if not isinstance(value, str):
        return value

    prev = None
    cur = value
    for _ in range(10):  # resolve nested references, bounded
        if cur == prev:
            break
        prev = cur

        def _repl(match):
            token = match.group(1)
            if token in os.environ:
                return os.environ[token]
            sub = _get_dotted(root, token)
            return str(sub) if sub is not None else match.group(0)

        cur = _TOKEN.sub(_repl, cur)
    return cur


class Config:
    """Read-only view over the resolved configuration."""

    def __init__(self, data: dict, source: str):
        self._data = data
        self.source = source

    def get(self, dotted: str, default=None):
        val = _get_dotted(self._data, dotted)
        return default if val is None else val

    def path(self, dotted: str, default=None):
        val = self.get(dotted, default)
        return Path(val).expanduser() if val is not None else None

    @property
    def repo_root(self) -> Path:
        return _REPO_ROOT

    def as_dict(self) -> dict:
        return dict(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return f"Config(source={self.source!r})"


@functools.lru_cache(maxsize=None)
def load(path: str | None = None) -> Config:
    cfg_path = Path(path) if path else _config_path()
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh) or {}
    # two passes so keys that reference other keys (which themselves reference
    # env vars) settle.
    resolved = _interpolate(raw, raw)
    resolved = _interpolate(resolved, resolved)
    return Config(resolved, str(cfg_path))


def get(dotted: str, default=None):
    """Module-level convenience: ``config.get('genome.fasta')``."""
    return load().get(dotted, default)


def path(dotted: str, default=None):
    """Module-level convenience returning a :class:`pathlib.Path`."""
    return load().path(dotted, default)


def repo_root() -> Path:
    return _REPO_ROOT
