# config/

Path and scheduler configuration. **Every filesystem path in this repository resolves through here** —
no pipeline script hardcodes a machine path.

| File | What |
|------|------|
| `paths.example.yaml` | Template for `config/paths.yaml`. Copy it, then edit `release_root` (and `work_root` if you are re-running the original training). |
| `slurm.example.yaml` | Template for `config/slurm.yaml` — partition/account profiles used by `scripts/common/submit.sh`. |

```bash
cp config/paths.example.yaml config/paths.yaml
```

Resolution order is `$SHORKIE_CONFIG` → `config/paths.yaml` → `config/paths.example.yaml`, so a fresh
clone works with no setup. `${token}` interpolates against environment variables first, then other keys
in the same file; `${repo_root}` is injected automatically.

```python
from shorkie import config
config.path("models.shorkie_finetuned")
config.path("genome.fasta")
```

The defaults point at the layout `data/download.sh` creates, so for normal use you only need
`release_root`. Keys under `results:` are work-directory outputs that are **not** part of the public
release — they are only needed for the gated figure notebooks. See [`../data/README.md`](../data/README.md)
for what is downloadable.
