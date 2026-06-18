# notebooks/

Figure-reproduction notebooks (added in Phase 7). Conventions:

- One notebook per major paper figure / panel group, named `figNN_<topic>.ipynb`.
- Notebooks **import from `src/shorkie`** (helpers, config, model ensemble) — they
  do not redefine utilities or hardcode paths.
- Each notebook reads inputs through `shorkie.config` and the artifacts listed in
  `data/manifest.json`, and states at the top which figure it reproduces and which
  upstream `scripts/` stage must have run first.
- Pin the kernel to the `yeast_ml` environment.
