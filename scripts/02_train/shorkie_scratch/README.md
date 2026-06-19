# Shorkie_scratch — supervised training from random init

`make_model.sh` trains the 8-fold supervised ensemble **from random initialization**
(no LM pretraining) — the ablation baseline that isolates the contribution of the
`Shorkie_LM` pretraining used by `Shorkie_finetuned`.

*Notes:*

- It is **identical** to `../shorkie_finetuned/make_model.sh` except it omits the
  `--restore` flag, and its `params.json` differs in five `train` fields
  (`task`, `learning_rate`, `warmup_steps`, `adam_beta1`, `adam_beta2`). The full
  comparison is in [`../README.md`](../README.md).
- Multi-fold training runs through `westminster_train_folds.py` from the
  [westminster](https://github.com/calico/westminster) submodule, which fans out to
  `hound_train.py` from [baskerville-yeast](https://github.com/calico/baskerville-yeast).
- This variant is **not publicly released** (work-dir only); reproduce it with this
  script. Run `bash make_model.sh --dry-run` to print the exact resolved command.
