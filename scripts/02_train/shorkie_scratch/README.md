# Shorkie_scratch — supervised training from random init

`make_model.sh` trains the 8-fold supervised ensemble **from random initialization**
(no LM pretraining) — the ablation baseline that isolates the contribution of the
`Shorkie_LM` pretraining used by `Shorkie_finetuned`.

*Notes:*

- It is **identical** to `../shorkie_finetuned/make_model.sh` except it omits the
  `--restore` flag, and its `params.json` differs in exactly two `train` fields
  (`task`: `supervised` vs `fine-tune`, and `learning_rate`: `5e-4` vs `2e-5`); the
  `model` block is identical. The full comparison is in [`../README.md`](../README.md).
- Multi-fold training runs through `westminster_train_folds.py` from the
  [westminster](https://github.com/calico/westminster) submodule, which fans out to
  `hound_train.py` from [baskerville-yeast](https://github.com/calico/baskerville-yeast).
- This is the ablation released as **Shorkie_Random_Init**
  (`gs://seqnn-share/shorkie_models/shorkie_random_init/`, fetch with
  `data/download.sh --models random_init`); the committed `params.json` matches the
  released one. Run `bash make_model.sh --dry-run` to print the exact resolved command.
