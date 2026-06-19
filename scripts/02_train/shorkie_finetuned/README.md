# Shorkie_finetuned — supervised fine-tuning from Shorkie_LM

`make_model.sh` contains the command used to train the published **Shorkie**
(fine-tuned) 8-fold supervised ensemble, restoring weights from the `Shorkie_LM`
checkpoint. See [`../README.md`](../README.md) for the exact diff vs `Shorkie_scratch`
and the restore mechanism.

*Notes:*

- Multi-fold training runs through `westminster_train_folds.py` from the
  [westminster](https://github.com/calico/westminster) submodule, which fans out to
  `hound_train.py` from [baskerville-yeast](https://github.com/calico/baskerville-yeast).
- `--restore` loads the pretrained LM trunk (`models.shorkie_lm_checkpoint`); the
  supervised head is trained on the 5215-track dataset (`datasets.supervised_data`).
- The published weights + `params.json` + `targets.txt` are released at
  `gs://seqnn-share/shorkie/` and catalogued in `data/manifest.json`
  (fetch with `data/download.sh --models finetuned`, or `--minimal` for the
  `minimal_example/` layout).
- Run `bash make_model.sh --dry-run` to print the exact resolved command.
