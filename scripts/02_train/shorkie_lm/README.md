# Shorkie_LM — language-model pretraining

`train.sh` contains the command used to pretrain the published **Shorkie_LM** masked
DNA language model (the `--restore` checkpoint that `Shorkie_finetuned` starts from).
See [`../README.md`](../README.md) for how the three variants relate.

*Notes:*

- Training runs through `hound_train.py` from the
  [baskerville-yeast](https://github.com/calico/baskerville-yeast) submodule
  (single fold; `loss=mlm`, `mask_rate=0.15`, `use_bert=true`).
- Trains on the 165_Saccharomycetales corpus (`datasets.lm_train_dir`); GPU required.
- The published weights + `params.json` are released at `gs://seqnn-share/shorkie_lm/`
  and catalogued in `data/manifest.json` (fetch with `data/download.sh --models lm`).
- Run `bash train.sh --dry-run` to print the exact resolved command.
