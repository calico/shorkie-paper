#!/bin/sh

source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"

mkdir train

python ${BASKERVILLE_SCRIPTS}/hound_train.py --eval_dir ${WORK_ROOT}/seqnn/data/yeast/ensembl_fungi_59_3/data_saccharomycetales_gtf/ -o train/ params.json ${WORK_ROOT}/seqnn/data/yeast/ensembl_fungi_59_3/data_saccharomycetales_gtf 1>train/train.out 2>train/train.err # TODO: verify path
