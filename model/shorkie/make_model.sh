#!/bin/bash

#SBATCH --job-name=self-supervised_unet_small
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=20g
#SBATCH -o supervised_unet_small.%j.out
#SBATCH --mail-type=start,end

source "$(git rev-parse --show-toplevel)/scripts/common/env.sh"

python ${WESTMINSTER_SCRIPTS}/westminster_train_folds.py \
--restart \
-f 8 \
-e yeast_ml \
--restore ${WORK_ROOT}/lm_experiment/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/LM_Johannes/lm_saccharomycetales_gtf/lm_saccharomycetales_gtf_unet_small_bert_drop/train/model_best.h5 \
--eval_dir ${WORK_ROOT}/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/ \
-o train \
-q a100 \
--rc \
--shifts "0,1" \
params.json \
../data \
1>train.out 2>train.err