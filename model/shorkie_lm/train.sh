#!/bin/sh

mkdir train

python /home/jlinder/baskerville-yeast/src/baskerville/scripts/hound_train.py --eval_dir /scratch4/jlinder/seqnn/data/yeast/ensembl_fungi_59_3/data_saccharomycetales_gtf/ -o train/ params.json /scratch4/jlinder/seqnn/data/yeast/ensembl_fungi_59_3/data_saccharomycetales_gtf 1>train/train.out 2>train/train.err
