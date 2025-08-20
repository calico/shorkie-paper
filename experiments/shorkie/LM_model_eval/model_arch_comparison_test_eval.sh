#!/bin/sh

#SBATCH --partition=parallel
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --job-name=tfmodisco
#SBATCH --output=job_output_%A_%a.log
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#SBATCH --array=0

# Run TF-MoDISco
python model_arch_comparison_test_eval.py > model_arch_comparison_test_eval.out