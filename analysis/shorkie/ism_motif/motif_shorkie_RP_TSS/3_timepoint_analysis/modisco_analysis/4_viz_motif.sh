#!/bin/sh


#SBATCH --job-name=4_viz_motif
#SBATCH --output=job_output_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --mem=32G
#SBATCH --array=0-2


# Run TF-MoDISco
python 4_viz_motif.py > 4_viz_motif.out