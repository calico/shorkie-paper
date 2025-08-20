#!/bin/bash
#SBATCH --job-name=atten_map
#SBATCH --output=job_output_%A_%a.log
#SBATCH --partition=bigmem
#SBATCH -A ssalzbe1_bigmem
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --export=ALL
#SBATCH --mail-type=END
#SBATCH --mail-user=kuanhao.chao@gmail.com
#SBATCH --array=0


mkdir -p embeddings_LM_sequence/viz_gene_intergenic

python 3_viz_clusters_LM_clustering.py \
  --embedding_pattern "./embeddings_LM_sequence/embeddings_chr*.h5" \
  --gtf_file /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_r64_gtf/gtf/GCA_000146045_2.59.fixed.gtf \
  --n_components 2 \
  --out_prefix ./embeddings_LM_sequence/viz_gene_intergenic/ \
  --n_clusters 12 \
  --groups "gene,intergenic,promoter"
  # --groups "protein_coding,snoRNA,tRNA,transposable_element,ncRNA,pseudogene,intergenic"
