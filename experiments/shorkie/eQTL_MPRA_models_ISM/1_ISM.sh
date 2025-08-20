# python 1_ISM.py \
#   --model-dir /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/prixfixe_model_weights/0_1_1_0 \
#   --input results/output_pos_sequences.tsv \
#   --ref-or-alt ref \
#   --output results/ism_ref_results.tsv

python 1_ISM.py \
  --model-dir /home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/prixfixe_model_weights/0_1_1_0 \
  --input results/output_pos_sequences.tsv \
  --ref-or-alt alt \
  --output results/ism_alt_results.tsv