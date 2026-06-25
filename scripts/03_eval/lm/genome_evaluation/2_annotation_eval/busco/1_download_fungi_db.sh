cfg() { python -c "import sys; from shorkie import config; print(config.get(sys.argv[1]) or '')" "$1"; }
CORPUS_BUILD_DATA_ROOT="$(cfg corpus_build_data_root)"

wget https://busco-data.ezlab.org/v5/data/lineages/fungi_odb10.2024-01-08.tar.gz --directory-prefix ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/busco/
tar -xzf ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/busco/fungi_odb10.2024-01-08.tar.gz -C ${CORPUS_BUILD_DATA_ROOT}/yeast/ensembl_fungi_59/busco
