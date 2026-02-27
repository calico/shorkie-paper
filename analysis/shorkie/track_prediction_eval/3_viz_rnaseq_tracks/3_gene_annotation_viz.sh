#!/bin/bash
ROOT_DIR=${ROOT_DIR_ENV:-"../../.."}
python 3_gene_annotation_viz.py --region "chrI:5000-30000" --output genes.png --root_dir "$ROOT_DIR"