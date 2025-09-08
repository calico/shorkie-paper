# cat list_taxanomic_id.txt | ete4 ncbiquery --tree > species_tree.nwk
# cat list_taxanomic_id.txt | ete4 ncbiquery --taxid_attr=sci_name --tree > species_tree.nwk


cut -f1 list_taxanomic_id.txt | ete4 ncbiquery --tree > species_tree.nwk
