from ete4 import Tree

highlighted = {
    "Saccharomyces_cerevisiae_S288C_-_559292",
    "Candida_albicans_SC5314_-_237561",
    "Nakaseomyces_glabratus_CBS_138_-_284593",
    "Schizosaccharomyces_pombe_972h-_-_284812",
    "Neurospora_crassa_OR74A_-_367110",
    "Agaricus_bisporus_var._burnettii_JB137-S8_-_597362",
    "Lentinula_edodes_-_5353",
    "Pleurotus_ostreatus_PC15_-_1137138",
    "Tuber_melanosporum_Mel28_-_656061"
}

# Read the entire Newick file and strip whitespace
with open("new_tree_with_annotation_names.nwk", "r") as f:
    tree_str = f.read().strip()

print("Original tree:", tree_str)
if not tree_str.endswith(";"):
    tree_str += ";"

# Load the tree from the file generated above.
# t = Tree("species_tree.nwk")
# Load the tree
t = Tree(tree_str)
print("tree:", t)


# Define a helper function to check if any leaf under a node is highlighted.
def has_highlighted(node):
    for leaf in node.get_leaves():
        if leaf.name in highlighted:
            return True
    return False

collapse_nodes = []
# Traverse through all nodes in the tree.
for node in t.traverse():
    # Check if node is not a leaf.
    if not node.is_leaf:
        if not has_highlighted(node):
            # If the node has a name, use it; otherwise, generate an identifier based on its descendant leaves.
            node_id = node.name if node.name else ",".join(sorted(leaf.name for leaf in node.get_leaves()))
            collapse_nodes.append(node_id)

# Print the COLLAPSE annotation file.
print("COLLAPSE")
print("DATA")
for node_id in collapse_nodes:
    print(node_id)