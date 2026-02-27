#!/usr/bin/env python3
from ete4 import Tree

# Read the entire Newick file and strip whitespace
with open("species_tree.nwk", "r") as f:
    tree_str = f.read().strip()

print("Original tree:", tree_str)
if not tree_str.endswith(";"):
    tree_str += ";"

# Load the tree from the file generated above.
# t = Tree("species_tree.nwk")
# Load the tree
t = Tree(tree_str)
print("tree:", t)

# Optional: Shorten taxon labels if needed
for node in t.traverse():
    if node.is_leaf:
        # Remove any enclosing quotes and split at " - " to get a shorter label.
        original_label = node.name.strip("'")
        short_label = original_label.split(" - ")[0]  # customize as needed
        # short_label = " ".join(short_label.split(" ")[:2])

        print("Shortened label:", short_label)
        print("Original label:", node.name)
        node.name = short_label

# Save the modified tree to a new Newick file
output_nwk = "new_tree.nwk"
t.write(outfile=output_nwk)
print("Newick tree saved as:", output_nwk)

# Render and save the tree as an image (PNG)
output_img = "tree.png"
# You can optionally specify width, dpi, etc.
t.render(output_img, w=600)
print("Tree image saved as:", output_img)