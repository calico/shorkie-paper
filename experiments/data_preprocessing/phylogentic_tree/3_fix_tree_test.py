#!/usr/bin/env python3
from ete4 import Tree
import difflib

def clean_label(label):
    """Remove quotes and brackets from a label."""
    return label.replace("'", "").replace("[", "").replace("]", "").strip()

def fix_label_for_itol(label):
    """Fix a label for ITOL by replacing spaces with underscores."""
    return label.replace(" ", "_")

# --- Step 1: Read the Newick Tree ---
with open("species_tree.nwk", "r") as f:
    tree_str = f.read().strip()
if not tree_str.endswith(";"):
    tree_str += ";"

t = Tree(tree_str)

# --- Step 2: Define the Species List ---
species_str = """
Ascoidea rubescens DSM 1968
Ashbya gossypii
Babjeviella inositovora NRRL Y-12698
Brettanomyces bruxellensis
Brettanomyces bruxellensis
Brettanomyces bruxellensis AWRI1499
Brettanomyces naardenensis
Candida albicans
Candida albicans 12C
Candida albicans 19F
Candida albicans Ca529L
Candida albicans Ca6
Candida albicans GC75
Candida albicans L26
Candida albicans P34048
Candida albicans P37005
Candida albicans P37037
Candida albicans P37039
Candida albicans P57055
Candida albicans P57072
Candida albicans P60002
Candida albicans P75010
Candida albicans P75016
Candida albicans P75063
Candida albicans P76055
Candida albicans P76067
Candida albicans P78042
Candida albicans P78048
Candida albicans P87
Candida albicans P94015
Candida albicans SC5314
Candida albicans SC5314
Candida albicans WO-1
Candida auris
Candida dubliniensis CD36
Candida duobushaemulonis
Candida glabrata
Candida maltosa Xu316
Candida orthopsilosis Co 90-125
Candida parapsilosis
Candida tropicalis
Candida viswanathii str. ATCC 20962
Clavispora lusitaniae
Clavispora lusitaniae
Clavispora lusitaniae
Clavispora lusitaniae
Clavispora lusitaniae
Clavispora lusitaniae ATCC 42720
Clavispora lusitaniae str. CBS 6936
Cyberlindnera fabianii str. 65
Cyberlindnera fabianii str. YJS4271
Cyberlindnera jadinii NRRL Y-1542
Cyberlindnera jadinii str. CBS1600
Debaryomyces hansenii CBS767
Diutina rugosa
Eremothecium cymbalariae DBVPG#7215
Eremothecium gossypii FDAG1
Hanseniaspora guilliermondii
Hanseniaspora opuntiae
Hanseniaspora osmophila
Hanseniaspora uvarum
Hanseniaspora uvarum
Hanseniaspora uvarum
Hanseniaspora uvarum
Hanseniaspora uvarum DSM 2768
Hanseniaspora valbyensis NRRL Y-1626
Hyphopichia burtonii NRRL Y-1933
Kazachstania africana CBS 2517
Kazachstania naganishii CBS 8797
Kluyveromyces lactis str. NRRL Y-1140
Kluyveromyces marxianus DMKU3-1042
Kluyveromyces marxianus str. FIM1
Kluyveromyces marxianus str. NBRC 1777
Komagataella pastoris
Komagataella phaffii CBS 7435
Komagataella phaffii CBS 7435
Komagataella phaffii GS115
Komagataella phaffii str. WT
Kuraishia capsulata CBS 1993
Lachancea dasiensis CBS 10888
Lachancea fermentati
Lachancea lanzarotensis str. CBS 12615
Lachancea meyersii CBS 8951
Lachancea mirantina
Lachancea nothofagi CBS 11611
Lachancea quebecensis
Lachancea sp. CBS 6924
Lachancea thermotolerans CBS 6340
Lipomyces starkeyi NRRL Y-11557
Lodderomyces elongisporus NRRL YB-4239
Metschnikowia aff. pulcherrima str. APC 1.2
Metschnikowia bicuspidata str. Baker2002
Metschnikowia bicuspidata var. bicuspidata NRRL YB-4993
Metschnikowia sp. JCM 33374
Meyerozyma guilliermondii ATCC 6260
Meyerozyma sp. JA9
Millerozyma farinosa CBS 7064
Nadsonia fulvescens var. elongata DSM 6958
Naumovozyma castellii CBS 4309
Naumovozyma dairenensis CBS 421
Ogataea parapolymorpha DL-1
Ogataea polymorpha str. NCYC 495 leu1.1
Pachysolen tannophilus NRRL Y-2460
Pichia kudriavzevii str. 129
Pichia kudriavzevii str. CBS573
Pichia kudriavzevii str. Ckrusei653
Pichia kudriavzevii str. SD108
Pichia membranifaciens NRRL Y-2026
Pichia membranifaciens str. KS47-1
Saccharomyces cerevisiae
Saccharomycetaceae sp. 'Ashbya aceri'
Saccharomycodes ludwigii
Scheffersomyces stipitis CBS 6054
Spathaspora passalidarum NRRL Y-27907
Spathaspora sp. JA1
Sugiyamaella lignohabitans str. CBS 10342
Suhomyces tanzawaensis NRRL Y-17324
Tetrapisispora blattae CBS 6284
Tetrapisispora phaffii CBS 4417
Tortispora caseinolytica NRRL Y-17796
Torulaspora delbrueckii str. CBS 1146
Torulaspora globosa str. CBS2947
Torulaspora globosa str. CBS764
Vanderwaltozyma polyspora DSM 70294
Wickerhamiella sorbophila str. DS02
Wickerhamomyces anomalus NRRL Y-366-8
Wickerhamomyces ciferrii str. NRRL Y-1031
Yamadazyma tenuis ATCC 10573
Yarrowia lipolytica str. CLIB89 (W29)
Yarrowia lipolytica str. DSM 3286
Yarrowia lipolytica str. FKP355
Yarrowia lipolytica str. YB392
Yarrowia lipolytica str. YB419
Yarrowia lipolytica str. YB420
Yarrowia lipolytica str. YB566
Yarrowia lipolytica str. YB567
Zygosaccharomyces bailii
Zygosaccharomyces bailii ISA1307
Zygosaccharomyces mellis str. Ca-7
Zygosaccharomyces parabailii str. ATCC 60483
Zygosaccharomyces rouxii str. CBS 732
Zygosaccharomyces rouxii str. NBRC110957
Zygotorulaspora mrakii str. NRRL Y-6702
[Candida] arabinofermentans NRRL YB-2248
[Candida] auris
[Candida] auris
[Candida] auris
[Candida] auris
[Candida] auris
[Candida] auris str. 6684
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] glabrata
[Candida] haemuloni
[Candida] inconspicua
[Candida] intermedia
[Candida] intermedia
[Candida] pseudohaemulonii
""".strip()

species_list = [line.strip() for line in species_str.splitlines() if line.strip()]

# --- Step 3: Clean Tree Leaf Names and Match Species ---
# Update each leaf's name: first clean, then fix for ITOL (replace spaces with underscores)
for node in t.traverse():
    if node.is_leaf:
        fixed = fix_label_for_itol(clean_label(node.name))
        node.name = fixed

# Now, when matching, we also fix the species name (we assume species names in your list should match a part of the fixed label)
threshold = 0.8
matches = {}

for species in species_list:
    clean_species = fix_label_for_itol(clean_label(species))
    matches[clean_species] = []
    for node in t.traverse():
        if node.is_leaf:
            node_label = node.name  # already fixed for ITOL
            # First try a simple substring check.
            if clean_species.lower() in node_label.lower():
                matches[clean_species].append(node_label)
            else:
                # Fallback to fuzzy matching.
                sim = difflib.SequenceMatcher(None, clean_species.lower(), node_label.lower()).ratio()
                if sim >= threshold:
                    matches[clean_species].append(node_label)

# Create a unique set of matched node names.
matched_nodes = set()
for node_list in matches.values():
    for node in node_list:
        matched_nodes.add(node)

print("Total nodes to annotate:", len(matched_nodes))

# --- Step 4: Write the ITOL Annotation File using DATASET_COLORSTRIP Template ---
# We use space as the separator (and the fixed node IDs have no spaces).
annotation_filename = "itol_colorstrip_annotation.txt"
with open(annotation_filename, "w") as f:
    f.write("DATASET_COLORSTRIP\n")
    f.write("SEPARATOR SPACE\n")
    f.write("DATASET_LABEL Highlighted_Nodes\n")
    # Set the dataset color; here we use green (#00FF00).
    f.write("COLOR #00FF00\n")
    f.write("COLOR_BRANCHES 0\n")
    f.write("DATA\n")
    for node in matched_nodes:
        # Each line: node label, space, color
        f.write(f"{node} #00FF00\n")

print("Annotation file generated for ITOL:", annotation_filename)

# --- Step 5: Write the Tree with Updated Node Names ---
new_tree_filename = "new_tree_with_annotation_names.nwk"
t.write(outfile=new_tree_filename)
print("Tree with updated node names saved as:", new_tree_filename)
