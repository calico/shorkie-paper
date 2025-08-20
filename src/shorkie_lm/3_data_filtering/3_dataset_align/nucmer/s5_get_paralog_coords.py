import sys 

# Get the dataset argument
dataset = sys.argv[1]

# Define the combinations
combinations = [
    ("train", "train"),
    ("train", "test"),
    ("train", "valid"),
    ("test", "test"),
    ("test", "valid"),
    ("valid", "valid")
]

# Loop through each combination and assign fa1 and fa2
for combination in combinations:
    fa1, fa2 = combination
    # Add your processing code here
    # store a list of paralogs here:
    paralogs = []
    paralog_ids = f"/home/khc/projects/ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{dataset}/dataset_stats/dataset_similarity/mummer/4_filter_aln_{dataset}_{fa1}_{fa2}_paralogs.bed"
    paralog_coords = f"/home/khc/projects/ssm/results/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/{dataset}/dataset_stats/dataset_similarity/mummer/5_paralog_coords_{dataset}_{fa1}_{fa2}.bed"
    with open(paralog_ids, "r") as fr:
        lines = fr.read().splitlines()
        for line in lines:
            paralogs.append(line)
    print("paralogs: ", paralogs)

    # # Read the gtf file (do the filtration here)
    # fw = open(paralog_coords, "w")
    # with open(f"/scratch4/khc/yeast_ssm/data/yeast/ensembl_fungi_59/test_chrXI_chrXIII_chrXV__valid_chrXII_chrXIV_chrXVI/data_{dataset}_gtf/gtf/GCA_000146045_2.59.gtf", "r") as fr:
    #     lines = fr.read().splitlines()
    #     for line in lines:
    #         eles = line.split("\t")
    #         if (len(eles) < 9): continue
    #         trans_id = eles[8].split(";")[0][3:]
    #         print("trans_id: ", trans_id)
    #         if trans_id in paralogs:
    #             print(eles)
    #             # eles
    #             fw.write(eles[0]+ "\t" + eles[3] + "\t" + eles[4] + "\t" + trans_id + "\t0\t" + eles[6] + "\n")