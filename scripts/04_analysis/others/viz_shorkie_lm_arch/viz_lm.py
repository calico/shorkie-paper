from tensorflow.keras.utils import plot_model

# Suppose your model is called 'model'


# read model parameters
with open(params_file) as params_open:
    params = json.load(params_open)
params_model = params["model"]
params_train = params["train"]
print("* Before params_model: ", params_model)
if params_train["task"] == "fine-tune":
    num_species = 165
    params_model["num_features"] = num_species + 5
    params_train['r64_idx'] = 109
print("* After params_model: ", params_model)

# read targets
if options.targets_file is None:
    parser.error("Must provide targets file to clarify stranded datasets")
targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

# handle strand pairs
if "strand_pair" in targets_df.columns:
    # prep strand
    targets_strand_df = dataset.targets_prep_strand(targets_df)

    # set strand pairs (using new indexing)
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array(
        [orig_new_index[ti] for ti in targets_df.strand_pair]
    )
    params_model["strand_pair"] = [targets_strand_pair]

    # construct strand sum transform
    strand_transform = dataset.make_strand_transform(targets_df, targets_strand_df)
else:
    targets_strand_df = targets_df
    strand_transform = None
num_targets = targets_strand_df.shape[0]

#################################################################
# setup model

seqnn_model = seqnn.SeqNN(params_model)
seqnn_model.restore(model_file)
seqnn_model.build_slice(targets_df.index)
seqnn_model.build_ensemble(options.rc)

print(seqnn_model.model.summary())
model='conv_small.txt'
plot_model(seqnn_model.model, 
           to_file='model.png', 
           show_shapes=True, 
           show_layer_names=True)
