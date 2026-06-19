import numpy as np
import os
import pandas as pd

def check_npz_file(file_path):
    # Load the .npz file
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Print all the keys and corresponding array shapes
    print(f"Contents of '{file_path}':")
    for key in data.files:
        print(f"Key: '{key}', Shape: {data[key].shape}, Data Type: {data[key].dtype}")
        print("min: ", np.min(data[key]), "; max: ", np.max(data[key]))
        # Summing along the second axis (axis=1) to get a result with shape (480, 200)
        summed_result = np.sum(data[key], axis=1)

        # Displaying the shape of the result to confirm it is (480, 200)
        print("summed_result: ", summed_result)
        
    # Close the file after checking
    data.close()





dataset = 'saccharomycetales'
model_archs = ['unet_small', 'unet_small_retry_1', 'unet_small_retry_2', 'unet_small_bert_aux_drop', 'unet_small_bert_aux_drop_retry_1', 'unet_small_bert_aux_drop_retry_2']

for model_arch in model_archs:
    output_dir = f'{dataset}_viz_seq/{model_arch}/'
    print("Processing model architecture: ", model_arch)
    # Save x_true and x_pred to an npz file
    x_true_npz_file_path = os.path.join(f'{output_dir}/x_true.npz')
    # Replace 'your_file.npz' with the path to your .npz file
    check_npz_file(x_true_npz_file_path)

    x_pred_npz_file_path = os.path.join(f'{output_dir}/x_pred.npz')
    check_npz_file(x_pred_npz_file_path)