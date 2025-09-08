#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
from math import log
from tqdm import tqdm

# Import modules from your prixfixe package.
from prixfixe.autosome import AutosomeDataProcessor, AutosomeFirstLayersBlock, AutosomeCoreBlock, AutosomeFinalLayersBlock, AutosomeTrainer, AutosomePredictor
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
from prixfixe.prixfixe import PrixFixeNet

def setup_model(MODEL_LOG_DIR, cuda_device_id=0, MODEL_NAME="DREAM_Atten"):
    """
    Set up and return the model and predictor.
    """
    # Set random seed
    generator = torch.Generator()
    generator.manual_seed(2147483647)
    
    # Initialize the data processor (needed for some parameters)
    TRAIN_DATA_PATH = "data/demo_train.txt"       # change filename to actual training data
    VALID_DATA_PATH = "data/demo_val.txt"          # change filename to actual validation data
    PLASMID_PATH = "data/plasmid.json"
    SEQ_SIZE = 150

    dataprocessor = AutosomeDataProcessor(
        path_to_training_data=TRAIN_DATA_PATH,
        path_to_validation_data=VALID_DATA_PATH,
        train_batch_size=512,
        batch_per_epoch=10,
        train_workers=8,
        valid_batch_size=4096,
        valid_workers=8,
        shuffle_train=True,
        shuffle_val=False,
        plasmid_path=PLASMID_PATH,
        seqsize=SEQ_SIZE,
        generator=generator
    )
    # Prepare a dataloader to initialize internals (if needed)
    next(dataprocessor.prepare_train_dataloader())


    if MODEL_NAME == "DREAM_Atten":
        first = AutosomeFirstLayersBlock(in_channels=dataprocessor.data_channels(),
                                        out_channels=256, 
                                        seqsize=dataprocessor.data_seqsize())

        core = UnlockDNACoreBlock(
            in_channels = first.out_channels, out_channels= first.out_channels, seqsize = dataprocessor.data_seqsize(), n_blocks = 4,
                                            kernel_size = 15, rate = 0.1, num_heads = 8)

        final = AutosomeFinalLayersBlock(in_channels=core.out_channels, 
                                        seqsize=core.infer_outseqsize())
        model = PrixFixeNet(
            first=first,
            core=core,
            final=final,
            generator=generator
        )
    elif MODEL_NAME == "DREAM_CNN":
        first = BHIFirstLayersBlock(
            in_channels = dataprocessor.data_channels(),
            out_channels = 320,
            seqsize = dataprocessor.data_seqsize(),
            kernel_sizes = [9, 15],
            pool_size = 1,
            dropout = 0.2
            )

        core = AutosomeCoreBlock(in_channels=first.out_channels,
                                 out_channels =64,
                                 seqsize=first.infer_outseqsize())

        final = AutosomeFinalLayersBlock(in_channels=core.out_channels, 
                                         seqsize=core.infer_outseqsize())
        model = PrixFixeNet(
            first=first,
            core=core,
            final=final,
            generator=generator
        )
    elif MODEL_NAME == "DREAM_RNN":
    
        # Initialize the model components
        first = BHIFirstLayersBlock(
            in_channels = dataprocessor.data_channels(),
            out_channels = 320,
            seqsize = dataprocessor.data_seqsize(),
            kernel_sizes = [9, 15],
            pool_size = 1,
            dropout = 0.2
        )

        core = BHICoreBlock(
            in_channels = first.out_channels,
            out_channels = 320,
            seqsize = first.infer_outseqsize(),
            lstm_hidden_channels = 320,
            kernel_sizes = [9, 15],
            pool_size = 1,
            dropout1 = 0.2,
            dropout2 = 0.5
        )

        final = AutosomeFinalLayersBlock(
            in_channels=core.out_channels, 
            seqsize=core.infer_outseqsize()
        )

        model = PrixFixeNet(
            first=first,
            core=core,
            final=final,
            generator=generator
        )
    
    # Load the pre-trained model weights.
    model_pth = os.path.join(MODEL_LOG_DIR, 'model_best.pth')
    state_dict = torch.load(model_pth, map_location=torch.device(f"cuda:{cuda_device_id}"))
    model.load_state_dict(state_dict)
    
    # Set up the predictor on the proper device.
    predictor = AutosomePredictor(
        model=model,
        model_pth=model_pth,
        device=torch.device(f"cuda:{cuda_device_id}")
    )
    
    return predictor

def process_predictions(predictor, input_sequences_file, output_file):
    """
    Reads the input sequences TSV file, predicts expression for ref and alt sequences,
    computes the logSED score, and writes results to a TSV.
    """
    # Read input sequences from previous step.
    try:
        df = pd.read_csv(input_sequences_file, sep="\t")
    except Exception as e:
        print(f"Error reading {input_sequences_file}: {e}")
        return
    
    results = []
    # Iterate through each eQTL entry.
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing eQTLs"):
        Chr = row.get("Chr", "")
        ChrPos = row.get("ChrPos", "")
        Gene = row.get("Gene", "")
        Position_Gene = Chr + ":" + str(ChrPos) + "_" + Gene
        try:
            ref_seq = row["final_ref_seq"]
            alt_seq = row["final_alt_seq"]
        except KeyError as e:
            print(f"Missing expected column in row {idx}: {e}", flush=True)
            continue

        try:
            # Predict expression for both sequences.
            ref_pred = predictor.predict(ref_seq)
            alt_pred = predictor.predict(alt_seq)
        except Exception as e:
            print(f"Prediction error at row {idx}: {e}", flush=True)
            continue
        
        # Convert predictions to float (in case they are tensors or other types)
        try:
            ref_pred_val = float(ref_pred)
            alt_pred_val = float(alt_pred)
        except Exception as e:
            print(f"Conversion error at row {idx}: {e}", flush=True)
            continue
        
        # Calculate logSED = log(alt_pred + 1) - log(ref_pred + 1)
        try:
            log_sed = np.log(alt_pred_val + 1) - np.log(ref_pred_val + 1)
        except Exception as e:
            print(f"Log calculation error at row {idx}: {e}", flush=True)
            continue
        
        # Append results, preserving identifying columns
        results.append({
            "Gene": row.get("Gene", idx),
            "Chr": row.get("Chr", ""),
            "ChrPos": row.get("ChrPos", ""),
            "Position_Gene": Position_Gene,
            "tss_dist": row.get("tss_dist", ""),
            "ref_pred": ref_pred_val,
            "alt_pred": alt_pred_val,
            "logSED": log_sed
        })
    
    # Write output results to a TSV file.
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(output_file, sep="\t", index=False)
        print("Prediction and logSED score calculation completed. Results written to", output_file)
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")

def main():
    negsets = [1, 2, 3, 4]
    models = ["DREAM_Atten", "DREAM_CNN", "DREAM_RNN"]
    for negset in negsets:
        for model in models:
            print(f"Processing negative set {negset} with model {model}")
            # Define model directory and file paths.
            if model == "DREAM_Atten":
                # DREAM-Atten
                MODEL_NAME = "DREAM_Atten"
                MODEL_LOG_DIR = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/prixfixe_model_weights/0_0_2_0"
            elif model == "DREAM_CNN":
                # DREAM-CNN
                MODEL_NAME = "DREAM_CNN"
                MODEL_LOG_DIR = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/prixfixe_model_weights/0_1_0_0"

            elif model == "DREAM_RNN":
                # DREAM-RNN
                MODEL_NAME = "DREAM_RNN"
                MODEL_LOG_DIR = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/MPRA/prixfixe_model_weights/0_1_1_0"
            # Use the output from the previous negative sequence generation step.
            input_sequences_file = f"./results/output_neg_sequences_{negset}.tsv"
            output_file = f"results/{MODEL_NAME}/final_neg_predictions_{negset}.tsv"

            os.makedirs(f"results/{MODEL_NAME}", exist_ok=True)
                
            # Set up the predictor.
            predictor = setup_model(MODEL_LOG_DIR, cuda_device_id=0, MODEL_NAME=MODEL_NAME)
            
            # Process predictions and write output.
            process_predictions(predictor, input_sequences_file, output_file)

if __name__ == "__main__":
    main()
