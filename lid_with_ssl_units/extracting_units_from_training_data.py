import torch
torch.ones(1).to("cuda") # This is to ensure that the GPU is initialized before the next import
print(torch.cuda.is_available())
import os, sys
import numpy as np
import random
from collections import defaultdict
import torchaudio
from transformers import AutoProcessor, WavLMModel
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining, Wav2Vec2Processor
from datasets import load_dataset, concatenate_datasets, Dataset, Audio
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
from multiprocessing import Pool, cpu_count

sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/dataloading/")
from dataset_loader import load_lid_dataset
from vl107 import load_vl107, load_vl107_lang
from edacc import load_edacc
# from vl107 import load_vl107, load_vl107_lang
# import lhotse
print("Available CPU cores:", cpu_count())

from argparse import ArgumentParser

from kmeans_on_units import KMeansOnUnits

import logging

def get_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

global logger

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="Model name")
    parser.add_argument("--layer", type=int, default=8, help="Layer to extract representations from")
    parser.add_argument("--dataset_dir", type=str, default="/exp/jvillalba/corpora/voxlingua107", help="Directory containing audio files")
    parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
    parser.add_argument("--lang", type=str, default=None, help="Language to extract audio files from")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--kmeans_dir", type=str, default=None, help="Directory to save or load kmeans model")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters for kmeans")
    parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory for units")
    parser.add_argument("--log_file", type=str, default="train.log", help="Log file")
    return parser.parse_args()


def prepare_dataset(batch, processor = None):
    
    array_values = [np.array(signal).squeeze() for signal in batch["signal"]]
    batch["input_values"] = processor(array_values, sampling_rate = 16_000, \
                                    padding = True, \
                                    return_tensors="pt").input_values
    batch["lengths"] = [array.shape[0] for array in array_values]
    batch["lang"] = [item for item in batch["lang"]]
    batch["accent"] = [item for item in batch["accent"]]
    batch["audio_file"] = [item for item in batch["audio_file"]]

    # batch["accents"] = [item["accent"] for item in batch["accent"]]
    # logger.info(type(batch["input_values"]))
    # logger.info(batch["input_values"].shape)
    # array_values = [item["array"] for item in batch["signal"]]
    # batch["input_values"] = processor(array_values, sampling_rate = 16_000 , \
    #                                 padding = True, \
    #                                 return_tensors="pt").input_values
    # batch["lengths"] = [array.shape[0] for array in array_values]
    return batch


def collate_fn(batch):
    return {
        "input_values": torch.stack([torch.tensor(item["input_values"]) for item in batch]),
        # "wav_files": [item["wav_file"] for item in batch],
        # "text_files": [item["text_file"] for item in batch],
        # "timestamp_files": [item["timestamp_file"] for item in batch],
        "lengths": [item["lengths"] for item in batch],
        "lang": [item["lang"] for item in batch],
        "audio_file": [item["audio_file"] for item in batch],
        "accent": [item["accent"] for item in batch],
    }


def extract_embeddings(output_dir, model = None, dataloader = None, layer = None, save = False):
    '''Load embeddings from disk if already computed, otherwise compute and save them'''

    if os.path.exists(os.path.join(output_dir, f"embeddings_data.pkl")):
        print("WARNING: Loading model representations of audio from disk (extract_embeddings)")
        with open(os.path.join(output_dir, f"embeddings_data.pkl"), "rb") as f:
            data = pkl.load(f)
        
        # with open(os.path.join(output_dir, f"all_full_reps.pkl"), "rb") as f:
        #     all_full_reps = pkl.load(f)
        # with open(os.path.join(output_dir, f"all_lengths.pkl"), "rb") as f:
        #     all_lengths = pkl.load(f)
        # with open(os.path.join(output_dir, f"all_audio_files.pkl"), "rb") as f:
        #     all_audio_files = pkl.load(f)
        # with open(os.path.join(output_dir, f"all_accents.pkl"), "rb") as f:
        #     all_accent = pkl.load(f)

        # return all_full_reps, all_lengths, all_audio_files
        return data

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_full_reps = [] # Representations of each sample 
    all_lengths = [] # Length of each sample
    all_audio_files = []
    all_accents = []
    all_langs = []

    model.eval()
    model.to("cuda")
    ###############################
    #### TESTING ONLY, ON CPU ####
    # model.to("cpu")
    ###############################

    for batch in tqdm(dataloader):
        # logger.info(batch)
        # logger.info(batch["input_values"].shape)
        # logger.info(batch["text_files"])
        # batch_size, sequence_length, -1

        # batch["input_values"] = batch["input_values"].to("cpu")
        with torch.no_grad():
            # logger.info(f"Sample input_values: {batch['input_values']}")
            # logger.info(f"Precision: {batch['input_values'].dtype}")
            # logger.info(f"Model precision: {model.get_input_embeddings().weight.dtype}")
            batch["input_values"] = batch["input_values"].to("cuda")
            outputs = model(batch["input_values"], output_hidden_states=True)
            outputs_layer = outputs.hidden_states[layer] # (1st output is embedding layer)
            del outputs
            # logger.info(f"Sample output_layer rep: {outputs_layer}")
            # logger.info(f"outputs.hidden_states: {outputs.hidden_states}")

            # Shape of ouputs_layer: batch_size, sequence_length, hidden_size
            for i in range(len(batch["audio_file"])):
                length = batch["lengths"][i]
                all_lengths.append(length)
                true_sequence_length = length//320 + 1 # going from samples to frames, 1 frame = 20ms, 320 samples = 20ms  since sampling rate is 16kHz
                rep = outputs_layer[i][:true_sequence_length, :].cpu() # We only want the representations for the original length
                all_full_reps.append(rep)
                # logger.info(f"Length: {length}")
                # logger.info(f"True sequence length: {true_sequence_length}")
                # logger.info(f"Og rep shape: {outputs_layer[i].shape}")
                # logger.info(f"Rep shape: {rep.shape}")

            all_audio_files.extend(batch["audio_file"])
            all_accents.extend(batch["accent"])
            all_langs.extend(batch["lang"])

    data = {"all_full_reps": all_full_reps, "all_lengths": all_lengths, \
            "all_audio_files": all_audio_files, "all_accents": all_accents, \
                "all_langs": all_langs}
    # logger.info(f"Number of representations: {len(all_full_reps)}")
    # logger.info(f"Example representation: {all_full_reps[0].shape}, {all_full_reps[0]}")
    # Save the representations and labels
    if save:
        with open(os.path.join(output_dir, f"embeddings_data.pkl"), "wb") as f:
            pkl.dump(data, f)

        # with open(os.path.join(output_dir, f"all_full_reps.pkl"), "wb") as f:
        #     pkl.dump(all_full_reps, f)
        # with open(os.path.join(output_dir, f"all_lengths.pkl"), "wb") as f:
        #     pkl.dump(all_lengths, f)
        # with open(os.path.join(output_dir, f"all_audio_files.pkl"), "wb") as f:
        #     pkl.dump(all_audio_files, f)
        # with open(os.path.join(output_dir, f"all_accents.pkl"), "wb") as f:
        #     pkl.dump(all_accents, f)

    return data
    

def extract_phoneme_segment_embeddings(output_dir = None, data = None, segment_size = 100, save = False):

    '''Extract phoneme segment embeddings from the full representations. segment_size is in ms
    data: {
        "all_full_reps": List of model (e.g. W2V2) representations of data,
        "all_lengths": List of lengths of each sample,
        "all_audio_files": List of audio files,
        "all_accents": List of accents,
        "all_langs": List of languages
    }
    Returns:
    data: {
        "all_segment_reps": List of segment representations,
        "all_lengths": List of lengths of each sample,
        "all_audio_files": List of audio files,
        "all_accents": List of accents,
        "all_langs": List of languages
    }
    '''

    if output_dir:
        if os.path.exists(os.path.join(output_dir, f"phoneme-embeddings_data.pkl")):
            with open(os.path.join(output_dir, f"phoneme-embeddings_data.pkl"), "rb") as f:
                data = pkl.load(f)
            return data
    
    all_full_reps, all_lengths, all_audio_files, all_accents, all_langs = data["all_full_reps"], data["all_lengths"], data["all_audio_files"], data["all_accents"], data["all_langs"]
    # pool_size is the number of frames in a segment
    pool_size = segment_size//20 # 20ms per frame
    
    # divisible_size = all_full_reps[0].shape[0] - all_full_reps[0].shape[0] % pool_size
    # all_full_reps = all_full_reps[:, :divisible_size, :]
    # all_full_reps = all_full_reps.reshape(all_full_reps.shape[0], pool_size, -1, all_full_reps.shape[-1])
    # all_segment_reps = all_full_reps.mean(axis=1)
    # logger.info(f"all_segment_reps.shape: {all_segment_reps.shape}")

    all_segment_reps = []
    # We want to average the representations over 100ms segments i.e. 5 frames
    ## Currently, we have k frames where each is 20 ms
    ## Shape of full_rep: k, hidden_size
    ## We want to collapse k such that we are pooling over 100 ms segments 
    ## We want: k//pool_size, hidden_size
 
    for idx in range(len(all_full_reps)):
        full_rep = all_full_reps[idx]
        seg_reps = []
        # logger.info(f"reps.shape: {reps.shape}")
        # Wav2Vec2 timestamps --> sequence index 
        # Each frame is 20ms
        # Split the sequence into 100ms segments
        divisible_size = full_rep.shape[0] - full_rep.shape[0] % pool_size
        full_rep = full_rep[:divisible_size, :]
        full_rep = full_rep.reshape(-1, pool_size, full_rep.shape[-1])
        seg_reps = full_rep.mean(axis=1)

        # for i in range(num_reps):
        #     ## 1 frame = 20ms, 5 frames = 100ms
        #     start_idx, end_idx = i*5, (i+1)*5
        #     rep = full_rep[start_idx:end_idx, :].mean(dim=0)
        #     # logger.info(f"rep.shape: {rep.shape}")
        #     seg_reps.append(rep)
        all_segment_reps.append(seg_reps)
    
    data = {
        "all_segment_reps": all_segment_reps,
        "all_lengths": all_lengths,
        "all_audio_files": all_audio_files,
        "all_accents": all_accents,
        "all_langs": all_langs
    }
    # We save the segment representations by default
    if save:
        assert output_dir is not None, "Output directory not provided, cannot save segment representations. Pass save=False to skip saving."
        with open(os.path.join(output_dir, f"phoneme-embeddings_data.pkl"), "wb") as f:
            pkl.dump(data, f)

    return data


def compute_kmeans_reps(data, kmeans_dir = None, output_dir = None, n_clusters = 100, save = False):
    '''
    Train kmeans on the segment representations, returns the units and centroids
    all_segment_reps: List of segment representations
    kmeans_dir: Directory to save the kmeans model
    output_dir: Directory to save the units
    n_clusters: Number of clusters for kmeans
    save: whether to save the units to disk
    '''
    all_segment_reps, all_lengths, all_audio_files, all_accents, all_langs = data["all_segment_reps"], data["all_lengths"], data["all_audio_files"], data["all_accents"], data["all_langs"]

    kmeans = KMeansOnUnits(n_clusters = n_clusters, output_dir = kmeans_dir)
    if kmeans.trained == False: # This is true if the model has already been saved to kmeans_dir
        print("WARNING: Training Kmeans from scratch!")
        kmeans.train(all_segment_reps)
        kmeans.save_model()
    else:
        print("WARNING: Using pre-trained Kmeans model")

    kmeans_reps = kmeans.predict(all_segment_reps)
    data = {
        "sequences": kmeans_reps,
        "all_lengths": all_lengths,
        "all_audio_files": all_audio_files,
        "all_accents": all_accents,
        "all_langs": all_langs
    }
    if save:
        # kmeans.save_centroid_sequences(kmeans_reps, output_dir)
        with open(os.path.join(output_dir, f"kmeans_data.pkl"), "wb") as f:
            pkl.dump(data, f)
    return data


def main():
    global logger
    args = parse_args()
    model_name = args.model_name
    layer = args.layer
    dataset_dir = args.dataset_dir
    per_lang = args.per_lang
    lang = args.lang
    batch_size = args.batch_size
    output_dir = args.output_dir
    kmeans_dir = args.kmeans_dir
    n_clusters = args.n_clusters
    
    logger = get_logger(args.log_file)

    os.makedirs(output_dir, exist_ok=True)


    # Load the dataset
    logger.info(f"Loading dataset: {dataset_dir}")

    dataset = load_lid_dataset(dataset_dir, lang = lang, per_lang = per_lang)
    if dataset is None:
        logger.info(f"Dataset {dataset_dir} for lang {lang} not found")
        return

    # Load the model
    logger.info(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTraining.from_pretrained(model_name)

    
    dataset = dataset.map(prepare_dataset, fn_kwargs = {"processor": processor} , batched=True, batch_size=batch_size, remove_columns=["signal"])

    # dataset = dataset.map(prepare_dataset, fn_kwargs = {"processor": processor} , batched=True, batch_size=batch_size, remove_columns=["signal"], \
    #                       num_proc = 4, keep_in_memory=False)
    
    # Shape of input
    # logger.info(f"Shape of input: {torch.tensor(dataset[0]["input_values"]).shape}")
          
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

    # Extract embeddings
    logger.info(f"Extracting embeddings from layer {layer}")
    data = extract_embeddings(output_dir, model = model, dataloader = dataloader, layer = layer)
    logger.info(f"Number of representations: {len(data["all_full_reps"])}")

    # Extract phoneme segment embeddings
    logger.info("Extracting phoneme segment embeddings")
    data = extract_phoneme_segment_embeddings(output_dir, data)
    logger.info(f"Number of segment representations: {len(data["all_segment_reps"])}")

    # Train kmeans and save the units
    logger.info(f"Training kmeans on segment representations")
    return compute_kmeans_reps(data, kmeans_dir = kmeans_dir, output_dir = output_dir, n_clusters = n_clusters, save = True)


if __name__ == "__main__":

    main()


