'''Loads a dataset that looks like this:
{'signal': np.array, 'lang': str, 'accent': str, 'audio_file': str}
And adds reps from a given encoder model, as well as phoneme sequences from a given transcriber model.

{'lang': str, 'accent': str, \
    'audio_file': str, \
    'reps': torch.tensor, \
    'phone_sequence': list}

Writes the dataset to disk.

Can also be used to load a dataset that has previously  been computed.
''' 


import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.ones(1).to(device)


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ForPreTraining, AutoFeatureExtractor
from datasets import load_dataset
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import os, sys
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/dataloading/")
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_ssl_units/")
from dataset_loader import load_lid_dataset
from kmeans_on_units import KMeansOnUnits
# from vl107 import load_vl107, load_vl107_lang
# from edacc import load_edacc

from tqdm import tqdm

import logging

def get_logger(filename):
    if not filename:
        return None
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
    parser.add_argument("--transcriber_model", type=str, required=True, help="Model used to transcribe the audio files")
    parser.add_argument("--encoder_model", type=str, required=True, help="Model used to encode the audio files for acoustic representations")
    parser.add_argument("--ssl_model", type=str, required=True, help="Model used to extract the SSL units")
    parser.add_argument("--kmeans_dir", type=str, required=True, help="Directory containing the kmeans model")

    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for loading (see dataset_loader.py)")

    parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
    parser.add_argument("--lang", type=str, default=None, help="Language to extract audio files from")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--write_every_n_batches", type=int, default=100, help="Write to disk every n batches")
    parser.add_argument("--log_file", type=str, default="train.log", help="Log file")
    
    return parser.parse_args()


def prepare_dataset(batch, processor = None):
    
    array_values = [np.array(signal).squeeze() for signal in batch["signal"]]
    batch["input_values"] = processor(array_values, sampling_rate = 16_000, \
                                    padding = True, \
                                    return_tensors="pt").input_values
    batch["lengths"] = [array.shape[0] for array in array_values]

    return batch


def map_extract_embeddings(batch, model = None, layer = None):

    all_full_reps = [] # Representations of each sample 
    
    with torch.no_grad():

        batch["input_values"] = torch.stack([torch.tensor(input_value) for input_value in batch["input_values"]])
        batch["input_values"] = batch["input_values"].to("cuda")
        outputs = model(batch["input_values"], output_hidden_states=True)
        outputs_layer = outputs.hidden_states[layer] # (1st output is embedding layer)
        del outputs
        # logger.info(f"Sample output_layer rep: {outputs_layer}")
        # logger.info(f"outputs.hidden_states: {outputs.hidden_states}")

        # Shape of ouputs_layer: batch_size, sequence_length, hidden_size
        for i in range(len(batch["audio_file"])):
            length = batch["lengths"][i]

            true_sequence_length = length//320 + 1 # going from samples to frames, 1 frame = 20ms, 320 samples = 20ms  since sampling rate is 16kHz
            rep = outputs_layer[i][:true_sequence_length, :].cpu() # We only want the representations for the original length
            all_full_reps.append(rep)


    batch["ssl_all_full_reps"] = all_full_reps

    return batch
    

def map_extract_phoneme_segment_embeddings(batch, segment_size = 100):

    '''
    Extract phoneme segment embeddings from the full representations. segment_size is in ms
    '''

    all_full_reps = batch["ssl_all_full_reps"]
    all_lengths = batch["lengths"]
    
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
        full_rep = np.array(full_rep)
        # print(f"full_rep.shape: {full_rep.shape}")
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
    
    batch["ssl_all_segment_reps"] = all_segment_reps
    
    return batch

def map_compute_kmeans_reps(batch, kmeans = None):

    all_segment_reps = batch["ssl_all_segment_reps"]
    batch_size = len(all_segment_reps)
    # Shape of all_segment_reps: batch_size, sequence_length, hidden_size  
    all_segment_reps = np.concatenate(all_segment_reps, axis = 0)
    # Shape of all_segment_reps: (batch_size*sequence_length, hidden_size)

    kmeans_reps = kmeans.predict(all_segment_reps)
    # Shape of kmeans_reps: (batch_size*sequence_length,)

    kmeans_reps = np.reshape(kmeans_reps, (batch_size, -1))
    
    batch["ssl_kmeans_reps"] = kmeans_reps
    # print(batch)
    return batch



def map_encode_audio(batch, model_name, model, processor = None):
    '''
    Adds the acoustic representations to the batch
    '''
    if model_name == "ecapa-tdnn":
        signals = torch.stack([torch.tensor(signal) for signal in batch["signal"]])
        signals = signals.to(device)
        representations = model.encode_batch(signals)
        assert representations.shape[1] == 1, f"Expected shape (batch_size, 1, embedding_dim), got {representations.shape}"
        representations = representations.squeeze(axis = 1)
        batch["reps"] = representations
    
    else:
        raise NotImplementedError("Model not supported")

    return batch


def map_transcribe(batch, transcriber_model, processor = None):
    '''
    Adds the phoneme sequences to the batch
    '''
    input_values = torch.stack([torch.tensor(input_value) for input_value in batch["input_values"]])
    input_values = input_values.to(device)

    with torch.no_grad():
        logits = transcriber_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["phone_sequence"] = transcription

    return batch
    

def main(transcriber_model, encoder_model, ssl_unit_model, kmeans_dir, dataset_dir, per_lang, lang, batch_size, output_dir, log_file=None, rewrite=False):

    global logger
    logger = get_logger(log_file)

    outfile = os.path.join(output_dir, f"{lang}_data.jsonl")
    print(f"Outfile: {outfile}")
    # Load the dataset if it exists
    if os.path.exists(outfile) and not rewrite:
        if logger:
            logger.info(f"Dataset {outfile} already exists")
        print(f"Dataset {outfile} already exists")
        dataset = load_dataset("json", data_files = outfile)

        return dataset["train"]

        # Load the dataset
    if logger:
        logger.info(f"Loading dataset: {dataset_dir}")

    dataset = load_lid_dataset(dataset_dir, lang = lang, per_lang = per_lang)
    if dataset is None:
        if logger:
            logger.info(f"Dataset {dataset_dir} for lang {lang} not found")
        return None
    
    ######### FOR DEBUGGING ####
    # dataset = dataset.select(range(100))
    # debug = True
    # Load the model
    if encoder_model == "ecapa-tdnn":
        # Load the model
        if logger:
            logger.info(f"Loading model: {encoder_model}")
        from speechbrain.inference.classifiers import EncoderClassifier

        encoder = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"})
        print(f"Calculating acoustic representations for {len(dataset)} samples")
        dataset = dataset.map(map_encode_audio, fn_kwargs = {"model_name": encoder_model, "model": encoder}, \
                            batched=True, batch_size=64)



    ######## TRANSCRIBING ##########
    # if logger:
    #     logger.info(f"Loading transcriber model: {transcriber_model}")
    # if transcriber_model != "facebook/wav2vec2-xlsr-53-espeak-cv-ft":
    #     raise NotImplementedError("Model not supported")

    # # processor = AutoProcessor.from_pretrained(model_name)
    # transcriber_processor = Wav2Vec2Processor.from_pretrained(transcriber_model)
    # # "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    # print(f"Preparing dataset with processor for {transcriber_model}")
    # dataset = dataset.map(prepare_dataset, fn_kwargs = {"processor": transcriber_processor} , \
    #                       batched=True, batch_size=batch_size, \
    #                       num_proc=4, writer_batch_size=100, keep_in_memory=False)

    
    # model = Wav2Vec2ForCTC.from_pretrained(transcriber_model)
    # model.eval()
    # model.to(device)
    # print(f"Transcribing {len(dataset)} samples")
    # dataset = dataset.map(map_transcribe, fn_kwargs = {"transcriber_model": model, "processor": transcriber_processor}, \
    #                         batched=True, batch_size=batch_size, remove_columns=["input_values", "lengths"])
    


    ############ Find SSL units #############

    # Extract embeddings
    ssl_processor = AutoFeatureExtractor.from_pretrained(ssl_unit_model)
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    ssl_model = Wav2Vec2ForPreTraining.from_pretrained(ssl_unit_model)
    ssl_model.eval()
    ssl_model.to(device)

    # Prepare dataset
    dataset = dataset.map(prepare_dataset, fn_kwargs = {"processor": ssl_processor} , \
                          batched=True, batch_size=batch_size, \
                          num_proc=2, writer_batch_size=50, keep_in_memory=False, remove_columns=["signal"])

    dataset = dataset.map(map_extract_embeddings, fn_kwargs = {"model": ssl_model, "layer": 8}, \
                            batched=True, batch_size=batch_size, remove_columns=["input_values"])
    
    # Extract phoneme segment embeddings
    dataset = dataset.map(map_extract_phoneme_segment_embeddings, fn_kwargs = {"segment_size": 100}, \
                            batched=True, batch_size=batch_size, remove_columns=["ssl_all_full_reps"])
    
    # Compute Kmeans reps
    
    kmeans = KMeansOnUnits(output_dir = kmeans_dir)
    if kmeans.trained == False: # This is true if the model has already been saved to kmeans_dir
        raise NotImplementedError("Kmeans model not found")
    else:
        print("WARNING: Using pre-trained Kmeans model")

    dataset = dataset.map(map_compute_kmeans_reps, fn_kwargs = {"kmeans": kmeans}, \
                            batched=True, batch_size=1000, remove_columns=["ssl_all_segment_reps", "lengths"]) # num_proc=4, writer_batch_size=100, keep_in_memory=False

    # Write HF dataset to disk
    dataset.to_json(outfile)
    if logger:
        logger.info(f"Dataset written to {outfile}")
    print(f"Dataset written to {outfile}")

    return dataset


    
if __name__ == "__main__":
    args = parse_args()
    
    transcriber_model = args.transcriber_model
    encoder_model = args.encoder_model

    ssl_model = args.ssl_model
    kmeans_dir = args.kmeans_dir

    dataset_dir = args.dataset_name
    per_lang = args.per_lang # This is number of samples per language for VL107 and FLEURS, 
                             # number of total samples over all accents for EDACC
                             # and number of samples per accent for CV 
    

    lang = args.lang
    batch_size = args.batch_size
    output_dir = args.output_dir

    log_file = args.log_file
    
    main(transcriber_model, encoder_model, ssl_model, kmeans_dir, dataset_dir, per_lang, lang, batch_size, output_dir, log_file)