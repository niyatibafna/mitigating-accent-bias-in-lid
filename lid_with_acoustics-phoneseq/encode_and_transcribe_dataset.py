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


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import os, sys
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/dataloading/")
from dataset_loader import load_lid_dataset
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

    return batch


# def collate_fn(batch):
#     return {
#         "input_values": torch.stack([torch.tensor(item["input_values"]) for item in batch]),
#         # "wav_files": [item["wav_file"] for item in batch],
#         # "text_files": [item["text_file"] for item in batch],
#         # "timestamp_files": [item["timestamp_file"] for item in batch],
#         "lengths": [item["lengths"] for item in batch],
#         "lang": [item["lang"] for item in batch],
#         "audio_file": [item["audio_file"] for item in batch],
#         "accent": [item["accent"] for item in batch],
#     }



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
    

def main(transcriber_model, encoder_model, dataset_dir, per_lang, lang, batch_size, output_dir, log_file=None):

    global logger

    outfile = os.path.join(output_dir, f"{lang}_transcriptions.jsonl")
    # Load the dataset if it exists
    if os.path.exists(outfile):
        logger.info(f"Dataset {outfile} already exists")
        dataset = load_dataset("json", data_files = outfile)

        return dataset

    
    logger = get_logger(log_file)
    
        # Load the dataset
    if logger:
        logger.info(f"Loading dataset: {dataset_dir}")

    dataset = load_lid_dataset(dataset_dir, lang = lang, per_lang = per_lang)
    if dataset is None:
        if logger:
            logger.info(f"Dataset {dataset_dir} for lang {lang} not found")
        return None

    # Load the model
    if encoder_model == "ecapa-tdnn":
        # Load the model
        if logger:
            logger.info(f"Loading model: {encoder_model}")
        from speechbrain.inference.classifiers import EncoderClassifier

        encoder = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"})
        dataset = dataset.map(map_encode_audio, fn_kwargs = {"model_name": encoder_model, "model": encoder}, \
                            batched=True, batch_size=batch_size)



    ######## TRANSCRIBING ##########
    if logger:
        logger.info(f"Loading transcriber model: {transcriber_model}")
    if transcriber_model != "facebook/wav2vec2-xlsr-53-espeak-cv-ft":
        raise NotImplementedError("Model not supported")

    # processor = AutoProcessor.from_pretrained(model_name)
    transcriber_processor = Wav2Vec2Processor.from_pretrained(transcriber_model)
    # "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    
    dataset = dataset.map(prepare_dataset, fn_kwargs = {"processor": transcriber_processor} , \
                          batched=True, batch_size=batch_size, \
                            remove_columns=["signal"])

    
    model = Wav2Vec2ForCTC.from_pretrained(transcriber_model)
    model.eval()
    model.to(device)
    dataset = dataset.map(map_transcribe, fn_kwargs = {"transcriber_model": model, "processor": transcriber_processor}, \
                            batched=True, batch_size=batch_size, \
                            remove_columns=["input_values"])
    
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

    dataset_dir = args.dataset_name
    per_lang = args.per_lang # This is number of samples per language for VL107 and FLEURS, 
                             # number of total samples over all accents for EDACC
                             # and number of samples per accent for CV 
    lang = args.lang
    batch_size = args.batch_size
    output_dir = args.output_dir

    log_file = args.log_file
    
    main(transcriber_model, encoder_model, dataset_dir, per_lang, lang, batch_size, output_dir, log_file)