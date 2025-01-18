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
from vl107 import load_vl107, load_vl107_lang
from edacc import load_edacc

from tqdm import tqdm

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

def sort_outfile(outfile):
    '''Sort the outfile by audio file. This is so that multiple segments per audio file appear together'''
    with open(outfile, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(",") for line in lines]
    lines = sorted(lines, key = lambda x: x[0])
    with open(outfile, "w") as f:
        for line in lines:
            f.write(",".join(line) + "\n")



def transcribe_audio_files(processor, model, dataloader, output_file, write_every_n_batches = 100):
    '''
    Output is a csv file with the following columns:
    audio_file, lang, accent, transcription
    '''
    global logger

    transcriptions_outputs = []
    model.eval()
    model.to(device)
    out_file_writer = open(output_file, "w")
    for i, batch in tqdm(enumerate(dataloader)):
        logger.info(f"Batch {i}")
        input_values = batch["input_values"].to(device)
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        outputs = list(zip(batch["audio_file"], batch["lang"], batch["accent"], transcription))
        transcriptions_outputs.extend(outputs)
        if i % write_every_n_batches == 0 and i > 0:
            logger.info(f"Writing {len(transcriptions_outputs)} lines to disk")
            for output in transcriptions_outputs:
                out_file_writer.write(",".join(output) + "\n")
            transcriptions_outputs.clear()

    if len(transcriptions_outputs) > 0:
        logger.info(f"Writing {len(transcriptions_outputs)} lines to disk")
        for output in transcriptions_outputs:
            out_file_writer.write(",".join(output) + "\n")
        transcriptions_outputs.clear()
    out_file_writer.close()
    

def main():

    global logger
    args = parse_args()
    model_name = args.model_name
    dataset_dir = args.dataset_name
    per_lang = args.per_lang
    lang = args.lang
    batch_size = args.batch_size
    output_dir = args.output_dir
    write_every_n_batches = 100
    

    logger = get_logger(args.log_file)
    # Load the dataset
    logger.info(f"Loading dataset: {dataset_dir}")

    dataset = load_lid_dataset(dataset_dir, lang = lang, per_lang = per_lang)
    if dataset is None:
        logger.info(f"Dataset {dataset_dir} for lang {lang} not found")
        return

    # Load the model
    logger.info(f"Loading model: {model_name}")
    # processor = AutoProcessor.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    # "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    
    dataset = dataset.map(prepare_dataset, fn_kwargs = {"processor": processor} , \
                          batched=True, batch_size=batch_size, \
                            num_proc = 2, keep_in_memory=False,
                            writer_batch_size = 50,
                            remove_columns=["signal"])

    
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

    outfile = os.path.join(output_dir, f"{lang}.csv")

    logger.info(f"Transcribing audio files to {outfile}...")
    transcribe_audio_files(processor, model, dataloader, outfile, write_every_n_batches)
    sort_outfile(outfile)
    logger.info(f"Transcriptions saved to {outfile}")

    
if __name__ == "__main__":
    main()