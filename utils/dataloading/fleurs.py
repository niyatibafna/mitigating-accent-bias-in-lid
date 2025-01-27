'''
Load FLEURS dataset
dataset = load_fleurs(per_lang = per_lang)
This results in a HF dataset:
dataset: {"signal": [np.array], 
        "lang": [str], 
        "accent": [str],
        "audio_file": [str]}, 
"signal" contains segments of length clip_length = 6.0s, sampled at 16kHz.
'''

import torch
if torch.cuda.is_available():
    torch.ones(1).to("cuda")
import torchaudio
import os, sys
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset, Audio
import time
from multiprocessing import cpu_count
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/")
from lang_code_maps import vl107_to_fleurs_map

def map_create_audio_chunks(batch, clip_length = 6):

    # Rearrange into structure {"signal": audio_arrays, "path": paths, "lang": langs}
    #                                         "accent": ["-"]*len(lang_dataset)})
    
    '''Create audio chunks of 6s from the audio signal'''
    signals = [f["array"] for f in batch["signal"]]
    langs = batch["lang"]
    accents = batch["accent"]
    audio_paths = [f["path"] for f in batch["signal"]]

    # Truncate each signal to a multiple of 6s
    for idx, signal in enumerate(signals):
        if len(signal) % (clip_length*16000) != 0:
            signals[idx] = signal[:clip_length*16000*(len(signal)//(clip_length*16000))]

    # Remove empty or too short signals
    audio_paths = [audio_paths[idx] for idx, signal in enumerate(signals) if len(signal) >= clip_length*16000]
    langs = [langs[idx] for idx, signal in enumerate(signals) if len(signal) >= clip_length*16000]
    accents = [accents[idx] for idx, signal in enumerate(signals) if len(signal) >= clip_length*16000]
    signals = [signal for signal in signals if len(signal) >= clip_length*16000]

    # Create appropriate copies of audio paths and langs
    audio_paths = [audio_paths[idx] for idx, signal in enumerate(signals) for _ in range(len(signal)//(clip_length*16000))]
    langs = [langs[idx] for idx, signal in enumerate(signals) for _ in range(len(signal)//(clip_length*16000))]
    accents = [accents[idx] for idx, signal in enumerate(signals) for _ in range(len(signal)//(clip_length*16000))]
    
    # Split the signals into 6s chunks
    # Flatten signals
    signals = [s for signal in signals for s in signal]
    signals = np.array(signals)
    signals = signals.reshape(-1, clip_length*16000).tolist()

    assert len(signals) == len(langs) == len(audio_paths) == len(accents), "Lengths of signals, langs, accents, and audio_paths do not match"

    chunked_audio_data = {"signal": signals, "lang": langs, "audio_file": audio_paths, "accent": accents}

    return chunked_audio_data




def split_files(lang = None, split = "train"):

    lang_filespath = f"/export/common/data/corpora/fleurs/metadata/{lang}/{split}.tsv"
    with open(lang_filespath, "r") as f:
        files = f.readlines()
    files = [f.strip() for f in files]
    audio_files = [line.split("\t")[1] for line in files if len(line.split("\t")) > 1 and line.split("\t")[1].endswith(".wav")]
    return audio_files

def load_fleurs_lang(lang = None, per_lang = None, split="train", dataset_dir = None):
    '''Load audio filepaths from VL107 dataset'''

    # Check if lang is a VL107 code; if so, convert to FLEURS code
    fl_lang = lang
    if "_" not in lang: # This is not a FLEURS code
        if lang not in vl107_to_fleurs_map():
            print(f"Language code {lang} not in FLEURS")
            return None
        fl_lang = vl107_to_fleurs_map()[lang]

    files = split_files(lang = fl_lang, split = split)

    dataset_dir = os.path.join(dataset_dir, fl_lang, split)

    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} not found")
        return None

    print(f"Loading audio files for {lang} from {dataset_dir}")

    random.shuffle(files)
    if per_lang is not None:
        files = files[:per_lang]

    # Note that the labels preserve the original language code type (VL107 or FLEURS). 
    # We assume that the language code type passed matches with the model training dataset.
    # This is for compatibility with the training dataset and model labels.
    lang_dataset = Dataset.from_dict({"signal": [f"{dataset_dir}/{audio}" for audio in files], "lang": [lang]*len(files), "accent": ["-"]*len(files)}).cast_column("signal", Audio(sampling_rate=16_000))
    
    print(f"Size of dataset for {lang}: {len(lang_dataset)}")
    lang_dataset = lang_dataset.map(map_create_audio_chunks, batched=True, batch_size = 100, \
                                    num_proc = 16, keep_in_memory=False, writer_batch_size=100)
    # lang_dataset = lang_dataset.map(map_create_audio_chunks, batched=True, batch_size = 100)
    

    # lang_dataset = Dataset.from_dict({"signal": [f["signal"]["array"] for f in lang_dataset], \
    #                                   "lang": [f["lang"] for f in lang_dataset], \
    #                                     "audio_file": [f["signal"]["path"] for f in lang_dataset],\
    #                                         "accent": ["-"]*len(lang_dataset)})
    
    print(f"Size of dataset for {lang}: {len(lang_dataset)}")
    return lang_dataset

def load_fleurs(per_lang = None, lang=None, split = "train", dataset_dir = "/export/common/data/corpora/fleurs/"):
    '''
    per_lang: Number of audio clips to be loaded per language
    '''
    if lang is not None:
        return load_fleurs_lang(lang = lang, per_lang = per_lang, split=split, dataset_dir = dataset_dir)

    print(f"Loading all languages from {dataset_dir}")

    dataset = Dataset.from_dict({"signal": [], "lang": [], "accent":[], "audio_file": []})
    all_datasets = []
    langs = os.listdir("/export/common/data/corpora/fleurs/metadata")
    for lang in langs:
        # lang_dir = os.p/\ath.join(dataset_dir, lang, split)
        # if not os.path.isdir(lang_dir):
            # continue
        # print(f"Loading audio files for {lang} from {lang_dir}")
        start_time = time.time()
        lang_dataset = load_fleurs_lang(lang = lang, per_lang = per_lang, split = split, dataset_dir = dataset_dir)
        print(f"Time taken to load {lang}: {time.time()-start_time}")
        load_time = time.time()
        all_datasets.append(lang_dataset)
        print(f"Time taken to concatenate {lang}: {time.time()-load_time}")
    
    dataset = concatenate_datasets(all_datasets)
    dataset = dataset.shuffle()

    return dataset


