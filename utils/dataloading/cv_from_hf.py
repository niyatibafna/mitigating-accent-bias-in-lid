'''
This is for loading the English accented part of the CommonVoice dataset.
This results in a HF dataset:
dataset: {"signal": [np.array], 
        "lang": [str], 
        "accent": [str],
        "audio_file": [str]}, 
    where lang is always "en", accent is the
manually annotated/modified accent of each speaker in the dataset.
"signal" contains segments of length clip_length = 6.0s, sampled at 16kHz.

Usage:
dataset = load_cv_lang(lang = "en", per_accent = 5000)
'''

import torch
import os, sys
import json
import random
import numpy as np
from collections import defaultdict
if torch.cuda.is_available():
    torch.ones(1).to("cuda")
import torchaudio
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, Audio


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


def load_cv_lang(lang, per_lang = None):

    dataset_file = f"/exp/nbafna/data/commonvoice/accented_data/{lang}/{lang}_accented_samples-5k"
    lang_dataset = load_from_disk(dataset_file)
    lang_dataset = lang_dataset.remove_columns(["text_transcription", "audio_file", "client_id"])

    print(f"Size of dataset for {lang}: {len(lang_dataset)}")
    lang_dataset = lang_dataset.map(map_create_audio_chunks, batched=True, batch_size = 100, \
                                    num_proc = 16, keep_in_memory=False, writer_batch_size=100)

    print(f"Size of dataset for {lang}: {len(lang_dataset)}")
    if per_lang:
        indices = random.sample(range(len(lang_dataset)), per_lang)
        lang_dataset = lang_dataset.select(indices)

    return lang_dataset


def load_cv_from_hf(lang = None, per_lang = None):

    langs = ["es", "fr", "de", "it"]

    if lang is not None:
        if lang not in langs:
            print(f"Accented speech for language {lang} not yet downloaded.")
            return None
        return load_cv_lang(lang = lang, per_lang = per_lang)
    
    
    dataset = Dataset.from_dict({"signal": [], "lang": [], "accent":[], "audio_file": []})
    all_datasets = []
    for lang in langs:
        lang_dataset = load_cv_lang(lang = lang, per_lang = per_lang)
        all_datasets.append(lang_dataset)

    dataset = concatenate_datasets(all_datasets)
    return dataset