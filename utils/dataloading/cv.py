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
from datasets import load_dataset, concatenate_datasets, Dataset, Audio


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



def split_files(lang = None, split = "train", per_accent = None):

    files = defaultdict(list)
    accents = {'indian', 'singapore', 'scotland', 'us', 'canada', 'wales', 'england', 'philippines', 'african', 'newzealand', 'ireland', 'malaysia', 'hongkong', 'australia'}
    for line in open(f"/export/common/data/corpora/ASR/commonvoice/{lang}/{split}.tsv"):
        if len(line.strip().split("\t")) < 8:
            continue
        audio, accent = line.strip().split("\t")[1], line.strip().split("\t")[7]
        if accent not in accents or (not audio.endswith(".mp3") and not audio.endswith(".wav")):
            continue
        files[accent].append(audio)
    
    for accent in accents:
        if accent not in files:
            continue
        random.shuffle(files[accent])
        if per_accent:
            files[accent] = files[accent][:per_accent]
    
    accents = [accent for accent in files for _ in range(len(files[accent]))]
    files = [f for accent in files for f in files[accent]]

    clips_folder = f"/export/common/data/corpora/ASR/commonvoice/{lang}/clips/"
    files = [os.path.join(clips_folder, f) for f in files]

    return files, accents    


def load_cv(lang = "en", per_accent = None, split = "train", dataset_dir = "/export/common/data/corpora/ASR/commonvoice/en/clips"):

    files, accents = split_files(lang=lang, split = split, per_accent = per_accent)

    print("Loading audio files...")
    lang_dataset = Dataset.from_dict({"signal": files, "lang": [lang]*len(files), \
        "accent": accents}).cast_column("signal", Audio(sampling_rate=16_000))

    print(f"Size of dataset for {lang}: {len(lang_dataset)}")
    lang_dataset = lang_dataset.map(map_create_audio_chunks, batched=True, batch_size = 100, \
                                    num_proc = 16, keep_in_memory=False, writer_batch_size=100)
    

    print(f"Size of dataset for {lang}: {len(lang_dataset)}")
    return lang_dataset