'''
Load EdAcc dataset
dataset = load_edacc(num_samples = num_samples)
This results in a HF dataset:
dataset: {"signal": [np.array], 
        "lang": [str], 
        "accent": [str],
        "audio_file": [str]}, 
    where lang is always "en", accent is the
manually annotated/modified accent of each speaker in the dataset.
"signal" contains segments of length clip_length = 6.0s, sampled at 16kHz.
'''

import torch
import os, sys
import json
import random
torch.ones(1).to("cuda")
import torchaudio
from datasets import load_dataset, concatenate_datasets, Dataset, Audio


def stm_reader(stm_path):
    stm_data = []
    with open(stm_path, "r") as f:
        for line in f:
            line_data = {}
            parts = line.split()
            line_data["audio_file"] = parts[0]
            line_data["channel"] = parts[1]
            line_data["speaker"] = parts[2]
            line_data["start_time"] = float(parts[3])
            line_data["end_time"] = float(parts[4])
            line_data["label"] = parts[5]
            line_data["transcript"] = " ".join(parts[6:])
            stm_data.append(line_data)
    return stm_data

def load_edacc(num_samples = None):
    '''
    num_samples: Number of samples to be randomly sampled from all samples (regardless of accent)
                 If None, all samples are loaded.
    '''

    test_stm_path = "/exp/nbafna/data/edacc/edacc_v1.0/test/stm"
    dev_stm_path = "/exp/nbafna/data/edacc/edacc_v1.0/dev/stm"
    data_path = "/exp/nbafna/data/edacc/edacc_v1.0/data"

    audio_files = {}
    for audio_file in os.listdir(data_path):
        audio_files[audio_file[:-4]], sr = torchaudio.load(os.path.join(data_path, audio_file))
        # audio_files[audio_file[:-4]] = language_id.load_audio(os.path.join(data_path, audio_file))
        if sr != 16000:
            audio_files[audio_file[:-4]] = torchaudio.transforms.Resample(sr, 16000)(audio_files[audio_file[:-4]])

    print(f"Loaded {len(audio_files)} audio files")

    speaker2lang = {}
    # linguistic_background = "/exp/nbafna/data/edacc/edacc_v1.0/linguistic_background.csv"
    # with open(linguistic_background, "r") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         speaker2lang[row[1]] = row[12]
    participant2accent_path = "/exp/nbafna/data/edacc/edacc_v1.0/participant2accent.json"
    with open(participant2accent_path, "r") as f:
        speaker2lang = json.load(f)
    print(f"Recorded {len(speaker2lang)} speakers")


    clip_length = 6
    all_data = []
    stm_data = stm_reader(test_stm_path) + stm_reader(dev_stm_path)
    for line in stm_data:
        audio_file = line["audio_file"]
        signal = audio_files[audio_file]
        signal = signal.squeeze().numpy()
        segment = signal[int(line["start_time"]*sr):int(line["end_time"]*sr)]
        # Filter out signals with less than 6 seconds
        if segment.shape[0] < clip_length*16000:
            continue
        # Chunk into uniform windows of K seconds
        
        for i in range(0, len(segment), clip_length*16000):
            if i+clip_length*16000 > len(segment):
                break
            all_data.append({"signal": segment[i:i+clip_length*16000], "lang":"en", \
                "accent": speaker2lang[line["speaker"]], "audio_file": line["audio_file"]})

        # segment = segment[:10*16000]
        # lang = speaker2lang[line["speaker"]]
        # all_data.append({"signal": segment, "lang": lang})

        # if len(all_data)%10 == 0:
        #     print(f"Printing out sample")
        #     print(f"Lang: {lang}")
        #     print(f"Speaker: {line['speaker']}")
        #     print(f"Start and end times: {line['start_time']}, {line['end_time']}")
        #     print(f"Expected length: {int((line['end_time']-line['start_time'])*sr)}")
        #     print(f"Length of audio: {segment.shape}")
    
    print(f"Loaded {len(all_data)} segments")
    print(f"Sample: {all_data[0]}")
    all_langs = set([f["accent"] for f in all_data])
    print(f"Accents: {all_langs}")

    if num_samples is not None:
        all_data = random.sample(all_data, min(len(all_data), num_samples))
    all_data = {"signal": [f["signal"] for f in all_data], \
        "accent": [f["accent"] for f in all_data], \
        "lang": [f["lang"] for f in all_data], \
        "audio_file": [f["audio_file"] for f in all_data]}
    
    return Dataset.from_dict(all_data)

