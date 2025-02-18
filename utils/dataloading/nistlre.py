'''
Load NISTLRE dataset
dataset = load_nistlre(num_samples = num_samples)
This results in a HF dataset:
dataset: {"signal": [np.array], 
        "lang": [str], 
        "accent": [str],
        "audio_file": [str]}, 
    where lang uses the VL107 code, accent is the
accent of each speaker in the dataset.
"signal" contains segments of length clip_length = 6.0s, sampled at 16kHz.
'''

import os, sys, csv
import random
from collections import defaultdict
from pandas import DataFrame as df
import torch
import torchaudio
from datasets import Dataset, concatenate_datasets
# from pydub import AudioSegment
from lhotse import Recording, MonoCut
from lang_code_maps import vl107_to_nistlre_map


def load_nistlre_lang(lang = None, per_accent = None):

    vl107_to_nistlre = vl107_to_nistlre_map()
    nistlre_to_vl107 = {v: k for k, v in vl107_to_nistlre.items()}
    
    # Now, lang is always the VL107 code, and lang_nistlre is the NISTLRE code
    if lang not in vl107_to_nistlre and lang not in nistlre_to_vl107:
        print(f"Language code {lang} not in NISTLRE")
        return None
    
    if lang in nistlre_to_vl107:
        lang_nistlre = lang
        lang = nistlre_to_vl107[lang]

    else:
        lang_nistlre = vl107_to_nistlre[lang]


    lang2accents = {
        "ara": ["apc", "acm", "ary", "arz"],
        "eng": ["gbr", "usg"],
        "spa": ["car", "eur", "lac"],
        "zho": ["cmn", "nan"],
        "por": ["brz"],
        "qsl": ["pol", "rus"], # Not dealing with this because Polish and Russian are different languages.
    }

    accents = lang2accents[lang_nistlre]
    lang_accents = [f"{lang_nistlre}-{accents[i]}" for i in range(len(accents))]
    print(f"Loading NISTLRE for {lang_nistlre} with accents {accents}")

    per_accent_num_samples = dict()
    all_data = []
    with open("/exp/jvillalba/corpora/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets/docs/train_info.tab") as f:
        for line in f:
            lang_accent, path = line.strip().split()[0], line.strip().split()[1]
            if lang_accent.strip() not in lang_accents:
                continue
            if per_accent and per_accent_num_samples.get(lang_accent.strip(), 0) >= per_accent:
                continue
            data_path = "/exp/jvillalba/corpora/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets/data/train/"
            filepath = f"{data_path}{lang_accent.strip()}/{path.strip()}"
            recording = Recording.from_file(filepath)
            cut = MonoCut(recording=recording, start=0.0, duration=recording.duration, id = "rec", channel = 0)
            cut = cut.resample(16000)
            segment = cut.load_audio()[0]
            if segment.shape[0] < 6*16000:
                continue
            # Chunk into uniform windows of K seconds
            K = 6
            for i in range(0, len(segment), K*16000):
                if i+K*16000 > len(segment):
                    break
                if per_accent and per_accent_num_samples.get(lang_accent.strip(), 0) >= per_accent:
                    break
                all_data.append({"signal": segment[i:i+K*16000], "lang": lang, \
                                 "accent": lang_accent.strip()[-3:], "audio_file": filepath})
                per_accent_num_samples[lang_accent.strip()] = per_accent_num_samples.get(lang_accent.strip(), 0) + 1
                

            print(f"Loaded {len(all_data)} segments")

    
    print(f"Loaded {len(all_data)} segments")
    print(f"Sample: {all_data[0]}")

    all_data = {"signal": [f["signal"] for f in all_data], \
        "accent": [f["accent"] for f in all_data], \
        "lang": [f["lang"] for f in all_data], \
        "audio_file": [f["audio_file"] for f in all_data]}
    
    return Dataset.from_dict(all_data)


def load_nistlre(lang = None, per_accent = None):
    '''
    Load NISTLRE dataset
    '''
    if lang is not None:
        return load_nistlre_lang(lang = lang, per_accent = per_accent)
    
    # Load all languages
    dataset = Dataset.from_dict({"signal": [], "lang": [], "accent":[], "audio_file": []})
    all_datasets = []
    langs = {"ara", "eng", "spa", "zho", "por"}
    for lang in langs:
        lang_dataset = load_nistlre_lang(lang = lang, per_accent = per_accent)
        all_datasets.append(lang_dataset)
    
    dataset = concatenate_datasets(all_datasets)
    # dataset = dataset.shuffle()

    return dataset