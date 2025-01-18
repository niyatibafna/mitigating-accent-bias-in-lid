'''
Load transcribed FLEURS/VL107/Edacc dataset
dataset = load_phoneseq_dataset(dataset_name = "fleurs", per_lang = per_lang)
This results in a HF dataset:
batch: {"phone_sequence": [str], "lang": [str], "accent": [str], "audio_file": [str]}
"phone_sequence" contains the phone sequence of one 6s chunk of audio_file.

This data is the output of 
/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_phoneme_transcription/transcribe_data.py
'''

from datasets import Dataset, concatenate_datasets
import os

def load_phoneseq_dataset_lang(dataset_dir = None, lang = None, per_lang = None):
    '''
    Load transcribed dataset for a specific lang
    '''    
    dataset = Dataset.from_csv(f"{dataset_dir}/{lang}.csv", column_names = ["audio_file", "lang", "accent", "phone_sequence"])
    original_length = len(dataset)
    dataset = dataset.filter(lambda x: x["phone_sequence"] is not None)
    print(f"Filtered out {original_length - len(dataset)} samples with None phone sequences")
    dataset = dataset.map(lambda x: {"phone_sequence": x["phone_sequence"].strip().split(" "), "lang": x["lang"], "accent": x["accent"], "audio_file": x["audio_file"]})
    if per_lang is not None:
        dataset = dataset.shuffle(seed = 42).select(range(per_lang))
    return dataset
    

def load_phoneseq_dataset(dataset_name = None, per_lang = None, lang=None):
    '''
    Load trascribed dataset. The transcriptions are already chunked into 6s segments,
    saved in a single csv per lang. The csv contains the following columns:
    audio_file, lang, accent, transcription
    '''
    if dataset_name == "fleurs":
        dataset_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/transcriptions/fleurs/wav2vec2-xlsr-53-espeak-cv-ft"
    elif dataset_name == "vl107":
        dataset_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/transcriptions/vl107/wav2vec2-xlsr-53-espeak-cv-ft"
    elif dataset_name == "edacc":
        dataset_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/transcriptions/edacc/wav2vec2-xlsr-53-espeak-cv-ft"

    if lang is not None:
        return load_phoneseq_dataset_lang(lang, per_lang, dataset_dir)
    
    all_datasets = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".csv"):
            lang = file.split(".")[0]
            dataset = load_phoneseq_dataset_lang(dataset_dir, lang, per_lang)
            all_datasets.append(dataset)

    lid_dataset = concatenate_datasets(all_datasets)
    return lid_dataset

