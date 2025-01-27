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
import os, sys
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/")
from lang_code_maps import vl107_to_fleurs_map


def load_phoneseq_dataset_lang(dataset_dir = None, lang = None, per_lang = None):
    '''
    Load transcribed dataset for a specific lang
    '''
    use_label_lang = lang
    if "fleurs" in dataset_dir:
        if "_" not in lang: # This is not a FLEURS code
            if lang not in vl107_to_fleurs_map():
                print(f"Language code {lang} not in FLEURS")
                return None
            lang = vl107_to_fleurs_map()[lang]

    dataset = Dataset.from_csv(f"{dataset_dir}/{lang}.csv", column_names = ["audio_file", "lang", "accent", "phone_sequence"])
    original_length = len(dataset)
    dataset = dataset.filter(lambda x: x["phone_sequence"] is not None)
    print(f"Filtered out {original_length - len(dataset)} samples with None phone sequences")
    # Space-separate the phone sequences, and use the label lang 
    dataset = dataset.map(lambda x: {"phone_sequence": x["phone_sequence"].strip().split(" "), "lang": use_label_lang, "accent": x["accent"], "audio_file": x["audio_file"]})
    if per_lang is not None:
        dataset = dataset.shuffle(seed = 42).select(range(per_lang))
    return dataset


def convert_to_target_code(dataset_name, dataset, target_code_type):
    '''
    Convert lang codes to target_code_type
    '''
    if "fleurs" in dataset_name:
        if target_code_type == "vl107":
            vl107_to_fleurs = vl107_to_fleurs_map()
            fleurs_to_vl107 = {v: k for k, v in vl107_to_fleurs.items()} # This contains all the FLEURS lang codes that are in VL107
            dataset = dataset.filter(lambda x: x["lang"] in fleurs_to_vl107)
            dataset = dataset.map(lambda x: {"phone_sequence": x["phone_sequence"], "lang": fleurs_to_vl107[x["lang"]], "accent": x["accent"], "audio_file": x["audio_file"]})
    return dataset



def load_phoneseq_dataset(dataset_name = None, per_lang = None, lang=None, target_code_type = None):
    '''
    Load trascribed dataset. The transcriptions are already chunked into 6s segments,
    saved in a single csv per lang. The csv contains the following columns:
    audio_file, lang, accent, transcription

    target_code_type: "vl107" or "fleurs". This is relevant if we are loading FLEURS but want to use VL107 language codes. Only this case is supported.
    '''
    # if dataset_name == "fleurs":
    #     dataset_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/transcriptions/fleurs/wav2vec2-xlsr-53-espeak-cv-ft"
    # elif dataset_name == "vl107":
    #     dataset_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/transcriptions/vl107/wav2vec2-xlsr-53-espeak-cv-ft"
    # elif dataset_name == "edacc":
    #     dataset_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/transcriptions/edacc/wav2vec2-xlsr-53-espeak-cv-ft"

    dataset_dir = f"/exp/nbafna/projects/mitigating-accent-bias-in-lid/transcriptions/{dataset_name}/wav2vec2-xlsr-53-espeak-cv-ft"

    if lang is not None:
        return load_phoneseq_dataset_lang(dataset_dir, lang, per_lang)
    
    all_datasets = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".csv"):
            lang = file.split(".")[0]
            print(f"Loading dataset for {lang}")
            dataset = load_phoneseq_dataset_lang(dataset_dir, lang, per_lang)
            all_datasets.append(dataset)

    lid_dataset = concatenate_datasets(all_datasets)

    if target_code_type is not None:
        lid_dataset = convert_to_target_code(dataset_name, lid_dataset, target_code_type)

    return lid_dataset

