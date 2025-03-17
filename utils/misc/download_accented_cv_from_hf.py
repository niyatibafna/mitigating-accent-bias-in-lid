from datasets import load_dataset, Dataset, DatasetDict, Audio
from collections import defaultdict, Counter

import numpy as np
import torch
import torchaudio
import sys

def map_resample(example):

    waveform = torch.tensor(example["audio"]["array"])
    example["audio"]["array"] = torchaudio.transforms.Resample(48_000, 16_000)(waveform)
    example["audio"]["array"] = example["audio"]["array"].numpy().tolist()
    example["audio"]["sampling_rate"] = 16_000
    return example

def download_accented_dataset(lang):
    print(f"Processing {lang}")
    ds = load_dataset("mozilla-foundation/common_voice_13_0", lang, streaming=True)
    print(f"Converting to dataset")
    filtered_ds = ds['train'].filter(lambda x: x.get('accent'))
    majority_accents = {
        "de": "Deutschland Deutsch",
        "fr": "Français de France",
        "es": "México",
        "it": "Tendente al siculo, ma non marcato",
    }
    if lang in majority_accents:
        majority_accent = majority_accents[lang]
        filtered_majority = filtered_ds.filter(lambda x: x['accent'] == majority_accent)
        filtered_minority = filtered_ds.filter(lambda x: x['accent'] != majority_accent)

        majority_dataset = filtered_majority.take(500)
        minority_dataset = filtered_minority.take(4500)
        subset = list(majority_dataset) + list(minority_dataset)
    else:
        subset = filtered_ds.take(5000)
        subset = list(subset)
        
    dataset = Dataset.from_list(subset)
    # print(f"Printing accents...")
    # accents = [x['accent'] for x in dataset]
    # print(f"Accents: {Counter(accents)}")
    # print(f"Total: {len(accents)}")

    # Resample the audio to 16kHz
    print(f"Resampling...")
    dataset = dataset.map(map_resample)

    # Rename columns
    
    formatted_dataset = dataset.rename_column("audio", "signal")
    formatted_dataset = formatted_dataset.rename_column("locale", "lang")
    formatted_dataset = formatted_dataset.rename_column("path", "audio_file")
    formatted_dataset = formatted_dataset.rename_column("sentence", "text_transcription")

    formatted_dataset = formatted_dataset.remove_columns([c for c in formatted_dataset.column_names \
                                        if c not in ["signal", "accent", "lang", "audio_file", \
                                                     "client_id", "text_transcription"]])

    # Save the dataset
    formatted_dataset.save_to_disk(f"/exp/nbafna/data/commonvoice/accented_data/{lang}/{lang}_accented_samples-5k")


if __name__ == "__main__":
    lang = sys.argv[1]
    download_accented_dataset(lang)