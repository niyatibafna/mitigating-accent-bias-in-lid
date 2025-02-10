'''
Load vl107 dataset
dataset = load_vl107(per_lang = per_lang)
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

def load_audio_file(vl107_dir, audio, lang):

    audio_path = os.path.join(vl107_dir, audio)
    # data.append({"signal": audio_path, "lang": lang, "audio_file": audio_path})
    # data = {"signal": [f["signal"] for f in data], "lang": [f["lang"] for f in data], "audio_file": [f["audio_file"] for f in data]}
    # return Dataset.from_dict(data).cast_column("signal", Audio(sampling_rate=16_000))

    
    signal, sr = torchaudio.load(audio_path)
    signal = signal.squeeze()
    # logger.info(f"Signal shape: {signal.shape}, Sampling rate: {sr}")
    if sr != 16000:
        signal = torchaudio.transforms.Resample(sr, 16000)(signal)

    # data.append({"signal": signal, "lang": lang, "audio_file": audio})
    clip_length = 6 # 6s

    if signal.shape[0] < clip_length*16000:
        return None
    
    # Split the audio into clips of 6s
    signal = signal[:clip_length*16000*(signal.shape[0]//(clip_length*16000))]
    signal = signal.reshape(-1, clip_length*16000)
    audio_data = [{"signal": s, "lang": lang, "audio_file": audio} for s in signal]

    # audio_data = []
    # for i in range(0, len(signal), clip_length*16000):
    #     if i+clip_length*16000 > len(signal):
    #         break
    #     audio_data.append({"signal": signal[i:i+clip_length*16000], "lang": lang, "audio_file": audio})

    return audio_data


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



def load_vl107_lang(lang = None, per_lang = None, vl107_dir = "/exp/jvillalba/corpora/voxlingua107"):
    '''Load audio filepaths from VL107 dataset'''
    vl107_dir = f"{vl107_dir}/{lang}"
    if not os.path.exists(vl107_dir):
        print(f"Directory {vl107_dir} not found")
        return None

    files = [f for f in os.listdir(vl107_dir) if f.endswith(".wav")]
    random.shuffle(files)
    if per_lang is not None:
        files = files[:per_lang]

    lang_dataset = Dataset.from_dict({"signal": [f"{vl107_dir}/{audio}" for audio in files], "lang": [lang]*len(files), "accent": ["-"]*len(files)}).cast_column("signal", Audio(sampling_rate=16_000))
    
    # lang_dataset = lang_dataset.map(map_create_audio_chunks, batched=True, batch_size = 100, \
    #                                 num_proc = 4, keep_in_memory=False, writer_batch_size=10)
    lang_dataset = lang_dataset.map(map_create_audio_chunks, batched=True, batch_size = 100)
    

    # lang_dataset = Dataset.from_dict({"signal": [f["signal"]["array"] for f in lang_dataset], \
    #                                   "lang": [f["lang"] for f in lang_dataset], \
    #                                     "audio_file": [f["signal"]["path"] for f in lang_dataset],\
    #                                         "accent": ["-"]*len(lang_dataset)})
    

    return lang_dataset




def load_vl107_lang_audios(lang = None, per_lang = None, vl107_dir = "/exp/jvillalba/corpora/voxlingua107"):
    '''Load audio files from VL107 dataset'''
    vl107_dir = f"{vl107_dir}/{lang}"
    if not os.path.exists(vl107_dir):
        print(f"Directory {vl107_dir} not found")
        return None

    files = [f for f in os.listdir(vl107_dir) if f.endswith(".wav")]
    random.shuffle(files)
    if per_lang is not None:
        files = files[:per_lang]

    # !--------- Currently training on shorter (e.g. 6s) clips? -------------!
    print(f"Loading audio files for {lang} from {vl107_dir}")
    data = []

    audio_data = [load_audio_file(vl107_dir, audio, lang) for audio in files]
    for audio in audio_data:
        if audio is not None:
            data.extend(audio)

    # for audio in files:
    #     audio_data = load_audio_file(vl107_dir, audio, lang, per_lang = per_lang)
    #     if audio_data is not None:
    #         data.extend(audio_data)
        

    print(f"Number of audio clips for {lang}: {len(data)}")
    data = {"signal": [f["signal"] for f in data], "lang": [f["lang"] for f in data], "accent":"-", "audio_file": [f["audio_file"] for f in data]}
    return Dataset.from_dict(data)

    

def load_vl107(per_lang = None, lang=None, vl107_dir = "/exp/jvillalba/corpora/voxlingua107"):
    '''
    per_lang: Number of audio clips to be loaded per language
    '''
    if lang is not None:
        print(f"Loading audio files for {lang} from {vl107_dir}")
        return load_vl107_lang(lang = lang, per_lang = per_lang, vl107_dir = vl107_dir)

    print(f"Loading all languages from {vl107_dir}")

    dataset = Dataset.from_dict({"signal": [], "lang": [], "accent":[], "audio_file": []})
    for lang in os.listdir(vl107_dir):
        lang_dir = os.path.join(vl107_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        print(f"Loading audio files for {lang} from {lang_dir}")
        start_time = time.time()
        lang_dataset = load_vl107_lang(lang = lang, per_lang = per_lang, vl107_dir = vl107_dir)
        print(f"Time taken to load {lang}: {time.time()-start_time}")
        load_time = time.time()
        dataset = concatenate_datasets([dataset, lang_dataset])
        print(f"Time taken to concatenate {lang}: {time.time()-load_time}")
    
    dataset = dataset.shuffle()

    return dataset

