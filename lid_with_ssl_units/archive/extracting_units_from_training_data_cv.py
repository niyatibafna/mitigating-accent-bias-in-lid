import os, sys
from pathlib import Path
import numpy as np
import random
from collections import defaultdict
import torch
import torchaudio
from transformers import AutoProcessor, WavLMModel
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining, Wav2Vec2Processor
from datasets import load_dataset, Dataset, Audio
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="Model name")
    parser.add_argument("--layer", type=int, default=8, help="Layer to extract representations from")
    parser.add_argument("--dataset_dir", type=str, default="/exp/jvillalba/corpora/voxlingua107", help="Directory containing audio files")
    parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
    parser.add_argument("--lang", type=str, default=None, help="Language to extract audio files from")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory")
    return parser.parse_args()



def load_cv_lang(lang = None, en_accent = None, per_lang = None):
    files = []
    
    if en_accent:
        accents = {'indian', 'singapore', 'scotland', 'us', 'canada', 'wales', 'england', 'philippines', 'african', 'newzealand', 'ireland', 'malaysia', 'hongkong', 'australia'}
        if en_accent not in accents:
            print(f"Accent {en_accent} not found in CommonVoice dataset")
            return None
        lang = "en_"+en_accent
    
    # Train tsv and clips folder based on lang
    train_tsv = f"/exp/nbafna/data/commonvoice/cv/cv-corpus-19.0-2024-09-13/{lang}/train.tsv"
    if lang == "en" or en_accent:
        train_tsv = "/export/common/data/corpora/ASR/commonvoice/en/train.tsv"

    clips_folder = f"/exp/nbafna/data/commonvoice/cv/cv-corpus-19.0-2024-09-13/{lang}/clips/"
    if lang == "en" or en_accent:
        clips_folder = "/export/common/data/corpora/ASR/commonvoice/en/clips/"

    # Find accent_idx
    header = open(train_tsv).readline().strip().split("\t")
    path_idx = header.index("path")
    accent_idx = None
    if "accent" in header:
        accent_idx = header.index("accent") 
    print(f"Path index: {path_idx}, Accent index: {accent_idx}")

    for line in open(train_tsv):
        if len(line.strip().split("\t")) < 8:
            continue
        audio = line.strip().split("\t")[path_idx]
        if ".mp3" not in audio:
            continue
        if en_accent:
            accent = line.strip().split("\t")[accent_idx]
            if accent != en_accent:
                continue

        files.append(audio)

    random.shuffle(files)
    if per_lang is not None:
        files = files[:per_lang]

    print(files)
    print("Loading audio files...")
    data = []
    for audio in files:
        audio_path = os.path.join(clips_folder, audio)
        
        data.append({"signal": audio_path, "lang": lang, "audio_file": audio_path})

    data = {"signal": [f["signal"] for f in data], "lang": [f["lang"] for f in data], "audio_file": [f["audio_file"] for f in data]}
    return Dataset.from_dict(data).cast_column("signal", Audio(sampling_rate=16_000))

def load_vl107_lang(lang = None, per_lang = None, vl107_dir = "/exp/jvillalba/corpora/voxlingua107"):
    '''Load audio files from VL107 dataset'''
    vl107_dir = f"{vl107_dir}/{lang}"
    if not os.path.exists(vl107_dir):
        print(f"Directory {vl107_dir} not found")
        return None

    files = [f for f in os.listdir(vl107_dir) if f.endswith(".wav")]
    random.shuffle(files)
    if per_lang is not None:
        files = files[:per_lang]

    # !--------- Consider training on shorter (e.g. 6s) clips? -------------!
    print(f"Loading audio files for {lang} from {vl107_dir}")
    data = []
    for audio in files:
        audio_path = os.path.join(vl107_dir, audio)
        data.append({"signal": audio_path, "lang": lang, "audio_file": audio_path})
        # signal, sr = torchaudio.load()
        # if sr != 16000:
        #     signal = torchaudio.transforms.Resample(sr, 16000)(signal)

        # data.append({"signal": signal, "lang": lang, "audio_file": audio})
        # if signal.shape[0] < 6*16000:
        #     continue
        # K = 6
        # for i in range(0, len(signal), K*16000):
        #     if i+K*16000 > len(signal):
        #         break
        #     data.append({"signal": signal[i:i+K*16000], "lang": lang, "audio": audio})

    data = {"signal": [f["signal"] for f in data], "lang": [f["lang"] for f in data], "audio_file": [f["audio_file"] for f in data]}
    return Dataset.from_dict(data).cast_column("signal", Audio(sampling_rate=16_000))


def prepare_dataset(batch):
    global processor

    # array_values = [np.array(signal).squeeze() for signal in batch["signal"]]
    # batch["input_values"] = processor(array_values, sampling_rate = 16_000, \
    #                                 padding = True, \
    #                                 return_tensors="pt").input_values
    # batch["lengths"] = [array.shape[0] for array in array_values]
    # print(type(batch["input_values"]))
    # print(batch["input_values"].shape)
    array_values = [item["array"] for item in batch["signal"]]
    batch["input_values"] = processor(array_values, sampling_rate = 16_000 , \
                                    padding = True, \
                                    return_tensors="pt").input_values
    batch["lengths"] = [array.shape[0] for array in array_values]
    return batch

def get_timit():
    timit_dir = "/export/common/data/corpora/LDC/LDC93S1/TIMIT/TRAIN"
    # Get a list of all wav files recursively
    wav_files = sorted(list(Path(timit_dir).rglob("*.WAV")))
    print(f"Wav files: {wav_files[:10]}")
    # Convert to string
    MAX_FILES = 10
    if MAX_FILES:
        wav_files = [str(wav_file) for wav_file in wav_files][:MAX_FILES]
    else:
        wav_files = [str(wav_file) for wav_file in wav_files]
    
    print(f"Wav files: {wav_files[:10]}")
    text_files = [wav_file.replace(".WAV", ".TXT") for wav_file in wav_files]
    timestamp_files = [wav_file.replace(".WAV", ".PHN") for wav_file in wav_files]

    print(f"Wav files to load audio: {wav_files[:10]}")
    langs = ["en"]*len(wav_files)
    timit = Dataset.from_dict({"signal": wav_files, "audio_file": text_files, "lang": langs}).cast_column("signal", Audio(sampling_rate=16_000))
    # print(f"Example: {timit[0]['audio']}")
    # print(f"Number of audio files: {len(timit)}")
    return timit

def collate_fn(batch):
    return {
        "input_values": torch.stack([torch.tensor(item["input_values"]) for item in batch]),
        # "wav_files": [item["wav_file"] for item in batch],
        # "text_files": [item["text_file"] for item in batch],
        # "timestamp_files": [item["timestamp_file"] for item in batch],
        "lengths": [item["lengths"] for item in batch],
        "audio_file": [item["audio_file"] for item in batch]
    }


def extract_embeddings(output_dir, model = None, dataloader = None, layer = None):
    '''Load embeddings from disk if already computed, otherwise compute and save them'''

    if os.path.exists(os.path.join(output_dir, f"all_full_reps.pkl")):
        with open(os.path.join(output_dir, f"all_full_reps.pkl"), "rb") as f:
            all_full_reps = pkl.load(f)
        with open(os.path.join(output_dir, f"all_lengths.pkl"), "rb") as f:
            all_lengths = pkl.load(f)
        with open(os.path.join(output_dir, f"all_audio_files.pkl"), "rb") as f:
            all_audio_files = pkl.load(f)

        return all_full_reps, all_lengths, all_audio_files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_full_reps = [] # Representations of each sample 
    all_lengths = [] # Length of each sample
    all_audio_files = []

    model.eval()
    #### TESTING ONLY, ON CPU ####
    # model.to("cpu")
    ###############################

    for batch in tqdm(dataloader):
        # print(batch)
        # print(batch["input_values"].shape)
        # print(batch["text_files"])
        # batch_size, sequence_length, -1

        # batch["input_values"] = batch["input_values"].to("cpu")
        with torch.no_grad():
            # print(f"Sample input_values: {batch['input_values']}")
            # print(f"Precision: {batch['input_values'].dtype}")
            # print(f"Model precision: {model.get_input_embeddings().weight.dtype}")
            outputs = model(batch["input_values"], output_hidden_states=True)
            outputs_layer = outputs.hidden_states[layer] # (1st output is embedding layer)
            # print(f"Sample output_layer rep: {outputs_layer}")
            # print(f"outputs.hidden_states: {outputs.hidden_states}")

            # Shape of ouputs_layer: batch_size, sequence_length, hidden_size
            for i in range(len(batch["audio_file"])):
                length = batch["lengths"][i]
                all_lengths.append(length)
                true_sequence_length = length//320 + 1 # going from samples to frames, 1 frame = 20ms, 320 samples = 20ms  since sampling rate is 16kHz
                rep = outputs_layer[i][:true_sequence_length, :] # We only want the representations for the original length
                all_full_reps.append(rep)
                # print(f"Length: {length}")
                # print(f"True sequence length: {true_sequence_length}")
                # print(f"Og rep shape: {outputs_layer[i].shape}")
                # print(f"Rep shape: {rep.shape}")

            all_audio_files.extend(batch["audio_file"])


    print(f"Number of representations: {len(all_full_reps)}")
    print(f"Example representation: {all_full_reps[0].shape}, {all_full_reps[0]}")
    # Save the representations and labels
    with open(os.path.join(output_dir, f"all_full_reps.pkl"), "wb") as f:
        pkl.dump(all_full_reps, f)
    with open(os.path.join(output_dir, f"all_lengths.pkl"), "wb") as f:
        pkl.dump(all_lengths, f)
    with open(os.path.join(output_dir, f"all_audio_files.pkl"), "wb") as f:
        pkl.dump(all_audio_files, f)

    return all_full_reps, all_lengths, all_audio_files
    

def extract_phoneme_segment_embeddings(output_dir, all_full_reps = None, segment_size = 100):

    '''Extract phoneme segment embeddings from the full representations. segment_size is in ms
    all_full_reps: List of tensors of shape (sequence_length, hidden_size). Each tensor may have different sequence length.
    '''

    if os.path.exists(os.path.join(output_dir, f"all_segment_reps.pkl")):
        with open(os.path.join(output_dir, f"all_segment_reps.pkl"), "rb") as f:
            all_segment_reps = pkl.load(f)
        return all_segment_reps
    
    # pool_size is the number of frames in a segment
    pool_size = segment_size//20 # 20ms per frame
    
    # divisible_size = all_full_reps[0].shape[0] - all_full_reps[0].shape[0] % pool_size
    # all_full_reps = all_full_reps[:, :divisible_size, :]
    # all_full_reps = all_full_reps.reshape(all_full_reps.shape[0], pool_size, -1, all_full_reps.shape[-1])
    # all_segment_reps = all_full_reps.mean(axis=1)
    # print(f"all_segment_reps.shape: {all_segment_reps.shape}")

    all_segment_reps = []
    # We want to average the representations over 100ms segments i.e. 5 frames
    for idx in range(len(all_full_reps)):
        full_rep = all_full_reps[idx]
        seg_reps = []
        # print(f"reps.shape: {reps.shape}")
        # Wav2Vec2 timestamps --> sequence index 
        # Each frame is 20ms
        # Split the sequence into 100ms segments
        divisible_size = full_rep.shape[0] - full_rep.shape[0] % pool_size
        full_rep = full_rep[:divisible_size, :]
        full_rep = full_rep.reshape(-1, pool_size, full_rep.shape[-1])
        seg_reps = full_rep.mean(axis=1)

        # for i in range(num_reps):
        #     ## 1 frame = 20ms, 5 frames = 100ms
        #     start_idx, end_idx = i*5, (i+1)*5
        #     rep = full_rep[start_idx:end_idx, :].mean(dim=0)
        #     # print(f"rep.shape: {rep.shape}")
        #     seg_reps.append(rep)
        all_segment_reps.append(seg_reps)
    
    with open(os.path.join(output_dir, f"all_segment_reps.pkl"), "wb") as f:
        pkl.dump(all_segment_reps, f)

    return all_segment_reps


def main():
    global processor
    args = parse_args()
    model_name = args.model_name
    layer = args.layer
    dataset_dir = args.dataset_dir
    per_lang = args.per_lang
    lang = args.lang
    batch_size = args.batch_size
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTraining.from_pretrained(model_name)

    # Load the dataset
    print(f"Loading dataset from {dataset_dir}")
    # train_dataset = load_vl107_lang(lang=lang, per_lang=per_lang, vl107_dir=dataset_dir)
    # train_dataset = load_cv_lang(lang=lang, per_lang=per_lang)
    train_dataset = get_timit()
    train_dataset = train_dataset.map(prepare_dataset, batched=True, batch_size=batch_size, remove_columns=["signal", "lang"])
    # Shape of input
    print(torch.tensor(train_dataset[0]["input_values"]).shape)    

    dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

    # Extract embeddings
    print(f"Extracting embeddings from layer {layer}")
    all_full_reps, _, _ = extract_embeddings(output_dir, model = model, dataloader = dataloader, layer = layer)
    print(f"Number of representations: {len(all_full_reps)}")

    # Extract phoneme segment embeddings
    print("Extracting phoneme segment embeddings")
    all_segment_reps = extract_phoneme_segment_embeddings(output_dir, all_full_reps)
    print(f"Number of segment representations: {len(all_segment_reps)}")


if __name__ == "__main__":

    main()


