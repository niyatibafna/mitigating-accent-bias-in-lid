import os, sys
import random
from collections import defaultdict
from pandas import DataFrame as df
import torch
import torchaudio
from datasets import Dataset
from speechbrain.inference.classifiers import EncoderClassifier

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"}) # , 
# wav_file = "/exp/nbafna/data/l2_arctic/l2arctic_release_v5/ABA/wav/arctic_a0001.wav"
# signal = language_id.load_audio(wav_file)
# print(type(signal))
# print(signal.shape)
# prediction = language_id.classify_batch(signal)
# print(prediction)

# Test on L2 Arctic dataset
def load_l2_arctic(data_folder, per_speaker = None):
    speaker2lang = {
        "ABA": "ara", "SKA": "ara", "YBAA": "ara", "ZHAA": "ara",  # Arabic
        "BWC": "cmn", "LXC": "cmn", "NCC": "cmn", "TXHC": "cmn",  # Mandarin
        "ASI": "hin", "RRBI": "hin", "SVBI": "hin", "TNI": "hin",  # Hindi
        "HJK": "kor", "HKK": "kor", "YDCK": "kor", "YKWK": "kor",  # Korean
        "EBVS": "spa", "ERMS": "spa", "MBMPS": "spa", "NJS": "spa",  # Spanish
        "HQTV": "vie", "PNV": "vie", "THV": "vie", "TLV": "vie"  # Vietnamese
    }
    files = []
    # Load the data
    for folder in os.listdir(data_folder):
        print(f"Processing {folder}")
        # Continue if not a folder
        if not os.path.isdir(os.path.join(data_folder, folder)):
            continue
        if folder not in speaker2lang:
            print(f"WARNING: {folder} not in speaker2lang")
            continue
        speaker_files = []
        for file in os.listdir(os.path.join(data_folder, folder+"/wav")):
            if file.endswith(".wav"):
                signal = language_id.load_audio(os.path.join(data_folder, folder, "wav", file))
                # Filter out signals with less than 2 seconds
                if signal.shape[0] < 2*16000:
                    continue
                speaker_files.append({"signal": signal, "lang": f"{speaker2lang[folder]}"})
        print(f"Found {len(speaker_files)} files for {folder}")
        # Shuffle the files
        random.shuffle(speaker_files)
        if per_speaker is not None:
            speaker_files = speaker_files[:per_speaker]
        files.extend(speaker_files)
        # if len(files) >= 10:
        #     break
    files = {"signal": [f["signal"] for f in files], "lang": [f["lang"] for f in files]}
    return Dataset.from_dict(files)
        

data_folder = "/exp/nbafna/data/l2_arctic/l2arctic_release_v5/"
preds = defaultdict(lambda: defaultdict(int))
dataset = load_l2_arctic(data_folder)

print(f"Dataset size: {len(dataset)}")

for data in dataset.iter(batch_size=3):
    for signal, lang in zip(data["signal"], data["lang"]):
        signal = torch.tensor(signal)
        # print(f"Signal shape: {signal.shape}")
        signal = signal.to(language_id.device)
        prediction = language_id.classify_batch(signal)
        preds[lang][prediction[3][0]] += 1
        # print(f"Lang: {lang}, Pred: {prediction[3]}")

preds_df = df(preds)
print(preds_df)
preds_df.to_csv("l2_arctic_preds.csv")

# Print accuracy for each L2 accent and speaker
for lang in preds:
    total = sum(preds[lang].values())
    correct = preds[lang]["en: English"]
    print(f"Lang: {lang}, Accuracy: {correct/total}")

