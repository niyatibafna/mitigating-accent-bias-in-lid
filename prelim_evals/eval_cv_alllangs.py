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

def load_cv_langs(per_lang=None):
    
    files = []
    
    langs = {"hi", "en", "as", "yo", "ha", "id", "ga-IE"}
    accents = {'indian', 'singapore', 'scotland', 'us', 'canada', 'wales', 'england', 'philippines', 'african', 'newzealand', 'ireland', 'malaysia', 'hongkong', 'australia'}
    # Train tsv and clips folder based on lang
    data = []
    for lang in langs.union(accents):
        print(f"Loading CV reps for {lang}")

        en_accent = None
        if lang in accents:
            en_accent = lang

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
        print(f"Loading CV reps for {lang}")
        print(f"Train tsv: {train_tsv}")
        print(f"Clips folder: {clips_folder}")

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

        # print(files)
        print("Loading audio files...")
        for audio in files:
            filename = os.path.join(clips_folder, audio)
            signal = language_id.load_audio(filename)
            if signal.shape[0] < 6*16000:
                continue
            K = 6
            for i in range(0, len(signal), K*16000):
                if i+K*16000 > len(signal):
                    break
                if en_accent:
                    lang = f"en_{en_accent}"
                data.append({"signal": signal[i:i+K*16000], "lang": lang, "filename": filename})
            if per_lang is not None and len(data) >= per_lang:
                break

        print(f"Loaded {len(data)} samples in total.")

    data = {"signal": [f["signal"] for f in data], "lang": [f["lang"] for f in data], "filename": [f["filename"] for f in data]}
    return Dataset.from_dict(data)


dataset = load_cv_langs(per_lang=5000)

print(f"Dataset size: {len(dataset)}")

batch_size = 16
correct_langid = {
    "en": "en: English",
    "hi": "hi: Hindi",
    "as": "as: Assamese",
    "yo": "yo: Yoruba",
    "ha": "ha: Hausa",
    "id": "id: Indonesian",
    "ga-IE": "ga-IE: Irish"
}
preds = defaultdict(lambda: defaultdict(int))
preds_record = []
for data in dataset.iter(batch_size = batch_size):
    signals = torch.stack([torch.tensor(signal) for signal in data["signal"]])
    predictions = language_id.classify_batch(signals)
    for lang, filename, prediction in zip(data["lang"], data["filename"], predictions[3]):
        preds[lang][prediction] += 1
        correct_lang = correct_langid[lang]
        preds_record.append({"lang": lang, "filename": filename, "prediction": prediction, "misclassified": prediction != correct_lang})
    

preds_df = df(preds)
print(preds_df)
preds_df.to_csv("cv_all-langs_confusion_matrix.csv")
preds_record_df = df(preds_record)
print(preds_record_df)
preds_record_df.to_csv("cv_all-langs-predictions.csv")


# Print accuracy for each L2 accent and speaker
for lang in preds:
    if lang.startswith("en"):
        correct_langid[lang] = "en: English"
    total = sum(preds[lang].values())
    correct = preds[lang][correct_langid[lang]]
    print(f"Lang: {lang}, Accuracy: {correct/total}")

