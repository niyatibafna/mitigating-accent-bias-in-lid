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

# Test on CV dataset
def load_cv(per_accent = None):
    files = {}
    accents = {'indian', 'singapore', 'scotland', 'us', 'canada', 'wales', 'england', 'philippines', 'african', 'newzealand', 'ireland', 'malaysia', 'hongkong', 'australia'}
    for line in open("/export/common/data/corpora/ASR/commonvoice/en/train.tsv"):
        if len(line.strip().split("\t")) < 8:
            continue
        audio, accent = line.strip().split("\t")[1], line.strip().split("\t")[7]
        if accent not in accents:
            continue
        if accent not in files:
            files[accent] = []
        files[accent].append(audio)

    for accent in accents:
        if accent not in files:
            continue
        random.shuffle(files[accent])
        
    # print(files)
    print("Loading audio files...")
    data = []
    clips_folder = "/export/common/data/corpora/ASR/commonvoice/en/clips/"
    for accent in files:
        for audio in files[accent]:
            signal = language_id.load_audio(os.path.join(clips_folder, audio))
            if signal.shape[0] < 6*16000:
                continue
            K = 6
            for i in range(0, len(signal), K*16000):
                if i+K*16000 > len(signal):
                    break
                data.append({"signal": signal[i:i+K*16000], "lang": accent, "filename": os.path.join(clips_folder, audio)})
            if per_accent and len(data) >= per_accent:
                break


    data = {"signal": [f["signal"] for f in data], "lang": [f["lang"] for f in data], "filename": [f["filename"] for f in data]}
    return Dataset.from_dict(data)
        
dataset = load_cv(per_accent = 5000)

print(f"Dataset size: {len(dataset)}")
batch_size = 32

preds = defaultdict(lambda: defaultdict(int))
preds_record = []
for data in dataset.iter(batch_size = batch_size):
    signals = torch.stack([torch.tensor(signal) for signal in data["signal"]])
    predictions = language_id.classify_batch(signals)
    for lang, filename, prediction in zip(data["lang"], data["filename"], predictions[3]):
        preds[lang][prediction] += 1
        correct_lang = "en: English"
        preds_record.append({"lang": lang, "filename": filename, "prediction": prediction, "misclassified": prediction != correct_lang})
    

preds_df = df(preds)
print(preds_df)
preds_df.to_csv("cv_accents_confusion_matrix.csv")
preds_record_df = df(preds_record)
print(preds_record_df)
preds_record_df.to_csv("cv_accents-predictions.csv")


# Print accuracy for each L2 accent and speaker
for lang in preds:
    total = sum(preds[lang].values())
    correct = preds[lang]["en: English"]
    print(f"Lang: {lang}, Accuracy: {correct/total}")

