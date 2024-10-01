import os, sys, csv
import random
from collections import defaultdict
from pandas import DataFrame as df
import torch
import torchaudio
from datasets import Dataset
from speechbrain.inference.classifiers import EncoderClassifier
# from pydub import AudioSegment
from lhotse import Recording, MonoCut


def load_nistlre(num_samples = None):
    accents = {
        "ara-apc",
        "ara-acm",
        "ara-ary",
        "ara-arz",
        "eng-gbr",
        "eng-usg",
        "spa-car",
        "spa-eur",
        "spa-lac",
        "zho-cmn",
        "zho-nan",
        "por-brz",
        "por-eur",
        "qsl-pol",
        "qsl-rus",
        "afr-afr",
        "ara-aeb",
        "ara-arq",
        "ara-ayl",
        "eng-ens",
        "eng-iaf",
        "fra-ntf",
    }
    all_data = []
    with open("/exp/jvillalba/corpora/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets/docs/train_info.tab") as f:
        for line in f:
            accent, path = line.strip().split()[0], line.strip().split()[1]
            if accent.strip() not in accents:
                continue
            data_path = "/exp/jvillalba/corpora/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets/data/train/"
            filepath = f"{data_path}{accent.strip()}/{path.strip()}"
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
                all_data.append({"signal": segment[i:i+K*16000], "lang": accent})

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
    all_langs = set([f["lang"] for f in all_data])
    print(f"Languages: {all_langs}")

    if num_samples is not None:
        all_data = random.sample(all_data, min(len(all_data), num_samples))
    all_data = {"signal": [f["signal"] for f in all_data], "lang": [f["lang"] for f in all_data]}
    
    return Dataset.from_dict(all_data)

dataset = load_nistlre(num_samples=100)
preds = defaultdict(lambda: defaultdict(int))

print(f"Dataset size: {len(dataset)}")

# If we have ragged tensors, or chunks of audio that are not of the same length, we can't use batch_size > 1
# for data in dataset.iter(batch_size=3):
#     for signal, lang in zip(data["signal"], data["lang"]):
#         signal = torch.tensor(signal)
#         # print(f"Signal shape: {signal.shape}")
#         signal = signal.to(language_id.device)
#         try:
#             prediction = language_id.classify_batch(signal)
#         except:
#             print("Signal shape: ", signal.shape)
#         preds[lang][prediction[3][0]] += 1
#         # print(f"Lang: {lang}, Pred: {prediction[3]}")

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"}) # , 
batch_size = 16
for data in dataset.iter(batch_size = batch_size):
    signals = torch.stack([torch.tensor(signal) for signal in data["signal"]])
    predictions = language_id.classify_batch(signals)
    for lang, prediction in zip(data["lang"], predictions[3]):
        preds[lang][prediction] += 1


preds_df = df(preds)
print(preds_df)
preds_df.to_csv("nistlre_preds.csv")

# Print accuracy for each L2 accent and speaker
iso2lang = {
    "eng": "en: English",
    "ara": "ar: Arabic",
    "spa": "es: Spanish",
    "zho": "zh: Chinese",
    "por": "pt: Portuguese",
    "qsl": "ru: Russian",
    "afr": "af: Afrikaans",
    "fra": "fr: French",
    "qsl": "pl: Polish",
}
for lang in preds:
    total = sum(preds[lang].values())
    correct_lang = iso2lang[lang.split("-")[0]]
    correct = preds[lang][correct_lang] if correct_lang in preds[lang] else 0
    print(f"Lang: {lang}, Accuracy: {correct/total}")

