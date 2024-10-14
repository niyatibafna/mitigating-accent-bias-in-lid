import os, sys, csv, json
import random
from collections import defaultdict
from pandas import DataFrame as df
import torch
import torchaudio
from datasets import Dataset
from speechbrain.inference.classifiers import EncoderClassifier


# wav_file = "/exp/nbafna/data/l2_arctic/l2arctic_release_v5/ABA/wav/arctic_a0001.wav"
# signal = language_id.load_audio(wav_file)
# print(type(signal))
# print(signal.shape)
# prediction = language_id.classify_batch(signal)
# print(prediction)
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"}) # , 
# Test on EdAcc dataset
def stm_reader(stm_path):
    stm_data = []
    with open(stm_path, "r") as f:
        for line in f:
            line_data = {}
            parts = line.split()
            line_data["audio_file"] = parts[0]
            line_data["channel"] = parts[1]
            line_data["speaker"] = parts[2]
            line_data["start_time"] = float(parts[3])
            line_data["end_time"] = float(parts[4])
            line_data["label"] = parts[5]
            line_data["transcript"] = " ".join(parts[6:])
            stm_data.append(line_data)
    return stm_data

def load_edacc(num_samples = None):
    test_stm_path = "/exp/nbafna/data/edacc/edacc_v1.0/test/stm"
    dev_stm_path = "/exp/nbafna/data/edacc/edacc_v1.0/dev/stm"
    data_path = "/exp/nbafna/data/edacc/edacc_v1.0/data"

    audio_files = {}
    for audio_file in os.listdir(data_path):
        # audio_files[audio_file[:-4]], sr = torchaudio.load(os.path.join(data_path, audio_file))
        audio_files[audio_file[:-4]] = language_id.load_audio(os.path.join(data_path, audio_file))
        sr = 16000 # This is the sampling rate for the language_id model, audio is normalized when loaded
    print(f"Loaded {len(audio_files)} audio files")

    speaker2lang = {}
    # linguistic_background = "/exp/nbafna/data/edacc/edacc_v1.0/linguistic_background.csv"
    # with open(linguistic_background, "r") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         speaker2lang[row[1]] = row[12]
    participant2accent_path = "/exp/nbafna/data/edacc/edacc_v1.0/participant2accent.json"
    with open(participant2accent_path, "r") as f:
        speaker2lang = json.load(f)
    print(f"Recorded {len(speaker2lang)} speakers")

    all_data = []
    stm_data = stm_reader(test_stm_path) + stm_reader(dev_stm_path)
    for line in stm_data:
        audio_file = line["audio_file"]
        signal = audio_files[audio_file]
        signal = signal.squeeze().numpy()
        segment = signal[int(line["start_time"]*sr):int(line["end_time"]*sr)]
        # Filter out signals with less than 6 seconds
        if segment.shape[0] < 6*16000:
            continue
        # Chunk into uniform windows of K seconds
        K = 6
        for i in range(0, len(segment), K*16000):
            if i+K*16000 > len(segment):
                break
            all_data.append({"signal": segment[i:i+K*16000], "lang": speaker2lang[line["speaker"]]})

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

                
dataset = load_edacc()
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

all_reps = []
batch_size = 16
for data in dataset.iter(batch_size = batch_size):
    signals = torch.stack([torch.tensor(signal) for signal in data["signal"]])
    representations = language_id.encode_batch(signals)
    assert representations.shape[1] == 1, f"Expected shape (batch_size, 1, embedding_dim), got {representations.shape}"
    representations = representations.squeeze(axis = 1)
    all_reps.extend(representations)
    
# Save the representations
import numpy as np
np.save("voxlingua_reps.npy", torch.stack(all_reps).cpu().numpy())
# Save the languages
langs = [f["lang"] for f in dataset]
np.save("voxlingua_langs.npy", langs)

# Plot the representations by reducing the dimensionality, e.g. PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

all_reps = torch.stack(all_reps).cpu().numpy()
pca = PCA(n_components=2)
pca_reps = pca.fit_transform(all_reps)
print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()}")
print(f"PCA explained variance: {pca.explained_variance_}")
print(f"PCA explained variance: {pca.explained_variance_.sum()}")
print(f"PCA explained variance: {pca.singular_values_}")
print(f"PCA explained variance: {pca.singular_values_.sum()}")
print(f"PCA explained variance: {pca.components_}")

# Plot the PCA
langs = list(set([f["lang"] for f in dataset]))
colors = plt.cm.get_cmap("tab20", len(langs))
colors = colors.colors  
for i, lang in enumerate(langs):
    indices = [j for j in range(len(dataset)) if dataset[j]["lang"] == lang]
    plt.scatter(pca_reps[indices, 0], pca_reps[indices, 1], color=colors[i], label=lang)
plt.legend()
plt.savefig("pca_reps.png", dpi=300, bbox_inches="tight")


# preds_df = df(preds)
# print(preds_df)
# preds_df.to_csv("edacc_preds.csv")

# # Print accuracy for each L2 accent and speaker
# for lang in preds:
#     total = sum(preds[lang].values())
#     correct = preds[lang]["en: English"]
#     print(f"Lang: {lang}, Accuracy: {correct/total}")

