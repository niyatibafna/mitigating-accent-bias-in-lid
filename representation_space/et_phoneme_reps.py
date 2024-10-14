from transformers import AutoProcessor, WavLMModel
import torch
from datasets import load_dataset, Dataset, Audio
from torch.utils.data import DataLoader
from speechbrain.inference.classifiers import EncoderClassifier

from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

MAX_FILES = None
output_dir = "outputs/et_phoneme_reps/"
os.makedirs(output_dir, exist_ok=True)


def get_timit():
    timit_dir = "/export/common/data/corpora/LDC/LDC93S1/TIMIT/TRAIN"
    # Get a list of all wav files recursively
    wav_files = list(Path(timit_dir).rglob("*.WAV"))
    # Convert to string
    if MAX_FILES:
        wav_files = [str(wav_file) for wav_file in wav_files][:MAX_FILES]
    else:
        wav_files = [str(wav_file) for wav_file in wav_files]
    text_files = [wav_file.replace(".WAV", ".TXT") for wav_file in wav_files]
    timestamp_files = [wav_file.replace(".WAV", ".PHN") for wav_file in wav_files]

    print(f"Wav files to load audio: {wav_files[:10]}")
    timit = Dataset.from_dict({"audio": wav_files, "text_file": text_files, \
                "timestamp_file": timestamp_files}).cast_column("audio", Audio(sampling_rate=16_000))
    print(f"Example: {timit[0]['audio']}")
    print(f"Number of audio files: {len(timit)}")
    return timit


def get_labels_from_timestamp_file(timestamp_file):
    with open(timestamp_file, "r") as f:
        lines = f.readlines()
        timestamps = [line.strip().split(" ") for line in lines]
        timestamps = [[float(timestamp[0]), float(timestamp[1]), timestamp[2]] for timestamp in timestamps]

    return timestamps

def get_timestamp_labels_from_start_end(start, end, timestamps):
    # Note that TIMIT start and end are frames, not time
    best_label = None
    max_overlap = 0
    for timestamp in timestamps:
        overlap = min(end, timestamp[1]) - max(start, timestamp[0])
        if overlap > max_overlap:
            best_label = timestamp[2]
            max_overlap = overlap

    return best_label


# Prepare dataset
def prepare_dataset(batch):

    # Resampling the audio given the sampling rate
    audios = batch["audio"]
    array_values = [audio["array"] for audio in audios]
    assert  audios[0]["sampling_rate"] == 16_000, f"Expected sampling rate 16_000, got {audios[0]['sampling_rate']}"

    # Split audio into 100ms segments
    new_array_values = []
    labels = []
    for array, timestamp_file, in zip(array_values, batch["timestamp_file"]):
        timestamps = get_labels_from_timestamp_file(timestamp_file)
        num_segments = array.shape[0] // 1600
        for i in range(num_segments):
            new_array_values.append(array[i*1600:(i+1)*1600])
            label = get_timestamp_labels_from_start_end(i*1600, (i+1)*1600, timestamps)
            labels.append(label)
    
    batch["signal"] = new_array_values
    batch["label"] = labels

    return batch

timit = get_timit()
dataset = timit.map(prepare_dataset, remove_columns=["audio", "text_file", "timestamp_file"], batched=True)

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"}) # , 

all_reps = []
batch_size = 16
for data in dataset.iter(batch_size = batch_size):
    signals = torch.stack([torch.tensor(signal) for signal in data["signal"]])
    representations = language_id.encode_batch(signals)
    assert representations.shape[1] == 1, f"Expected shape (batch_size, 1, embedding_dim), got {representations.shape}"
    representations = representations.squeeze(axis = 1)
    all_reps.extend(representations)


all_reps = torch.stack(all_reps).cpu().numpy()
all_labels = [f["label"] for f in dataset]

# Save the representations and labels
np.save(os.path.join(output_dir, f"phone-reps_100ms_et_layer-final.npy"), all_reps)
np.save(os.path.join(output_dir, f"phone-labels_100ms_et_layer-final.npy"), np.array(all_labels))

# Plot the representations
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(all_reps)
all_reps_pca = pca.transform(all_reps)
print(f"all_codevectors_pca.shape: {all_reps_pca.shape}")

# Then to 2
tsne = TSNE(n_components=2)
all_reps_pca = tsne.fit_transform(all_reps_pca)
print(f"all_codevectors_pca.shape: {all_reps_pca.shape}")


import matplotlib.pyplot as plt
# Map each label to RGB color for pyplot
label_to_color = dict()
for label in all_labels:
    if label not in label_to_color:
        label_to_color[label] = np.random.rand(3,)

# Plot
fig, ax = plt.subplots()
for i in range(len(all_reps_pca)):
    ax.scatter(all_reps_pca[i, 0], all_reps_pca[i, 1], c=label_to_color[all_labels[i]])

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]

plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
plt.savefig(f"{output_dir}/phone-clusters_100ms_layer-final.png", bbox_inches='tight')

