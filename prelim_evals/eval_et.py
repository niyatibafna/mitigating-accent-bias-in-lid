import os, sys, csv, json
import random
from collections import defaultdict
from pandas import DataFrame as df
import torch
import torchaudio
from datasets import Dataset
from speechbrain.inference.classifiers import EncoderClassifier
import pickle as pkl


sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/dataloading/")
from dataset_loader import load_lid_dataset

dataset_name = "edacc"
lang = sys.argv[1]
output_dir = f"/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/preds/{dataset_name}_predictions/"
os.makedirs(output_dir, exist_ok = True)

# Load the dataset
dataset = load_lid_dataset(dataset_name, lang = lang, target_code_type = "vl107")

if not dataset:
    print("Dataset not found")
    sys.exit(1)
# dataset: {"signal": [np.array], 
#         "lang": [str], 
#         "accent": [str],
#         "audio_file": [str]}, 
#     where lang codes match vl107, accent is the
# manually annotated/modified accent of each speaker in the dataset.
# "signal" contains segments of length clip_length = 6.0s, sampled at 16kHz.


preds = defaultdict(lambda: defaultdict(int))

print(f"Dataset size: {len(dataset)}")

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"}) # , 
# If we have ragged tensors, or chunks of audio that are not of the same length, we can't use batch_size > 1

batch_size = 16
all_preds = []
all_labels = []
all_accents = []
for data in dataset.iter(batch_size = batch_size):
    # print("here")
    signals = torch.stack([torch.tensor(signal) for signal in data["signal"]])
    predictions = language_id.classify_batch(signals)
    for accent, prediction in zip(data["accent"], predictions[3]):
        preds[accent][prediction] += 1
    
    all_preds.extend(predictions[3])
    all_labels.extend(data["lang"])
    all_accents.extend(data["accent"])


assert len(all_preds) == len(all_labels) == len(all_accents), "Lengths of predictions, labels, and accents do not match"
print(f"Length of all_preds: {len(all_preds)}")

preds = [pred.split(":")[0] for pred in all_preds] # ET returns labels like "en: English", we only want the language code

with open(os.path.join(output_dir, f"{lang}_predictions.pkl"), "wb") as f:
    # pkl.dump({"audio_files": audio_files_test, "preds": preds, "labels": labels}, f)
    pkl.dump({"preds": preds, "labels": all_labels, "accents": all_accents}, f)


correct = 0
total = 0
for pred, label in zip(preds, all_labels):
    if pred == label:
        correct += 1
    total += 1

print(f"Accuracy: {correct/total}")

# Save all predictions and labels


# preds_df = df(preds)
# print(preds_df)
# preds_df.to_csv(f"/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/preds/{dataset_name}_preds.csv")

# # Print accuracy for each L2 accent and speaker
# for accent in preds:
#     total = sum(preds[accent].values())
#     correct = preds[accent]["en: English"]
#     print(f"Lang: {accent}, Accuracy: {correct/total}")

