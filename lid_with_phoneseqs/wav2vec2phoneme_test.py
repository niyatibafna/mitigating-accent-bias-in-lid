#%%
import transformers
#%%


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
 
# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    
# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

#%%
# tokenize
input_values = processor(ds[6]["audio"]["array"], return_tensors="pt").input_values

# retrieve logits
with torch.no_grad():
  logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)

# => should give ['m ɪ s t ɚ k w ɪ l t ɚ ɪ z ð ɪ ɐ p ɑː s əl l ʌ v ð ə m ɪ d əl k l æ s ɪ z æ n d w iː aʊ ɡ l æ d t ə w ɛ l k ə m h ɪ z ɡ ɑː s p ə']
#%%
print(f"Text: {ds[6]['text']}")
#%%

import os, sys, csv, json
import random
from collections import defaultdict
from pandas import DataFrame as df
import torch
import torchaudio
from datasets import Dataset


# wav_file = "/exp/nbafna/data/l2_arctic/l2arctic_release_v5/ABA/wav/arctic_a0001.wav"

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
        audio_files[audio_file[:-4]], sr = torchaudio.load(os.path.join(data_path, audio_file))
        # audio_files[audio_file[:-4]] = language_id.load_audio(os.path.join(data_path, audio_file))
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
        transcript = line["transcript"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in transcript:
            continue
        # Filter out signals with less than 6 seconds
        if segment.shape[0] < 6*16000:
            continue
        # Chunk into uniform windows of K seconds
        K = 6
        for i in range(0, len(segment), K*16000):
            if i+K*16000 > len(segment):
                break
            all_data.append({"signal": segment[i:i+K*16000], \
                             "lang": speaker2lang[line["speaker"]],\
                                "transcript": transcript})

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
    all_data = {"signal": [f["signal"] for f in all_data], \
                "lang": [f["lang"] for f in all_data],\
                    "transcript": [f["transcript"] for f in all_data]}
    
    return Dataset.from_dict(all_data)

                
dataset = load_edacc()



# %%

for data in dataset.select(range(100)):
    if "IGNORE_TIME_SEGMENT_IN_SCORING" in data["transcript"]:
        continue
    print(data["transcript"])
    input_values = processor(data["signal"], return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription)
    print("\n")


# %%
transcripts = [data["transcript"] for data in dataset.select(range(1000))]
ignore_count = [1 for transcript in transcripts if "IGNORE_TIME_SEGMENT_IN_SCORING" in transcript]
print(f"Ignore count: {len(ignore_count)}")
print(f"Total count: {len(transcripts)}")
# %%
print(len(dataset))
# %%
transcripts = [data["transcript"] for data in dataset.select(range(100))]
# %%


test_stm_path = "/exp/nbafna/data/edacc/edacc_v1.0/test/stm"
dev_stm_path = "/exp/nbafna/data/edacc/edacc_v1.0/dev/stm"
data_path = "/exp/nbafna/data/edacc/edacc_v1.0/data"

# audio_files = {}
# for audio_file in os.listdir(data_path):
#     audio_files[audio_file[:-4]], sr = torchaudio.load(os.path.join(data_path, audio_file))
#     # audio_files[audio_file[:-4]] = language_id.load_audio(os.path.join(data_path, audio_file))
#     sr = 16000 # This is the sampling rate for the language_id model, audio is normalized when loaded
# print(f"Loaded {len(audio_files)} audio files")

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
less_than_six = 0
good_segments = 0
ignore_segments = 0
sr = 16000
stm_data = stm_reader(test_stm_path) + stm_reader(dev_stm_path)
for line in stm_data:
    audio_file = line["audio_file"]
    # signal = audio_files[audio_file]
    # signal = signal.squeeze().numpy()
    # segment = signal[int(line["start_time"]*sr):int(line["end_time"]*sr)]
    transcript = line["transcript"]
    if "IGNORE_TIME_SEGMENT_IN_SCORING" in transcript:
        ignore_segments += (line["end_time"]-line["start_time"]) // 6
    # Filter out signals with less than 6 seconds
    if line["end_time"]-line["start_time"] < 6:
        less_than_six += 1
    
    good_segments += (line["end_time"]-line["start_time"]) // 6


# %%
print(f"Less than six: {less_than_six}")
print(f"Good segments: {good_segments}")
print(f"Ignore segments: {ignore_segments}")

# %%



#### TEST ON CV DATASET ####

def load_cv(per_accent = None):
    files = {}
    accents = {'indian', 'singapore', 'scotland', 'us', 'canada', 'wales', 'england', 'philippines', 'african', 'newzealand', 'ireland', 'malaysia', 'hongkong', 'australia'}
    for line in open("/export/common/data/corpora/ASR/commonvoice/en/train.tsv"):
        if len(line.strip().split("\t")) < 8:
            continue
        audio, transcript, accent = line.strip().split("\t")[1], line.strip().split("\t")[2], line.strip().split("\t")[7]
        if accent not in accents:
            continue
        if accent not in files:
            files[accent] = []
        files[accent].append((audio, transcript))

    for accent in accents:
        if accent not in files:
            continue
        random.shuffle(files[accent])
        files[accent] = files[accent][:per_accent]
        
    # print(files)
    print("Loading audio files...")
    data = []
    clips_folder = "/export/common/data/corpora/ASR/commonvoice/en/clips/"
    for accent in files:
        for (audio, transcript) in files[accent]:
            print(f"Loading {audio}...")
            signal, sr = torchaudio.load(os.path.join(clips_folder, audio))
            signal = signal.squeeze().numpy()
            data.append({"signal": signal, \
                             "lang": accent, \
                                "filename": os.path.join(clips_folder, audio),\
                                    "transcript": transcript})
            
            # signal = language_id.load_audio(os.path.join(clips_folder, audio))
            # K = signal.shape[0] // 16000
            # if signal.shape[0] < 10*16000:
            #     print(f"Signal too short: {signal.shape[0]}")
            #     continue
            
            # for i in range(0, len(signal), K*16000):
            #     if i+K*16000 > len(signal):
            #         break
            #     data.append({"signal": signal[i:i+K*16000], \
            #                  "lang": accent, \
            #                     "filename": os.path.join(clips_folder, audio),\
            #                         "transcript": transcript})
            

    data = {"signal": [f["signal"] for f in data], \
            "lang": [f["lang"] for f in data], \
                "filename": [f["filename"] for f in data],
                "transcript": [f["transcript"] for f in data]}
    return Dataset.from_dict(data)
        
dataset = load_cv(per_accent = 5)

# %%
print(len(dataset))

# Transcrbe
for data in dataset.shuffle().select(range(30)):
    # if data["lang"] != "us":
    #     continue

    print(data["transcript"])
    print(data["filename"])
    input_values = processor(ds[7]["audio"]["array"], return_tensors="pt").input_values
    # input_values = processor(data["signal"], sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription)
    print("\n")
    print(f"Truth: {ds[7]['text']}")


# %%
print([data["lang"] for data in dataset.select(range(30))])
# %%


# ds = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="validation", streaming=True)
from datasets import load_dataset

ds = load_dataset("lmms-lab/librispeech", "librispeech_dev_clean")
# ds = load_dataset("lmms-lab/librispeech", "librispeech_dev_other", streaming=True)
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "other", split="validation")
ds = iter(ds)
print(next(ds))

# %%
ds
print(next(ds))
# %%
for data in range(30):
    # if data["lang"] != "us":
    #     continue
    data = next(ds)
    print(data["sentence"])
    print(data["path"])
    input_values = processor(data["audio"]["array"], return_tensors="pt").input_values

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    print(transcription)
    print("\n")

# %%
