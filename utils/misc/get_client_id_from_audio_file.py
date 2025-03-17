import json
import random
import os

participant2accent_filepath = "/exp/nbafna/data/edacc/edacc_v1.0/participant2accent.json"
with open(participant2accent_filepath) as f:
    participant2accent = json.load(f)

cv_mapping_file = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/misc/speaker_id_mappings/cv.json"
with open(cv_mapping_file) as f:
    cv_mapping = json.load(f)

cv_from_hf_mapping = {}
langs = ["es", "fr", "de", "it"]
for lang in langs:
    mapping_file = f"/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/misc/speaker_id_mappings/cv_from_hf_{lang}.json"
    with open(mapping_file) as f:
        cv_from_hf_mapping[lang] = json.load(f)

fleurs_test_mapping = {}
langs = os.listdir("/export/common/data/corpora/fleurs/metadata/")
for lang in langs:
    dataset_dir = f"/export/common/data/corpora/fleurs/{lang}/test"
    mapping_file = f"/export/common/data/corpora/fleurs/metadata/{lang}/test.tsv"
    audio_file2id = {}
    with open(mapping_file) as f:
        metadata = f.readlines()
    for line in metadata:
        file_id, audio_file = line.strip().split("\t")[0], line.strip().split("\t")[1]
        audio_file = os.path.join(dataset_dir, audio_file)
        audio_file2id[audio_file] = file_id
    fleurs_test_mapping[lang] = audio_file2id
# Convert langs to VL107 codes
for lang in langs:
    vl107_code = lang.split("_")[0]
    fleurs_test_mapping[vl107_code] = fleurs_test_mapping.pop(lang)



def get_speaker_id_from_audio_file_edacc(audio_file, accent):
    
    # Some audio files have _P* at the end. We need to remove that to match the participant ID.
    audio_file = audio_file.split("_P")[0]

    file_speakers = [spk for spk in participant2accent.keys() if audio_file in spk]
    # Shuffle file_speakers to avoid bias in selecting the first speaker
    # random.shuffle(file_speakers)
    # This is a hack because I didn't store the partipant ID in the eval data.
    # Since each audio_file has two unique speakers, given the audio_file, it can only be one of two speakers.
    # Since we also know the accent of the speaker, we can use that to match the speaker.
    # This is a problem because the conversation may have been between two speakers of the same accent.
    # In that case, this function will lump them together.
    # This is not really a problem if we only need this for independent sampling for bootstrap CIs.
    for file_speaker in file_speakers:
        if participant2accent[file_speaker] == accent:
            return file_speaker
        
    print(f"Could not find speaker for {audio_file} with accent {accent}")
        

def get_speaker_id_from_audio_file_cv(audio_file):
    return cv_mapping[audio_file]


def get_speaker_id_from_audio_file_cv_from_hf(audio_file, lang):
    return cv_from_hf_mapping[lang][audio_file]

def get_speaker_id_from_audio_file_fleurs_test(audio_file, lang):
    return fleurs_test_mapping[lang][audio_file]



def get_speaker_id_from_audio_file(dataset_name, audio_file, accent=None, lang=None):
    if dataset_name == "cv":
        return get_speaker_id_from_audio_file_cv(audio_file)
    elif dataset_name == "edacc":
        return get_speaker_id_from_audio_file_edacc(audio_file, accent)
    elif dataset_name in {"cv_from_hf", "cv_from_hf_l2"}:
        return get_speaker_id_from_audio_file_cv_from_hf(audio_file, lang)
    elif dataset_name == "fleurs_test":
        return get_speaker_id_from_audio_file_fleurs_test(audio_file, lang)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


# # We'll get audio_file2client_id mappings for the CV dataset
# lang = "en"
# split = "train"
# accents = {'indian', 'singapore', 'scotland', 'us', 'canada', 'wales', 'england', 'philippines', 'african', 'newzealand', 'ireland', 'malaysia', 'hongkong', 'australia'}

# audio_file2client_id = {}

# for line in open(f"/export/common/data/corpora/ASR/commonvoice/{lang}/{split}.tsv"):
#     if len(line.strip().split("\t")) < 8:
#         continue
#     client_id, audio, accent = line.strip().split("\t")[0], line.strip().split("\t")[1], line.strip().split("\t")[7]
#     if accent not in accents or (not audio.endswith(".mp3") and not audio.endswith(".wav")):
#         continue
#     audio_file2client_id[audio] = client_id

# outpath = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/misc/speaker_id_mappings/cv.json"
# with open(outpath, "w") as f:
#     json.dump(audio_file2client_id, f, indent=4)
# print(f"Saved to {outpath}")


# from datasets import load_from_disk
# import json
# langs = ["es", "fr", "de", "it"]
# outdir = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/misc/speaker_id_mappings/"
# for lang in langs:
#     print(f"Processing {lang}...")
#     audio_file2spk_id = {}
#     dataset_file = f"/exp/nbafna/data/commonvoice/accented_data/{lang}/{lang}_accented_samples-5k"
#     lang_dataset = load_from_disk(dataset_file)
#     client_ids = lang_dataset["client_id"]
#     audio_files = lang_dataset["audio_file"]
#     for audio_file, client_id in zip(audio_files, client_ids):
#         audio_file2spk_id[audio_file] = client_id

#     with open(os.path.join(outdir, f"cv_from_hf_{lang}.json"), "w") as f:
#         json.dump(audio_file2spk_id, f, indent=4)
#     print(f"Done with {lang}")


