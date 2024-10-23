import os, sys
import random
from collections import defaultdict
from pandas import DataFrame as df
import torch
import torchaudio
from datasets import Dataset
from speechbrain.inference.classifiers import EncoderClassifier
import numpy as np

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"}) # , 
# wav_file = "/exp/nbafna/data/l2_arctic/l2arctic_release_v5/ABA/wav/arctic_a0001.wav"
# signal = language_id.load_audio(wav_file)
# print(type(signal))
# print(signal.shape)
# prediction = language_id.classify_batch(signal)
# print(prediction)

# Test on CV dataset
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
        signal = language_id.load_audio(os.path.join(clips_folder, audio))
        if signal.shape[0] < 6*16000:
            continue
        K = 6
        for i in range(0, len(signal), K*16000):
            if i+K*16000 > len(signal):
                break
            data.append({"signal": signal[i:i+K*16000], "lang": lang})

    data = {"signal": [f["signal"] for f in data], "lang": [f["lang"] for f in data]}
    return Dataset.from_dict(data)

def get_reps(dataset, output_path):

    all_reps = []
    batch_size = 16
    for data in dataset.iter(batch_size = batch_size):
        signals = torch.stack([torch.tensor(signal) for signal in data["signal"]])
        representations = language_id.encode_batch(signals)
        assert representations.shape[1] == 1, f"Expected shape (batch_size, 1, embedding_dim), got {representations.shape}"
        representations = representations.squeeze(axis = 1)
        all_reps.extend(representations)
    
    
    reps = torch.stack(all_reps).cpu().numpy()
    np.save(output_path, reps)
    return reps
    
def compute_similarity(reps1, reps2):
    '''Compute cosine similarity between pairs of representations'''
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(reps1, reps2)
    return np.mean(similarity)


def visualize_reps(reps, labels, output_path):
    '''Visualize representations using PCA and TSNE'''
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    pca = PCA(n_components=50)
    pca.fit(reps)
    reps_pca = pca.transform(reps)
    print(f"Reps PCA shape: {reps_pca.shape}")

    # Then to 2
    tsne = TSNE(n_components=2)
    reps_pca = tsne.fit_transform(reps_pca)
    print(f"Reps PCA shape: {reps_pca.shape}")

    # Map each label to RGB color for pyplot
    label_to_color = dict()
    for label in labels:
        if label not in label_to_color:
            label_to_color[label] = np.random.rand(3,)

    # Plot
    for label in label_to_color:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reps_pca[indices, 0], reps_pca[indices, 1], label=label, color=label_to_color[label])

    plt.title("Language/Accent subspaces")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(output_path, bbox_inches='tight')


def main():
    lang2reps = {}
    output_dir = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/representation_space/accent_subspaces_outputs"
    langs = {"hi", "en", "as", "yo", "ha", "id", "ga-IE"}
    MAX_SAMPLES = 5000
    for lang in langs:
        output_path = os.path.join(output_dir, f"{lang}_reps.npy")
        if os.path.exists(output_path):
            print(f"Reps already computed for {lang}")
            reps = np.load(output_path)
        else:
            print("Computing reps for", lang)
            dataset = load_cv_lang(lang = lang, per_lang = MAX_SAMPLES)
            reps = get_reps(dataset, output_path)
        lang2reps[lang] = reps
        print(f"{lang} reps shape: {reps}")

    en_accents = {"indian", "scotland", "malaysia", "philippines", "scotland", "ireland", "wales", "african"}
    for en_accent in en_accents:
        lang = f"en_{en_accent}"
        output_path = os.path.join(output_dir, f"{lang}_reps.npy")
        if os.path.exists(output_path):
            print(f"Reps already computed for {lang}")
            reps = np.load(output_path)
        else:
            print("Computing reps for", lang)
            en_accented_dataset = load_cv_lang(en_accent = en_accent, per_lang = MAX_SAMPLES)
            reps = get_reps(en_accented_dataset, output_path)
        lang2reps[lang] = reps
        print(f"{lang} shape: {reps.shape}")

    # Compute distances between all pairs of languages
    similarities = defaultdict(lambda: defaultdict(float))
    for lang1 in lang2reps:
        for lang2 in lang2reps:
            reps1, reps2 = lang2reps[lang1], lang2reps[lang2]
            similarities[lang1][lang2] = compute_similarity(reps1, reps2)

    print(df(similarities))
    with open(os.path.join(output_dir, "cv_cosine_similarities.csv"), "w") as f:
        f.write(df(similarities).to_csv(sep = "\t"))

    # Visualize the representations
    all_reps = []
    for lang in lang2reps:
        reps = lang2reps[lang].tolist()
        all_reps.extend(lang2reps[lang])
    all_reps = np.array(all_reps)
    labels = []
    for lang in lang2reps:
        labels.extend([lang]*len(lang2reps[lang]))
    output_path = os.path.join(output_dir, f"all_accent_subspaces.png")

    visualize_reps(all_reps, labels, output_path)
    

if __name__ == "__main__":
    main()