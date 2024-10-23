'''
We generated ET reps for
1. Training data: (VL107) languages of interest from VL107, including English as well as "underlying L1" languages 
of our accented data (like Hindi, Tagalog, etc.)
2. Accented data: (CommonVoice) English with various accents 
3. Language data: (CommonVoice) underlying L1 languages of our accented data (like Hindi, Tagalog, etc.)

Now we want to see whether the accented data reps (2) is close to English or the underlying L1 language (1).
We'll also make the same comparison for the underlying L1 language reps (3) as a control.
'''

import os
import numpy as np
from collections import defaultdict
import pandas as pd

# Loading CV reps
cv_lang2reps = {}
output_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/representation_space/accent_subspaces_outputs/reps_et_cv"
cv_langs = {"hi", "en", "as", "yo", "ha", "id", "ga-IE"}
en_accents = {"indian", "scotland", "malaysia", "philippines", "scotland", "ireland", "wales", "african"}
en_accents = {f"en_{accent}" for accent in en_accents}
cv_langs = cv_langs.union(en_accents)
MAX_SAMPLES = 5000
for lang in cv_langs:
    print(f"Loading CV reps for {lang}")
    output_path = os.path.join(output_dir, f"{lang}_reps.npy")
    if not os.path.exists(output_path):
        print(f"Reps missing for {lang}")
    else:
        reps = np.load(output_path)
    cv_lang = f"cv_{lang}"
    cv_lang2reps[cv_lang] = reps
    print(f"{lang} reps shape: {reps}")


# Loading VL107 reps
vl107_lang2reps = {}
output_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/representation_space/accent_subspaces_outputs/reps_et_voxlingua107"
vl_langs = {"hi", "bn", "as", "te", "en", "yo", "ha", "af", "id", "tl", "cy", "gv", "la", "mr", "fr", "ma", "ta", "de", "es", "pa", "gu"}
MAX_SAMPLES = 5000
for lang in vl_langs:
    print(f"Loading ET reps for {lang}")
    output_path = os.path.join(output_dir, f"{lang}_reps.npy")
    if not os.path.exists(output_path):
        print(f"Reps missing for {lang}")
    else:
        reps = np.load(output_path)
    vl_lang = f"vl_{lang}"
    vl107_lang2reps[vl_lang] = reps
    print(f"{lang} reps shape: {reps}")

# Visualize the representations
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

    # Save the PCA representation
    output_path = os.path.join(output_dir, f"cv_and_vl107_langs_pca.npy")
    np.save(output_path, reps_pca)
    # Save the labels
    output_path = os.path.join(output_dir, f"cv_and_vl107_langs_labels.npy")
    np.save(output_path, labels)

    # Map each label to RGB color for pyplot
    label_to_color = dict()
    
    # Get contrasting colors for each language
    colours = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    for i, label in enumerate(set(labels)):
        label_to_color[label] = colours[i]

    # Plot
    for label in label_to_color:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reps_pca[indices, 0], reps_pca[indices, 1], label=label, color=label_to_color[label])

    output_path = os.path.join(output_dir, f"cv_and_vl107_langs.png")
    plt.title("Language/Accent subspaces")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(output_path, bbox_inches='tight')


# all_reps = []
# labels = []
# output_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/representation_space/accent_subspaces_outputs/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir, exist_ok=True)

# for lang in cv_lang2reps:
#     reps = cv_lang2reps[lang].tolist()
#     all_reps.extend(cv_lang2reps[lang])
#     labels.extend([f"cv_{lang}"]*len(cv_lang2reps[lang]))
# for lang in vl107_lang2reps:
#     reps = vl107_lang2reps[lang].tolist()
#     all_reps.extend(vl107_lang2reps[lang])
#     labels.extend([f"vl_{lang}"]*len(vl107_lang2reps[lang]))
# all_reps = np.array(all_reps)


# visualize_reps(all_reps, labels, output_path)

def compute_similarity(reps1, reps2):
    '''Compute cosine similarity between pairs of representations'''
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(reps1, reps2)
    return np.mean(similarity)

# Compute similarity between CV and VL107 reps

cosine_similarities = defaultdict(dict)

for cv_lang in cv_lang2reps:
    for vl_lang in vl107_lang2reps:
        cosine_similarities[cv_lang][vl_lang] = round(compute_similarity(cv_lang2reps[cv_lang], vl107_lang2reps[vl_lang]), 3)

# Sort rows and columns by language
df_cosine_similarities = pd.DataFrame(cosine_similarities)
df_cosine_similarities = df_cosine_similarities.reindex(sorted(df_cosine_similarities.columns), axis=1)
df_cosine_similarities = df_cosine_similarities.reindex(sorted(df_cosine_similarities.index), axis=0)


output_dir = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/representation_space/accent_subspaces_outputs/compare_pred_and_train_reps/"
output_path = os.path.join(output_dir, "cosine_similarities_cv_vl107.csv")
df_cosine_similarities.to_csv(output_path, sep="\t")

# Visualize heatmap of cosine similarities
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
sns.heatmap(df_cosine_similarities, annot=True, cmap="Reds")
plt.xticks(rotation=60)
plt.yticks(rotation=0)
plt.title("Cosine similarities between CV and VL107 reps")
plt.savefig(os.path.join(output_dir, "cosine_similarities_cv_vl107.png"), bbox_inches="tight")

                  

