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

# Visualize the representations
def visualize_reps(reps, labels, output_path, langfam_id):
    '''Visualize representations using PCA and TSNE'''
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Map each label to RGB color for pyplot
    label_to_color = dict()
    
    # Get contrasting colors for each language
    colours = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    for i, label in enumerate(set(labels)):
        label_to_color[label] = colours[i]

    label_set = set(labels)
    accent_labels = {label for label in label_set if "en_" in label}
    label_list = list(label_set - accent_labels) + list(accent_labels)
    # Plot
    for label in label_list:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reps_pca[indices, 0], reps_pca[indices, 1], label=label, color=label_to_color[label])

    output_path = os.path.join(output_dir, f"{langfam_id}_cv_and_vl107_langs.png")
    plt.title("Language/Accent subspaces")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(output_path, bbox_inches='tight')


def filter_by_langfam(reps, labels, langfam_id):
    '''Filter reps and labels by language family'''
    filtered_reps = []
    filtered_labels = []
    langfam_id2langs = {
        "indian": {"cv_hi", "cv_as", "cv_en_indian", "vl_hi", "vl_as", "vl_te", "vl_bn", "vl_en", "cv_en"},
        "african": {"cv_yo", "cv_ha", "vl_yo", "vl_ha", "vl_af", "cv_en_african", "vl_en", "cv_en"},
    }

    for rep, label in zip(reps, labels):
        if label in langfam_id2langs[langfam_id]:
            filtered_reps.append(rep)
            filtered_labels.append(label)

    print(f"Filtered reps shape: {np.array(filtered_reps).shape}")
    return np.array(filtered_reps), filtered_labels


cv_lang2reps = {}
cv_langs = {"hi", "en", "as", "yo", "ha", "id", "ga-IE"}
cv_lang_labels = {f"cv_{cv_lang}": cv_lang for cv_lang in cv_langs}

en_accents = {"indian", "scotland", "malaysia", "philippines", "scotland", "ireland", "wales", "african"}
en_accent_labels = {f"en_{en_accent}": en_accent for en_accent in en_accents}

langs = {"hi", "bn", "as", "te", "en", "yo", "ha", "af", "id", "tg", "cy", "gv", "lt"}
vl107_lang_labels = {f"vl_{vl107_lang}": vl107_lang for vl107_lang in langs}

# Load PCA reps
output_dir = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/representation_space/accent_subspaces_outputs/"
output_path = os.path.join(output_dir, f"cv_and_vl107_langs_pca.npy")
reps_pca = np.load(output_path)
output_path = os.path.join(output_dir, f"cv_and_vl107_langs_labels_pca.npy")
labels = np.load(output_path)

langfam_id = "african"
reps, labels = filter_by_langfam(reps_pca, labels, langfam_id)

output_dir = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/representation_space/accent_subspaces_outputs/compare_pred_and_train_reps/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

visualize_reps(reps, labels, output_path, langfam_id)