import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

reps = np.load('reps/voxlingua_edacc_reps.npy')
langs = np.load('reps/voxlingua_edacc_langs.npy')

print(f"Reps shape: {reps.shape}")
assert len(reps) == len(langs), f"Expected same number of representations and languages, got {len(reps)} and {len(langs)}"


pca = PCA(n_components=2)
pca_reps = pca.fit_transform(reps)
print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()}")
print(f"PCA explained variance: {pca.explained_variance_}")
print(f"PCA explained variance: {pca.explained_variance_.sum()}")
print(f"PCA explained variance: {pca.singular_values_}")
print(f"PCA explained variance: {pca.singular_values_.sum()}")
print(f"PCA explained variance: {pca.components_}")

# Plot the PCA
langset = list(set(langs))

# langset_groups = {
#     "slavic": ['polish', 'russian', 'bulgarian', 'macedonian', 'lithuanian'],
#     "scottish-irish": ['scottish', 'irish'],
#     "w-african": ['ghanian', 'nigerian'],
#     "south-african": ['south-african'],
#     "indian-pakistani": ['indian', 'pakistani'],
#     "se-asian": ['vietnamese', 'indonesian', 'filipino'],
#     "east-asian": ['japanese', 'korean', 'chinese'],
#     "hispanic": ['mexican', 'colombian', 'ecuadorian', 'chilean', 'spanish'],
#     "iberian": ['spanish', 'catalan'],
# # }
# langset_groups = {}
# langs_covered = {lang for group in langset_groups.values() for lang in group}
# langs_not_covered = set(langset) - langs_covered
# for lang in langs_not_covered:
#     langset_groups[lang] = [lang]

# langset = list(langset_groups.keys())

lang_per_plot = 20

num_groups = len(langset)//lang_per_plot

figure, axes = plt.subplots(num_groups, 1, figsize=(10, 10*num_groups))
axes = [axes] if num_groups == 1 else axes

# colors = plt.get_cmap("tab20", len(langset))
# colors = colors.colors  
colors = plt.get_cmap("tab20", lang_per_plot)
colors = colors.colors

for i, ax in enumerate(axes):
    for j, lang in enumerate(langset[i*lang_per_plot:(i+1)*lang_per_plot]):
        indices = [k for k, l in enumerate(langs) if l == lang]
        ax.scatter(pca_reps[indices, 0], pca_reps[indices, 1], label=lang, color=colors[j])
        ax.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("pca_reps.png", dpi=300, bbox_inches="tight")
