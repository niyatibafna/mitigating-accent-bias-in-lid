import pandas as pd
import os, sys
from collections import defaultdict
import pickle as pkl
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils")
from lang_code_maps import vl107_to_fleurs_map


# results_csv = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/preds/edacc_preds_new.csv"

results_dir = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/preds/fleurs_test_predictions"

output_path = None
# Use rglob to get all files in the directory
results_files = [d for d in os.listdir(results_dir) if d.endswith(".pkl")]
results_by_lang = defaultdict(lambda: defaultdict(int))

all_preds = []
all_accents = []
all_labels = []

for predictions_path in results_files:

    with open(os.path.join(results_dir, predictions_path), "rb") as f:
        eval_data = pkl.load(f)

    for prediction, accent, label in zip(eval_data["preds"], eval_data["accents"], eval_data["labels"]):
        all_preds.append(prediction)
        all_accents.append(accent)
        all_labels.append(label)


results_by_lang = defaultdict(lambda: defaultdict(int))
# print(list(zip(eval_data["preds"], eval_data["accents"], eval_data["labels"])))
for prediction, accent, label in zip(all_preds, all_accents, all_labels):
    if prediction == label:
        results_by_lang[label]["correct"] += 1
    results_by_lang[label]["total"] += 1
# %%

for lang, results in results_by_lang.items():
    results_by_lang[lang]["accuracy"] = round(results["correct"]/results["total"], 1)
    print(f"Accuracy for {lang}: {results['correct']/results['total']}")
    print(f"Total samples for {lang}: {results['total']}")
    print()

switch_to_vl107 = False
if switch_to_vl107:
    vl107_to_fleurs = vl107_to_fleurs_map()
    fleurs_to_vl107 = {v: k for k, v in vl107_to_fleurs.items()}
    print(fleurs_to_vl107)

    # for lang, results in results_by_lang.items():
    #     if lang in fleurs_to_vl107:
    #         results_by_lang[fleurs_to_vl107[lang]] = results_by_lang.pop(lang)
    #     else:
    #         results_by_lang.pop(lang)

    results_by_lang = {fleurs_to_vl107[lang]: results for lang, results in results_by_lang.items() if lang in fleurs_to_vl107}

# %%
# df = pd.DataFrame(results_by_lang).T
# if output_path:
#     df.to_csv(output_path)

# %%
langs = sorted(list(results_by_lang.keys()))
print(f"Langs:\n {"\n".join(langs)}")


for lang in langs:
    total = results_by_lang[lang]["total"]
    correct = results_by_lang[lang]["correct"]
    print(f"{round(correct*100/total, 1)}")

# Calculate overall accuracy
total_correct = sum([results["correct"] for results in results_by_lang.values()])
total = sum([results["total"] for results in results_by_lang.values()])
print(round(total_correct*100/total, 1))

print("Totals")
# Print totals

for accent in langs:
    total = results_by_lang[accent]["total"]
    print(f"{total}")

print(f"Number of langs: {len(langs)}")