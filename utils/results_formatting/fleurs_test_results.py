#%%
import pickle as pkl
from collections import defaultdict
import pandas as pd
import os, sys
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils")
from lang_code_maps import vl107_to_fleurs_map

# %%

# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-1000/cnn-attentions-linear-8/lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-1000/cnn-attentions-linear-8/lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-10000/cnn-attentions-linear-4/lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-500/cnn-attentions-linear-4/lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/phoneseq_exps/vl107/wav2vec2-xlsr-53-espeak-cv-ft/attentions-linear-8/phoneseq_lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/phoneseq_exps/fleurs/wav2vec2-xlsr-53-espeak-cv-ft/attentions-linear-8/phoneseq_lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/fleurs/wav2vec2-large-xlsr-53-layer21-5000/cnn-attentions-linear-4/lid_model_outputs/predictions.pkl" 
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-10000/cnn-attentions-linear-4/lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-500/cnn-attentions-linear-4/lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-10000/cnn-attentions-linear-8/lid_model_outputs/predictions.pkl"
# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-500/cnn-attentions-linear-4/lid_model_outputs/fleurs_test_predictions.pkl"
predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/phoneseq_exps/fleurs/wav2vec2-xlsr-53-espeak-cv-ft/attentions-linear-8/phoneseq_lid_model_outputs/fleurs_test_predictions.pkl"
# output_path = "wav2vec2-base-layer8-100/accuracy.csv"

output_path = None

with open(predictions_path, "rb") as f:
    eval_data = pkl.load(f)

results_by_lang = defaultdict(lambda: defaultdict(int))
# print(list(zip(eval_data["preds"], eval_data["accents"], eval_data["labels"])))
for prediction, accent, label in zip(eval_data["preds"], eval_data["accents"], eval_data["labels"]):
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
print(f"Number of langs: {len(langs)}")

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
