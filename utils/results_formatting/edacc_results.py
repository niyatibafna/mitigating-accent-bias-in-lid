#%%
import pickle as pkl
from collections import defaultdict
import pandas as pd

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
predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-10000/cnn-attentions-linear-8/lid_model_outputs/predictions.pkl"
# output_path = "wav2vec2-base-layer8-100/accuracy.csv"

output_path = None

with open(predictions_path, "rb") as f:
    eval_data = pkl.load(f)

results_by_accent = defaultdict(lambda: defaultdict(int))
print(list(zip(eval_data["preds"], eval_data["accents"], eval_data["labels"])))
for prediction, accent, label in zip(eval_data["preds"], eval_data["accents"], eval_data["labels"]):
    if prediction == label:
        results_by_accent[accent]["correct"] += 1
    results_by_accent[accent]["total"] += 1
# %%

# Merge "us" and "american"  accents
results_by_accent["us"] = {k: results_by_accent["us"].get(k, 0) + results_by_accent["american"].get(k, 0) for k in set(results_by_accent["us"]) | set(results_by_accent["american"])}
# results_by_accent["us"]["total"] = sum([results_by_accent["us"]["total"], results_by_accent["american"]["total"]])
del results_by_accent["american"]


for accent, results in results_by_accent.items():
    results_by_accent[accent]["accuracy"] = round(results["correct"]/results["total"], 1)
    print(f"Accuracy for {accent}: {results['correct']/results['total']}")
    print(f"Total samples for {accent}: {results['total']}")
    print()


# %%
df = pd.DataFrame(results_by_accent).T
if output_path:
    df.to_csv(output_path)

# %%
accents = sorted(list(results_by_accent.keys()))
print(f"Accents:\n {"\n".join(accents)}")

for accent in accents:
    total = results_by_accent[accent]["total"]
    correct = results_by_accent[accent]["correct"]
    print(f"{round(correct*100/total, 1)}")

# Calculate overall accuracy
total_correct = sum([results["correct"] for results in results_by_accent.values()])
total = sum([results["total"] for results in results_by_accent.values()])
print(round(total_correct*100/total, 1))

print("Totals")
# Print totals

for accent in accents:
    total = results_by_accent[accent]["total"]
    print(f"{total}")
