#%%
import pickle as pkl
from collections import defaultdict
import pandas as pd

# %%

# predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-1000/cnn-attentions-linear-8/lid_model_outputs/predictions.pkl"
predictions_path = "/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-1000/cnn-attentions-linear-8/lid_model_outputs/predictions.pkl"
# output_path = "wav2vec2-base-layer8-100/accuracy.csv"
output_path = None

with open(predictions_path, "rb") as f:
    eval_data = pkl.load(f)

results_by_accent = defaultdict(lambda: defaultdict(int))
for prediction, accent, label in zip(eval_data["preds"], eval_data["accents"], eval_data["labels"]):
    if prediction == label:
        results_by_accent[accent]["correct"] += 1
    results_by_accent[accent]["total"] += 1
# %%
for accent, results in results_by_accent.items():
    results_by_accent[accent]["accuracy"] = round(results["correct"]/results["total"], 1)
    print(f"Accuracy for {accent}: {results['correct']/results['total']}")
    print(f"Total samples for {accent}: {results['total']}")
    print()

# %%
df = pd.DataFrame(results_by_accent).T
if output_path:
    df.to_csv(output_path)

#%%

from collections import Counter
print(Counter(eval_data["preds"])["en"])
# %%
