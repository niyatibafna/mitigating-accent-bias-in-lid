import pandas as pd

# results_csv = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/preds/edacc_preds_new.csv"
dataset="cv"
results_csv = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/preds/cv_accents_confusion_matrix.csv"
with open(results_csv, "r") as f:
    results = pd.read_csv(f)

langs = list(results["Unnamed: 0"])
preds = {}
for accent in results:
    if accent == "Unnamed: 0":
        continue
    preds[accent] = {}
    preds[accent]["total"] = 0
    for i, val in enumerate(results[accent]):
        preds[accent][langs[i]] = val
        preds[accent]["total"] += val if not pd.isna(val) else 0


if dataset == "edacc":
    # Merge "us" and "american"  accents
    print(preds["us"]["total"])
    print(preds["american"]["total"])
    preds["us"] = {k: preds["us"].get(k, 0) + preds["american"].get(k, 0) for k in set(preds["us"]) | set(preds["american"])}
    del preds["american"]
    print(preds["us"]["total"])
accents = sorted(list(preds.keys()))
print(f"Accents\n: {"\n".join(accents)}")


for accent in accents:
    total = preds[accent]["total"]
    correct = preds[accent]["en: English"]
    # print(f"Accent: {accent}, Accuracy: {correct/total}")
    print(f"{round(correct*100/total, 1)}")

# Calculate overall accuracy
total_correct = sum([preds[accent]["en: English"] for accent in accents])
total = sum([preds[accent]["total"] for accent in accents])
print(round(total_correct*100/total, 1))

print("Totals")
# Print totals
for accent in accents:
    total = preds[accent]["total"]
    print(f"{total}")