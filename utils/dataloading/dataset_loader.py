import os, sys
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/dataloading")
from vl107 import load_vl107, load_vl107_lang
from edacc import load_edacc
from fleurs import load_fleurs


def load_lid_dataset(dataset_name, per_lang = None, lang = None, split = "train"):
    DATASET_REGISTRY = {
        "vl107": load_vl107,
        "edacc": load_edacc,
        "fleurs": load_fleurs
    }
    if dataset_name not in DATASET_REGISTRY:
        print(f"Dataset {dataset_name} not found in registry")
        return None

    if dataset_name == "vl107":
        return DATASET_REGISTRY[dataset_name](per_lang = per_lang, lang = lang)
    elif dataset_name == "edacc":
        if lang != "en":
            return None
        return DATASET_REGISTRY[dataset_name](num_samples = per_lang)
    elif dataset_name == "fleurs":
        return DATASET_REGISTRY[dataset_name](per_lang = per_lang, lang = lang, split = split)