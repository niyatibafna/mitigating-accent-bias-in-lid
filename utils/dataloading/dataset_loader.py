import os, sys
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/dataloading")
from vl107 import load_vl107, load_vl107_lang
from edacc import load_edacc
from fleurs import load_fleurs
from cv import load_cv
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/")
from lang_code_maps import vl107_to_fleurs_map

def convert_to_target_code(dataset_name, dataset, target_code_type):
    '''
    Convert lang codes to target_code_type
    '''

    def filter_and_map(batch):
        filtered_indices = [i for i, lang in enumerate(batch["lang"]) if lang in fleurs_to_vl107]
        return {
                "signal": [batch["signal"][i] for i in filtered_indices],
                "lang": [fleurs_to_vl107[batch["lang"][i]] for i in filtered_indices],
                "accent": [batch["accent"][i] for i in filtered_indices],
                "audio_file": [batch["audio_file"][i] for i in filtered_indices],
            }

    if "fleurs" in dataset_name:
        if target_code_type == "vl107":
            vl107_to_fleurs = vl107_to_fleurs_map()
            fleurs_to_vl107 = {v: k for k, v in vl107_to_fleurs.items()} # This contains all the FLEURS lang codes that are in VL107
            dataset = dataset.map(filter_and_map, batched = True, batch_size = 1000, \
                                  num_proc = 4, writer_batch_size = 100)
    return dataset



def load_lid_dataset(dataset_name, per_lang = None, lang = None, split = "train", target_code_type = None):
    DATASET_REGISTRY = {
        "vl107": load_vl107,
        "edacc": load_edacc,
        "fleurs": load_fleurs,
        "cv": load_cv,
    }
    # if dataset_name not in DATASET_REGISTRY:
    #     print(f"Dataset {dataset_name} not found in registry")
    #     return None

    dataset = None
    if dataset_name == "vl107":
        dataset = DATASET_REGISTRY[dataset_name](per_lang = per_lang, lang = lang)
    elif dataset_name == "edacc":
        if lang and lang not in {"en", "en_us"}: # These are English codes in VL107 and EDACC, both are used to load EdAcc as an eval dataset
            return None
        dataset = DATASET_REGISTRY[dataset_name](num_samples = per_lang)
    elif dataset_name == "fleurs":
        dataset = DATASET_REGISTRY[dataset_name](per_lang = per_lang, lang = lang, split = "train")
    elif dataset_name == "fleurs_test":
        dataset = DATASET_REGISTRY["fleurs"](per_lang = per_lang, lang = lang, split = "test")
    elif dataset_name == "cv":
        if lang and lang not in {"en", "en_us"}:
            return None
        # If we are loading CV, we are loading the English eval dataset, and 
        ## we reuse the per_lang parameter to specify the number of samples to load per *accent*
        ## Change this if we want to load multiple languages
        dataset = DATASET_REGISTRY[dataset_name](per_accent = per_lang, split = split) # These are English codes in VL107 and EDACC, both are used to load CV as an eval dataset
    
    if target_code_type is not None and lang is None: # If lang is None, we deal with the labels in the loading logic.
        dataset = convert_to_target_code(dataset_name, dataset, target_code_type)

    return dataset
    
