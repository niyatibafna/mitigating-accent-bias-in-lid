'''
This code gets KMeans reps from `extracting_units_from_training_data.py`.
    - This code loads audio from the training data and extracts segment representations of 100ms segments.
    - It then trains KMeans on these segment representations.
    - It then predicts KMeans on these segment representations.
    - This gives us a sequence of KMeans centroids for each audio file.

We can then use these KMeans centroids to train a sequence classifier to predict LID labels.
'''
import torch
torch.ones(1).cuda()
import os, sys
import numpy as np
import random
from collections import defaultdict
import torchaudio
from transformers import AutoProcessor, WavLMModel
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining, Wav2Vec2Processor
from datasets import load_dataset, Dataset, Audio
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
import re
# Log to wandb
import wandb
import logging
import json

from argparse import ArgumentParser

sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/")
from lid_with_ssl_units.extracting_units_from_training_data import prepare_dataset, extract_embeddings, extract_phoneme_segment_embeddings, compute_kmeans_reps
from lid_with_ssl_units.extracting_units_from_training_data import collate_fn as collate_fn_for_extracting_units
sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/dataloading/")
from dataset_loader import load_lid_dataset
from vl107 import load_vl107
from edacc import load_edacc


random.seed(42)

global logger

def get_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class LinearClassifieronPooledRepsModel(torch.nn.Module):
    '''
    Linear classifier on pooled representations
    '''
    def __init__(self, num_classes, hidden_size):
        super(LinearClassifieronPooledRepsModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)
        # We want to pool over the sequence length and then pass it through a linear layer
        x = torch.mean(x, dim=1)
        return self.linear(x)
    

class LinearClassifieronCNNsModel(torch.nn.Module):
    '''
    Linear classifier on pooled representations
    '''
    def __init__(self, num_classes, hidden_size):
        super(LinearClassifieronCNNsModel, self).__init__()
        # Input: (batch_size, sequence_size, hidden_size)
        # Reduce the sequence size by half
        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv1d(hidden_size//2, hidden_size//4, kernel_size=3, stride=2)
        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(hidden_size//4, num_classes)


    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)
        
        x = x.permute(0, 2, 1)
        # Shape of input: (batch_size, hidden_size, sequence_size)
        x = self.conv1(x)
        # Shape of input: (batch_size, hidden_size//2, sequence_size//2)
        x = self.relu(x)
        x = self.conv2(x)
        # Shape of input: (batch_size, hidden_size//4, sequence_size//4)
        x = self.relu(x)
        x = torch.mean(x, dim=2)
        # Shape of input: (batch_size, hidden_size//4)
        return self.linear(x)
    

class LinearClassifiereonAttentionLayersModel(torch.nn.Module):
    '''
    Linear classifier on pooled representations
    '''
    def __init__(self, num_classes, hidden_size, num_attention_layers, attention_dim):
        super(LinearClassifiereonAttentionLayersModel, self).__init__()
        # Model architecture: input --> linear --> relu --> cnn (seq_len reduced by half) --> relu --> attention layers (2) --> linear
        
        self.hidden_size = hidden_size
        self.num_attention_layers = num_attention_layers
        self.attention_dim = attention_dim

        self.linear_input = torch.nn.Linear(hidden_size, attention_dim)

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(attention_dim, attention_dim, kernel_size=3, stride=2)

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=attention_dim, 
                                                                   nhead=8, 
                                                                   dim_feedforward=256,
                                                                    dropout=0.1,
                                                                    activation='relu')
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=num_attention_layers)

        self.linear_output = torch.nn.Linear(attention_dim, num_classes)
    

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)
        x = self.linear_input(x)
        x = self.relu(x)
        # Shape of input: (batch_size, sequence_length, attention_dim)
        x = x.permute(0, 2, 1)
        # Shape of input: (batch_size, attention_dim, sequence_length)
        x = self.conv1(x)
        # Shape of input: (batch_size, attention_dim, sequence_length//2)
        x = self.relu(x)
        # Correct shape for transformer: (sequence_length//2, batch_size, attention_dim)
        x = x.permute(2, 0, 1)
        # Shape of input: (sequence_length//2, batch_size, attention_dim)
        x = self.transformer(x)
        # Shape of input: (sequence_length//2, batch_size, attention_dim)
        x = torch.mean(x, dim=0)
        # Shape of input: (batch_size, attention_dim)
        x = self.linear_output(x)
        return x
        

class PostEncoder:

    def __init__(self, load_from_dir = False,  output_dir = None, batch_size = None, lr = None, num_epochs = None):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model'''

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        

    def train(self, train_dataset, dev_dataset, collate_fn, evaluate_steps = None):

        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Please run on a machine with CUDA")
            raise ValueError("CUDA is not available. Please run on a machine with CUDA")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model = self.model.cuda()

        steps = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch: {epoch}")
            self.model.train()
            for batch in train_loader:
                logger.info(f"Steps: {steps}")
                optimizer.zero_grad()
                input_values = batch["input_values"]
                labels = batch["labels"]
                assert self.model is not None and next(self.model.parameters()).is_cuda, "Model is not on GPU!"
                assert input_values.is_cuda, "Input is not on GPU!"
                assert labels.is_cuda, "Labels are not on GPU!"
                outputs = self.model(input_values)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")
                wandb.log({"loss": loss.item()})
                steps += 1
                if evaluate_steps and steps % evaluate_steps == 0:

                    # Evaluate
                    *_, accuracy = self.evaluate(dev_dataset)
                    wandb.log({"loss": loss.item()})
                    wandb.log({"accuracy": accuracy, "epoch": epoch})
                    logger.info(f"Epoch: {epoch}, Steps: {steps}, Accuracy: {accuracy}")


            if not evaluate_steps:
                # Log per epoch if evaluate_steps is not provided
                wandb.log({"loss": loss.item()})
                # Evaluate
                *_, accuracy = self.evaluate(dev_dataset)
                wandb.log({"accuracy": accuracy, "epoch": epoch})
                logger.info(f"Epoch: {epoch}, Accuracy: {accuracy}")

            # Save the model after each epoch
            self.save()


    def predict(self, test_dataset, collate_fn):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        all_preds = []
        all_labels = []
        all_accents = []
        all_audio_files = []
        for batch in test_loader:
            input_values = batch["input_values"]
            with torch.no_grad():
                outputs = self.model(input_values)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_labels.append(batch["labels"])
                all_accents.extend(batch["accents"])
                all_audio_files.extend(batch["audio_files"])
        return torch.cat(all_preds), torch.cat(all_labels), all_accents, all_audio_files


    def evaluate(self, test_dataset, collate_fn):
        preds, labels, accents, audio_files = self.predict(test_dataset)

        preds = preds.cpu()
        labels = labels.cpu()
        ###### DEBUG ######
        # logger.info(f"Number of samples: {len(labels)}")
        # logger.info(f"preds: {preds}")
        # logger.info(f"labels: {labels}")
        ###################
        correct = (preds == labels).sum().item()
        total = len(labels)
        logger.info(f"Accuracy: {correct/total}")
        return preds, labels, accents, audio_files, correct/total
    
    def save(self, model_filename):
        torch.save(self.model, os.path.join(self.output_dir, model_filename))


class LinearClassifieronPooledReps(PostEncoder):

    '''
    This approach expects input of shape (batch_size, sequence_length, hidden_size).
    We'll mean pool over the sequence length and then pass it through a linear layer.
    '''

    def __init__(self, num_classes, hidden_size, load_from_dir = False,  output_dir = None, batch_size = None, lr = None, num_epochs = None):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model'''
        if load_from_dir:
            model = torch.load(os.path.join(output_dir, "linear_classifier.pth"))
        else:
            model = LinearClassifieronPooledRepsModel(num_classes=num_classes, hidden_size=hidden_size)
        self.model = model.cuda()

        super().__init__(load_from_dir, output_dir, batch_size, lr, num_epochs)
        


    def train(self, train_dataset, dev_dataset, evaluate_steps = None):
        wandb.init(project="train_lid_on_ssl_linear", config={
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.lr,
            "output_dir": self.output_dir
        })
        super().train(train_dataset, dev_dataset, collate_fn, evaluate_steps)


    def predict(self, test_dataset):
        return super().predict(test_dataset, collate_fn)


    def evaluate(self, test_dataset):
        return super().evaluate(test_dataset, collate_fn)
    
    def save(self):
        super().save("linear_classifier.pth")


class LinearClassifieronCNNs(PostEncoder):

    '''
    This approach expects input of shape (batch_size, sequence_length, hidden_size).
    We'll mean pool over the sequence length and then pass it through a linear layer.
    '''

    def __init__(self, num_classes, sequence_size, hidden_size, load_from_dir = False, output_dir = None, batch_size = None, lr = None, num_epochs = None):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model'''
        if load_from_dir:
            model = torch.load(os.path.join(output_dir, "cnns2-linear_classifier.pth"))
        else:
            model = LinearClassifieronCNNsModel(num_classes=num_classes, sequence_size=sequence_size, hidden_size=hidden_size)

        self.model = model.cuda()

        super().__init__(load_from_dir, output_dir, batch_size, lr, num_epochs)
        


    def train(self, train_dataset, dev_dataset, evaluate_steps = None):
        wandb.init(project="train_lid_on_ssl_cnns2-linear", config={
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.lr,
            "output_dir": self.output_dir
        })
        super().train(train_dataset, dev_dataset, collate_fn, evaluate_steps)


    def predict(self, test_dataset):
        return super().predict(test_dataset, collate_fn)


    def evaluate(self, test_dataset):
        return super().evaluate(test_dataset, collate_fn)
    
    def save(self):
        super().save("cnns2-linear_classifier.pth")

class LinearClassifiereonAttentionLayers(PostEncoder):

    def __init__(self, num_classes, hidden_size, num_attention_layers, attention_dim, load_from_dir = False,  output_dir = None, batch_size = None, lr = None, num_epochs = None):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model'''
        if load_from_dir:
            model = torch.load(os.path.join(output_dir, f"attentions-{num_attention_layers}_classifier.pth"))
        else:
            model = LinearClassifiereonAttentionLayersModel(num_classes=num_classes, hidden_size=hidden_size, \
                                                            num_attention_layers=num_attention_layers, \
                                                            attention_dim=attention_dim)
        
        self.model = model.cuda()

        super().__init__(load_from_dir, output_dir, batch_size, lr, num_epochs)
        


    def train(self, train_dataset, dev_dataset, evaluate_steps = None):
        if "wav2vec2-base-layer8" in self.output_dir:
            kmeans_units = int(re.search(r"wav2vec2-base-layer8-(\d+)", self.output_dir).group(1))
        elif "wav2vec2-large-xlsr-53-layer21" in self.output_dir:
            kmeans_units = int(re.search(r"wav2vec2-large-xlsr-53-layer21-(\d+)", self.output_dir).group(1))
        else:
            kmeans_units = None
        wandb.init(project=f"train_lid_on_ssl_attentions-{self.model.num_attention_layers}", config={
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.lr,
            "num_attention_layers": self.model.num_attention_layers,
            "attention_dim": self.model.attention_dim,
            "hidden_size": self.model.hidden_size,
            "output_dir": self.output_dir,
            "kmeans_units": kmeans_units
        })
        super().train(train_dataset, dev_dataset, collate_fn, evaluate_steps)


    def predict(self, test_dataset):
        return super().predict(test_dataset, collate_fn)


    def evaluate(self, test_dataset):
        return super().evaluate(test_dataset, collate_fn)
    
    def save(self):
        super().save(f"attentions-{self.model.num_attention_layers}_classifier.pth")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="Model name")
    parser.add_argument("--layer", type=int, default=8, help="Layer to extract representations from")
    parser.add_argument("--dataset_dir", type=str, default="/exp/jvillalba/corpora/voxlingua107", help="Directory containing audio files")
    parser.add_argument("--training_units_dir", type=str, default=None, help="Directory containing the training units if already computed")
    parser.add_argument("--kmeans_dir", type=str, default=None, help="Directory to save or load kmeans model")
    parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--evaluate_steps", type=int, default=None, help="Evaluate every n steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory for LID model")
    parser.add_argument("--load_trained_from_dir", action="store_true", help="Load the model from output_dir")
    parser.add_argument("--lid_model_type", type=str, default="linear", help="Type of model to train")
    parser.add_argument("--logfile", type=str, default="train.log", help="Log file")
    parser.add_argument("--num_attention_layers", type=int, default=None, help="Number of attention layers")

    parser.add_argument("--only_eval", action="store_true", help="Only evaluate the model")
    parser.add_argument("--eval_dataset_dir", type=str, default=None, help="Directory containing evaluation dataset")
    parser.add_argument("--eval_units_dir", type=str, default=None, help="Directory containing the evaluation units if already computed")
    return parser.parse_args()


def get_centroid_sequence_reps(model_name, layer, dataset_dir, per_lang, lang, batch_size, kmeans_dir, training_units_dir):
    '''
    Get the centroid sequences for a language and Kmeans centroids. This is a sequence of KMeans centroids (indices) for each audio file, and the KMeans centroids.
    If the centroid sequences are already computed, load them. Else, compute them by loading the model, dataset, extracting embeddings, extracting phoneme segment embeddings, training kmeans, and saving the units and Kmeans centroids.
    This will save the centroid sequences to disk, as well as the trained KMeans centroids.

    Returns: 
    data = {
        "kmeans_reps": kmeans_reps,
        "all_lengths": all_lengths,
        "all_audio_files": all_audio_files,
        "all_accents": all_accents,
        "all_langs": all_langs
    }
    centroids: np.ndarray(shape=(n_clusters, hidden_size), dtype=float)
    
    '''
    # If centroid sequences are already computed, load them
    if os.path.exists(os.path.join(training_units_dir, f"kmeans_data.pkl")):
        logger.info(f"LOADING CENTROID SEQUENCES FROM {training_units_dir}")
        with open(os.path.join(training_units_dir, f"kmeans_data.pkl"), "rb") as f:
            data = pkl.load(f)
        with open(os.path.join(kmeans_dir, f"kmeans_model_centroids.npy"), "rb") as f:
            centroids = np.load(f)

        # centroid_sequences = data["centroid_sequences"]
        # accents = data["accents"]

        # centroids_accents = list(zip(centroid_sequences, accents))
        # random.shuffle(centroids_accents)
        # centroids_accents = centroids_accents[:per_lang]
        # centroid_sequences, accents = zip(*centroids_accents)

        return data, centroids


    # raise ValueError("Centroid sequences not found. Please compute them first.")

    logger.info(f"Computing centroid sequences for {lang}: loading dataset, model, extracting embeddings, extracting phoneme segment embeddings, training kmeans, and saving the units and Kmeans centroids...")

    dataset = load_lid_dataset(dataset_dir, lang = lang, per_lang = per_lang)
    if dataset is None:
        return None, None

    # Load the model
    logger.info(f"Loading model: {model_name}")
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTraining.from_pretrained(model_name)

    # Load the dataset
    
    # logger.info(f"Loading dataset from {dataset_dir}")
    # if lang is None:
    #     train_dataset = load_vl107_all_langs(per_lang=per_lang, vl107_dir=dataset_dir)
    # else:
    #     train_dataset = load_vl107_lang(lang=lang, per_lang=per_lang, vl107_dir=dataset_dir)
    
    
    dataset = dataset.map(prepare_dataset, fn_kwargs = {"processor": processor} , batched=True, batch_size=batch_size, remove_columns=["signal"])
    # Shape of input
    logger.info(f"Shape of input: {torch.tensor(dataset[0]["input_values"]).shape}")
          
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn_for_extracting_units, batch_size=batch_size)

    logger.info(f"Extracting embeddings from layer {layer}")
    data = extract_embeddings(training_units_dir, model = model, dataloader = dataloader, layer = layer)
    logger.info(f"Number of representations: {len(data["all_full_reps"])}")

    # Extract phoneme segment embeddings
    logger.info("Extracting phoneme segment embeddings")
    data = extract_phoneme_segment_embeddings(training_units_dir, data)
    logger.info(f"Number of segment representations: {len(data["all_segment_reps"])}")

    # Train kmeans and save the units
    logger.info(f"Training kmeans on segment representations")
    # We pass n_clusters = None: we assume that the Kmeans model has already been trained and saved
    data = compute_kmeans_reps(data, kmeans_dir = kmeans_dir, output_dir = training_units_dir, n_clusters = None, save = True)

    with open(os.path.join(kmeans_dir, f"kmeans_model_centroids.npy"), "rb") as f:
        centroids = np.load(f)

    # random.shuffle(centroid_sequences)
    # centroid_sequences = centroid_sequences[:per_lang]
    
    ##### DO PER_LANG ##########

    return data, centroids


def get_dataset_splits(lid_dataset, dev_size = 0.05, test_size = 0.1):
    '''Split the dataset into train and dev'''
    if dev_size < 1:
        # Fraction
        num_dev = int(dev_size * len(lid_dataset))
        num_test = int(test_size * len(lid_dataset))
    else:
        # Number
        num_dev = dev_size
        num_test = test_size
    indices = list(range(len(lid_dataset)))
    random.shuffle(indices)
    dev_indices = indices[:num_dev]
    test_indices = indices[num_dev:num_dev+num_test]
    train_indices = indices[num_dev+num_test:]
    train_dataset = lid_dataset.select(train_indices)
    dev_dataset = lid_dataset.select(dev_indices)
    test_dataset = lid_dataset.select(test_indices)

    logger.info(f"Number of samples in train: {len(train_dataset)}")
    logger.info(f"Number of samples in dev: {len(dev_dataset)}")
    logger.info(f"Number of samples in test: {len(test_dataset)}")

    return train_dataset, dev_dataset, test_dataset


def construct_lid_dataset_indices(dataset_dir, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir):
    '''
    Construct the LID dataset using the KMeans centroids as codevectors
    Shape of each sample: {"sequence": np.ndarray(shape=(sequence_length), dtype=int), "lang": str}
    '''
    langs = os.listdir(dataset_dir)
    # langs = ["ar", "hi", "en"]
    logger.info(f"Number of languages: {len(langs)}")
    lid_dataset = []
    for lang in langs:
        logger.info(f"Processing language: {lang}")
        # kmeans_output_dir = os.path.join(output_dir, lang)
        training_units_dir = os.path.join(training_units_dir, lang)
        centroid_sequences, _ = get_centroid_sequence_reps(model_name, layer, dataset_dir, per_lang, lang, batch_size, kmeans_dir, training_units_dir)
        lid_dataset.extend([{"sequence": centroid_sequence, "lang": lang} for centroid_sequence in centroid_sequences])
    
    lid_dataset = {"sequence": [item["sequence"] for item in lid_dataset],\
                    "lang": [item["lang"] for item in lid_dataset]}
    lid_dataset = Dataset.from_dict(lid_dataset)
    logger.info(f"Total number of samples: {len(lid_dataset)}")
    logger.info(f"Example sequence: {lid_dataset[0]['sequence']}, Language: {lid_dataset[0]['lang']}, Length: {len(lid_dataset[0]['sequence'])}")

    train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset)
    return train_dataset, dev_dataset, test_dataset


def get_codevectors_reps(centroid_sequences, centroids):
    '''Get the codevector representations for each audio file'''
    codevector_reps = []
    for centroid_sequence in centroid_sequences:
        codevector_reps.append([centroids[centroid] for centroid in centroid_sequence])
    return codevector_reps

def map_get_codevectors_reps(batch):
    '''Get the codevector representations for each audio file'''
    global centroids
    centroid_sequences = batch["sequence"]
    # centroids = batch["centroids"]
    codevector_reps = centroids[centroid_sequences]
    batch["sequence"] = codevector_reps
    return batch

def get_lang2idx_map(dataset_dir):
    '''Get the mapping from language to index'''

    ### CHANGE THIS TO DATASET_DIR
    # langs = os.listdir(training_units_dir)

    if dataset_dir == "vl107":
        langs_dir = "/exp/jvillalba/corpora/voxlingua107"
    elif dataset_dir == "fleurs":
        langs_dir = "/export/common/data/corpora/fleurs/metadata"
    langs = sorted(os.listdir(langs_dir))
    lang2idx = {lang: idx for idx, lang in enumerate(langs)}
    idx2lang = {idx: lang for lang, idx in lang2idx.items()}
    # langs = ["ar", "hi", "en"]
    logger.info(f"Number of languages: {len(langs)}")
    return lang2idx, idx2lang, langs

def construct_lid_dataset_codevectors(dataset_dir, lang2idx, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir):
    '''
    Construct the LID dataset using the KMeans centroids as codevectors
    Shape of each sample: {"sequence": np.ndarray(shape=(sequence_length,768), dtype=int), "lang": str}
    '''
    global centroids
    langs = list(lang2idx.keys())
    idx2lang = {idx: lang for lang, idx in lang2idx.items()}
    
    lid_dataset = None
    centroids = None
    for lang in langs:
        logger.info(f"Processing language: {lang}")
        training_units_dir_lang = os.path.join(training_units_dir, lang)
        # kmeans_output_dir = os.path.join(training_units_dir, lang)
        data, local_centroids = get_centroid_sequence_reps(model_name, layer, dataset_dir, per_lang, lang, batch_size, kmeans_dir, training_units_dir_lang)
        
        if data is None: # This will be the case if some lang is not available for some dataset
            continue

        centroids = local_centroids

        if lid_dataset is None:
            lid_dataset = data
        else:
            for key in lid_dataset.keys():
                lid_dataset[key].extend(data[key])

    
    logger.info(f"Number of samples: {len(lid_dataset)}")

    ########## REMOVE ##########
    # lid_dataset = random.sample(lid_dataset, 50000)

    # lid_dataset: {
    #     "sequences": kmeans_reps,
    #     "all_lengths": all_lengths,
    #     "all_audio_files": all_audio_files,
    #     "all_accents": all_accents,
    #     "all_langs": all_langs
    # }

    # We may have a mismatch in language names between the dataset and the lang2idx map
    ## This happens because lang2idx is based on the training dataset, and we may be loading the eval dataset
    ## Since this only happens for the eval dataset i.e. for English, we can for now deal with it in a hacky way
    ## i.e. we first get the equivalent of "en" in the lang2idx map and then replace all instances of "en" in the dataset with the equivalent

    eng_eqv = [lang for lang in lang2idx if "en" in lang][0] # There should be only one
    for i, lang in enumerate(lid_dataset["all_langs"]):
        if lang not in lang2idx and "en" in lang:
            # Replace with the equivalent
            lid_dataset["all_langs"][i] = eng_eqv


    # Rename keys
    lid_dataset["sequence"] = lid_dataset["sequences"]
    lid_dataset.pop("sequences")
    lid_dataset["lang"] = [lang2idx[lang] for lang in lid_dataset["all_langs"]]
    lid_dataset.pop("all_langs")
    lid_dataset["accent"] = lid_dataset["all_accents"]
    lid_dataset.pop("all_accents")
    lid_dataset["audio_file"] = lid_dataset["all_audio_files"]
    lid_dataset.pop("all_audio_files")
    lid_dataset["length"] = lid_dataset["all_lengths"]
    lid_dataset.pop("all_lengths")    

    lid_dataset = Dataset.from_dict(lid_dataset)

    logger.info(f"Total number of samples: {len(lid_dataset)}")
    logger.info(f"Example sequence: Accent: {lid_dataset[0]['accent']}, Language: {idx2lang[lid_dataset[0]['lang']]}, Length: {len(lid_dataset[0]['sequence'])}")

    lid_dataset.set_transform(map_get_codevectors_reps, columns=["sequence", "lang", "accent", "audio_file"])

    return lid_dataset

    # train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset, dev_size = 1000, test_size = 1000)

    # return train_dataset, dev_dataset, test_dataset, lang2idx, idx2lang

def collate_fn(batch):
    '''Collate function for the dataloader. Batch comes from lid_dataset. Labels are integers representing the language'''
    input_values = [torch.tensor(item["sequence"]) for item in batch]
    labels = [item["lang"] for item in batch]    
    accents = [item["accent"] for item in batch]
    audio_files = [item["audio_file"] for item in batch]
    return {"input_values": torch.stack(input_values).cuda(), \
        "labels": torch.tensor(labels).cuda(), "accents": accents,\
            "audio_files": audio_files}

def main():
    global logger

    args = parse_args()
    model_name = args.model_name
    layer = args.layer
    dataset_dir = args.dataset_dir
    training_units_dir = args.training_units_dir
    kmeans_dir = args.kmeans_dir
    per_lang = args.per_lang
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    evaluate_steps = args.evaluate_steps
    output_dir = args.output_dir
    load_trained_from_dir = args.load_trained_from_dir
    lid_model_type = args.lid_model_type
    logfile = args.logfile
    num_attention_layers = args.num_attention_layers
    

    ### Put params in argparse
    only_eval = args.only_eval
    eval_dataset_dir = args.eval_dataset_dir
    eval_units_dir = args.eval_units_dir


    # logger.info(f"Model name: {model_name}, Layer: {layer}, Dataset dir: {dataset_dir}, Training units dir: {training_units_dir}, Kmeans dir: {kmeans_dir}, Per lang: {per_lang}, Num epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}, Output dir: {output_dir}, Load trained from dir: {load_trained_from_dir}")

    logger = get_logger(logfile)
    logger.info("Starting training/evaluating LID model on KMeans centroids")
    if only_eval:
        logger.info("ONLY EVALUATING!")

    os.makedirs(output_dir, exist_ok=True)    

    # Get dataset:
    ## For training, we get train, dev, test splits
    ## For *only* evaluation setting, we only get a test split from eval_dataset_dir

    lang2idx, idx2lang, _ = get_lang2idx_map(dataset_dir)
    # Only load training dataset if we are training
    if not only_eval:
        logger.info("Loading training dataset...")
        lid_dataset = construct_lid_dataset_codevectors(dataset_dir, lang2idx, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir)
        train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset, dev_size = 1000, test_size = 10000)
        # sequence_size = len(train_dataset[0]["sequence"])
        # train_dataset, dev_dataset, test_dataset, _, idx2lang = construct_lid_dataset_codevectors(dataset_dir, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir)

    # Get LID model depending on lid_model_type

    assert lid_model_type in ["linear", "cnn2-linear", "cnn-attentions2-linear", "cnn-attentions-linear"], "Invalid LID model type"
    if only_eval:
        assert load_trained_from_dir, "If only evaluating, you must load the model from a directory"

    logger.info(f"Type of LID model: {lid_model_type}")
    
    if lid_model_type == "linear":
        # train_dataset, dev_dataset, test_dataset, _, idx2lang = construct_lid_dataset_codevectors(dataset_dir, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir)
        hidden_size = 768 if "wav2vec2-base" in model_name else 1024
        num_classes = len(idx2lang)
        logger.info(f"Num classes: {num_classes}")
        lid_model = LinearClassifieronPooledReps(num_classes = num_classes, hidden_size = hidden_size, \
                                                  load_from_dir = load_trained_from_dir, output_dir = output_dir, \
                                                    batch_size = batch_size, lr = lr, num_epochs = num_epochs)

    elif lid_model_type == "cnn2-linear":
        # train_dataset, dev_dataset, test_dataset, _, idx2lang = construct_lid_dataset_codevectors(dataset_dir, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir)
        hidden_size = 768 if "wav2vec2-base" in model_name else 1024
        num_classes = len(idx2lang)
        logger.info(f"Num classes: {num_classes}")
        lid_model = LinearClassifieronCNNs(num_classes=num_classes, sequence_size=sequence_size, hidden_size=hidden_size, \
                                           load_from_dir = load_trained_from_dir, output_dir = output_dir, \
                                           batch_size = batch_size, lr = lr, num_epochs = num_epochs)
    
    elif lid_model_type == "cnn-attentions2-linear":
        hidden_size = 768 if "wav2vec2-base" in model_name else 1024
        num_classes = len(idx2lang)
        logger.info(f"Num classes: {num_classes}")
        lid_model = LinearClassifiereonAttentionLayers(num_classes=num_classes, hidden_size=hidden_size, num_attention_layers=2, attention_dim=128, \
                                                        load_from_dir = load_trained_from_dir, output_dir = output_dir, \
                                                        batch_size = batch_size, lr = lr, num_epochs = num_epochs)

    elif lid_model_type == "cnn-attentions-linear":
        assert num_attention_layers is not None, "Number of attention layers not provided"
        hidden_size = 768 if "wav2vec2-base" in model_name else 1024
        num_classes = len(idx2lang)
        logger.info(f"Num classes: {num_classes}")
        lid_model = LinearClassifiereonAttentionLayers(num_classes=num_classes, hidden_size=hidden_size, num_attention_layers=num_attention_layers, attention_dim=128, \
                                                        load_from_dir = load_trained_from_dir, output_dir = output_dir, \
                                                        batch_size = batch_size, lr = lr, num_epochs = num_epochs)

    
    # Train the model
    if not only_eval:
        logger.info(f"Training model...")
        lid_model.train(train_dataset, dev_dataset, evaluate_steps = evaluate_steps)
        lid_model.save()

        # Evaluate the model
        logger.info(f"Evaluating model on test split of train dataset...")
        preds, labels, accents, audio_files, accuracy = lid_model.evaluate(test_dataset)
        logger.info(f"Accuracy: {accuracy}")

        # Save the predictions
        ## We'll save a list of audio files, their predicted labels, and their true labels
        # audio_files_test = [item["audio_file"] for item in test_dataset]
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        # accents = accents.cpu().numpy()
        preds = [idx2lang[pred] for pred in preds]
        labels = [idx2lang[label] for label in labels]

        with open(os.path.join(output_dir, "testset_predictions.pkl"), "wb") as f:
            # pkl.dump({"audio_files": audio_files_test, "preds": preds, "labels": labels}, f)
            pkl.dump({"preds": preds, "labels": labels, "accents": accents, "audio_files": audio_files}, f)


        with open(os.path.join(output_dir, f"eval_accuracy.json"), "w") as f:
            json.dump({f"{dataset_dir}_test_accuracy": accuracy}, f)



    # Evaluate the model on eval dataset if provided

    if eval_dataset_dir:
        logger.info(f"Evaluating model on eval dataset...")
        print(f"lang2idx: {lang2idx}")
        eval_dataset = construct_lid_dataset_codevectors(eval_dataset_dir, lang2idx, per_lang, model_name, layer, batch_size, kmeans_dir, eval_units_dir)
        sequence_size = len(eval_dataset[0]["sequence"])

        logger.info(f"Evaluating model on eval dataset...")
        preds, labels, accents, audio_files, accuracy = lid_model.evaluate(eval_dataset)
        logger.info(f"Accuracy: {accuracy}")


        # Save the predictions
        ## We'll save a list of audio files, their predicted labels, and their true labels
        # audio_files_test = [item["audio_file"] for item in test_dataset]
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        # accents = accents.cpu().numpy()
        preds = [idx2lang[pred] for pred in preds]
        labels = [idx2lang[label] for label in labels]

        with open(os.path.join(output_dir, f"{eval_dataset_dir}_predictions.pkl"), "wb") as f:
            # pkl.dump({"audio_files": audio_files_test, "preds": preds, "labels": labels}, f)
            pkl.dump({"preds": preds, "labels": labels, "accents": accents, "audio_files": audio_files}, f)

        # Save accuracy to JSON file
        if os.path.exists(os.path.join(output_dir, f"eval_accuracy.json")):
            results = json.load(open(os.path.join(output_dir, f"eval_accuracy.json")))
        else:
            results = {}
        
        results[f"{eval_dataset_dir}_accuracy"] = accuracy

        with open(os.path.join(output_dir, f"eval_accuracy.json"), "w") as f:
            json.dump(results, f, indent=4)



if __name__ == "__main__":

    main()