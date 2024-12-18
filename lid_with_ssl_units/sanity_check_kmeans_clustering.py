'''
This code gets KMeans reps from `extracting_units_from_training_data.py`.
    - This code loads audio from the training data and extracts segment representations of 100ms segments.
    - It then trains KMeans on these segment representations.
    - It then predicts KMeans on these segment representations.
    - This gives us a sequence of KMeans centroids for each audio file.

We can then use these KMeans centroids to train a sequence classifier to predict LID labels.
'''
import torch
# torch.ones(1).cuda()
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
# Log to wandb
import wandb
import logging

from argparse import ArgumentParser

sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/")
from lid_with_ssl_units.extracting_units_from_training_data import load_vl107_all_langs, load_vl107_lang, prepare_dataset, collate_fn, extract_embeddings, extract_phoneme_segment_embeddings, compute_kmeans_reps

random.seed(42)

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger()

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
    

class LinearClassifieronPooledReps:

    '''
    This approach expects input of shape (batch_size, sequence_length, hidden_size).
    We'll mean pool over the sequence length and then pass it through a linear layer.
    '''

    def __init__(self, load_from_dir = False,  output_dir = None, batch_size = None, lr = None, num_epochs = None):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model'''
        if load_from_dir:
            model = torch.load(output_dir)
        else:
            model = LinearClassifieronPooledRepsModel(num_classes=107, hidden_size=768)
        self.model = model.cuda()

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs


    def train(self, train_dataset, dev_dataset, evaluate_steps = None):
        wandb.init(project="train_lid_on_ssl_linear", config={
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.lr
        })
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        steps = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch: {epoch}")
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                input_values = batch["input_values"].cuda()
                labels = batch["labels"].cuda()
                outputs = self.model(input_values)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")
                steps += 1
                if evaluate_steps and steps % evaluate_steps == 0:

                    # Evaluate
                    _, _, accuracy = self.evaluate(dev_dataset)
                    wandb.log({"loss": loss.item()})
                    wandb.log({"accuracy": accuracy, "epoch": epoch})
                    logger.info(f"Epoch: {epoch}, Steps: {steps}, Accuracy: {accuracy}")

            if not evaluate_steps:
                # Log per epoch if evaluate_steps is not provided
                wandb.log({"loss": loss.item()})
                # Evaluate
                _, _, accuracy = self.evaluate(dev_dataset)
                wandb.log({"accuracy": accuracy, "epoch": epoch})
                logger.info(f"Epoch: {epoch}, Accuracy: {accuracy}")


    def predict(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        self.model.eval()
        all_preds = []
        for batch in test_loader:
            input_values = batch["input_values"].cuda()
            with torch.no_grad():
                outputs = self.model(input_values)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
        return torch.cat(all_preds)


    def evaluate(self, test_dataset):
        preds = self.predict(test_dataset)
        labels = torch.cat([batch["labels"] for batch in DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)]).cuda()

        ###### DEBUG ######
        # logger.info(f"Number of samples: {len(labels)}")
        # logger.info(f"preds: {preds}")
        # logger.info(f"labels: {labels}")
        ###################
        correct = (preds == labels).sum().item()
        total = len(labels)
        logger.info(f"Accuracy: {correct/total}")
        return preds, labels, correct/total
    
    def save(self):
        torch.save(self.model, os.path.join(self.output_dir, "linear_classifier.pth"))



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
    parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory for LID model")
    parser.add_argument("--load_trained_from_dir", action="store_true", help="Load the model from output_dir")
    parser.add_argument("--lid_model_type", type=str, default="linear", help="Type of model to train")
    return parser.parse_args()


def get_centroid_sequence_reps(model_name, layer, dataset_dir, per_lang, lang, batch_size, kmeans_dir, training_units_dir):
    '''
    Get the centroid sequences for a language and Kmeans centroids. This is a sequence of KMeans centroids (indices) for each audio file, and the KMeans centroids.
    If the centroid sequences are already computed, load them. Else, compute them by loading the model, dataset, extracting embeddings, extracting phoneme segment embeddings, training kmeans, and saving the units and Kmeans centroids.
    This will save the centroid sequences to disk, as well as the trained KMeans centroids.
    '''
    # If centroid sequences are already computed, load them
    if os.path.exists(os.path.join(training_units_dir, f"all_segment_reps.pkl")):
        logger.info(f"LOADING CENTROID SEQUENCES FROM {training_units_dir}")
        with open(os.path.join(training_units_dir, f"all_segment_reps.pkl"), "rb") as f:
            centroid_sequences = pkl.load(f)
        with open(os.path.join(kmeans_dir, f"kmeans_model_centroids.npy"), "rb") as f:
            centroids = np.load(f)

        random.shuffle(centroid_sequences)
        centroid_sequences = centroid_sequences[:per_lang]

        return centroid_sequences, centroids

    logger.info(f"Computing centroid sequences for {lang}: loading model, dataset, extracting embeddings, extracting phoneme segment embeddings, training kmeans, and saving the units and Kmeans centroids...")
    # Load the model
    logger.info(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTraining.from_pretrained(model_name)

    # Load the dataset
    logger.info(f"Loading dataset from {dataset_dir}")
    if lang is None:
        train_dataset = load_vl107_all_langs(per_lang=per_lang, vl107_dir=dataset_dir)
    else:
        train_dataset = load_vl107_lang(lang=lang, per_lang=per_lang, vl107_dir=dataset_dir)
    
    train_dataset = train_dataset.map(prepare_dataset, fn_kwargs = {"processor": processor} , batched=True, batch_size=batch_size, remove_columns=["signal", "lang"])
    # Shape of input
    logger.info(torch.tensor(train_dataset[0]["input_values"]).shape)    

    dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

    # Extract embeddings
    logger.info(f"Extracting embeddings from layer {layer}")
    all_full_reps, _, _ = extract_embeddings(training_units_dir, model = model, dataloader = dataloader, layer = layer)
    logger.info(f"Number of representations: {len(all_full_reps)}")

    # Extract phoneme segment embeddings
    logger.info("Extracting phoneme segment embeddings")
    all_segment_reps = extract_phoneme_segment_embeddings(training_units_dir, all_full_reps)
    logger.info(f"Number of segment representations: {len(all_segment_reps)}")

    # Train kmeans and save the units
    logger.info(f"Training kmeans on segment representations or using trained KMeans in output_dir")  
    centroid_sequences = compute_kmeans_reps(all_segment_reps, kmeans_dir = kmeans_dir, output_dir = training_units_dir, n_clusters = 100, save = True)
    with open(os.path.join(kmeans_dir, f"kmeans_model_centroids.npy"), "rb") as f:
        centroids = np.load(f)

    random.shuffle(centroid_sequences)
    centroid_sequences = centroid_sequences[:per_lang]

    return centroid_sequences, centroids


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
    codevector_reps = []
    for centroid_sequence in centroid_sequences:
        codevector_reps.append([centroids[centroid] for centroid in centroid_sequence])

    batch["sequence"] = codevector_reps
    return batch


def construct_lid_dataset_codevectors(dataset_dir, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir):
    '''
    Construct the LID dataset using the KMeans centroids as codevectors
    Shape of each sample: {"sequence": np.ndarray(shape=(sequence_length,768), dtype=int), "lang": str}
    '''
    global centroids

    ### CHANGE THIS TO DATASET_DIR
    # langs = os.listdir(training_units_dir)

    langs = os.listdir(dataset_dir)

    #### For now, we'll just use langs that have non-empty directories in the output_dir ####
    # langs = [lang for lang in langs if len(os.listdir(os.path.join(training_units_dir, lang))) > 0]

    lang2idx = {lang: idx for idx, lang in enumerate(langs)}
    idx2lang = {idx: lang for lang, idx in lang2idx.items()}
    # langs = ["ar", "hi", "en"]
    logger.info(f"Number of languages: {len(langs)}")
    lid_dataset = []
    for lang in langs:
        logger.info(f"Processing language: {lang}")
        training_units_dir_lang = os.path.join(training_units_dir, lang)
        # kmeans_output_dir = os.path.join(training_units_dir, lang)
        centroid_sequences, centroids = get_centroid_sequence_reps(model_name, layer, dataset_dir, per_lang, lang, batch_size, kmeans_dir, training_units_dir_lang)
        
        lid_dataset.extend([{"sequence": centroid_sequence, "lang": lang2idx[lang]} for centroid_sequence in centroid_sequences])
    
    logger.info(f"Number of samples: {len(lid_dataset)}")

    ########## REMOVE ##########
    # lid_dataset = random.sample(lid_dataset, 50000)

    
    lid_dataset = {"sequence": [item["sequence"] for item in lid_dataset],\
                    "lang": [item["lang"] for item in lid_dataset]}
    lid_dataset = Dataset.from_dict(lid_dataset)

    logger.info(f"Total number of samples: {len(lid_dataset)}")
    logger.info(f"Example sequence: {len(lid_dataset[0]["sequence"])}, Language: {idx2lang[lid_dataset[0]['lang']]}, Length: {len(lid_dataset[0]['sequence'])}")

    # lid_dataset.set_transform(map_get_codevectors_reps, columns=["sequence", "lang"])
    lid_dataset = lid_dataset.map(map_get_codevectors_reps, batched=True, batch_size=1000)

    train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset, dev_size = 1000, test_size = 1000)

    return train_dataset, dev_dataset, test_dataset, lang2idx, idx2lang

def collate_fn(batch):
    '''Collate function for the dataloader. Batch comes from lid_dataset. Labels are integers representing the language'''
    input_values = [torch.tensor(item["sequence"]) for item in batch]
    labels = [item["lang"] for item in batch]    
    return {"input_values": torch.stack(input_values), "labels": torch.tensor(labels)}

def main():

    logger.info("Plotting KMeans centroids for all languages")

    args = parse_args()
    model_name = args.model_name
    layer = args.layer
    dataset_dir = args.dataset_dir
    training_units_dir = args.training_units_dir
    kmeans_dir = args.kmeans_dir
    per_lang = args.per_lang
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    output_dir = args.output_dir
    load_trained_from_dir = args.load_trained_from_dir
    lid_model_type = args.lid_model_type
    os.makedirs(output_dir, exist_ok=True)    


    train_dataset, dev_dataset, test_dataset, _, idx2lang = construct_lid_dataset_codevectors(dataset_dir, per_lang, model_name, layer, batch_size, kmeans_dir, training_units_dir)

    # Let's plot the pooled codevectors for each language
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    vectors = []
    labels = []

    codevectors = train_dataset["sequence"]
    labels = train_dataset["lang"]

    vectors = np.mean(codevectors, axis=1)
    print(f"Dimension of pooled_reps: {vectors.shape}")
    labels = np.array(labels)

    # for i in range(len(train_dataset)):
    #     print(f"Lenth of sequence: {len(train_dataset[i]['sequence'])}")
    #     print(f"Dimension of sequence: {len(train_dataset[i]['sequence'][0])}")
    #     pooled_rep = np.mean(train_dataset[i]["sequence"], axis=0)
    #     print(f"Dimension of pooled_rep: {pooled_rep.shape}")
    #     vectors.append(pooled_rep)
    #     labels.append(idx2lang[train_dataset[i]["lang"]])

    # vectors = np.array(vectors)
    # print(f"Dimension of vectors: {vectors.shape}")
    

    # PCA
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(vectors)
    
    # TSNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(pca_result)

    plt.figure(figsize=(10,10))
    for lang in np.unique(labels[:30]):
        plt.scatter(tsne_result[labels == lang, 0], tsne_result[labels == lang, 1], label=lang)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "pooled_codevectors.png"))





    
if __name__ == "__main__":

    main()