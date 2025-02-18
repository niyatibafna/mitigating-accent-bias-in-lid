'''
We will use both acoustic features and phonetic labels to train LID.
For the acoustic features, we directly use the representations of the ECAPA-TDNN model.
We concatenate these representations with learnt representations of the phonetic labels.
We then use a shallow network on top of these representations to train LID.

ET reps: 256 dim , phonetic reps: 256 dim
HF dataset:
{'lang': str, 'accent': str, \
    'audio_file': str, \
    'reps': torch.tensor, \
    'phone_sequence': list}

'''

import torch
if torch.cuda.is_available():
    torch.ones(1).cuda()
import os, sys
import json
import numpy as np
import random
from collections import defaultdict
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
import re
# Log to wandb
import wandb
import logging
from datasets import concatenate_datasets


from argparse import ArgumentParser

# sys.path.append("/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/phoneseq_dataloading")
from encode_transcribe_and_get_ssl_units import main as load_or_compute_encode_transcribe_ssl_dataset

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


class RepsDUseqsEmbedLinearClassifiereonAttentionLayersModel(torch.nn.Module):
    '''
    Linear classifier on pooled representations
    '''
    def __init__(self, num_classes, vocab_size, hidden_size, num_attention_layers, attention_dim, reps_dim):
        super(RepsDUseqsEmbedLinearClassifiereonAttentionLayersModel, self).__init__()
        # Model architecture: input --> linear --> relu --> cnn (seq_len reduced by half) --> relu --> attention layers (2) --> linear
        
        self.vocab_size = vocab_size # Size of the duseq vocabulary i.e. number of kmeans centroids
        self.hidden_size = hidden_size
        self.num_attention_layers = num_attention_layers
        self.attention_dim = attention_dim
        self.reps_dim = reps_dim

        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

        self.linear_input = torch.nn.Linear(hidden_size, attention_dim)

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(attention_dim, attention_dim, kernel_size=3, stride=2)

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=attention_dim, 
                                                                   nhead=8, 
                                                                   dim_feedforward=256,
                                                                    dropout=0.1,
                                                                    activation='relu')
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=num_attention_layers)

        self.linear_output = torch.nn.Linear(attention_dim + reps_dim, num_classes)
    

    def forward(self, input):
        
        x = input[0]
        # x: (batch_size, sequence_length)
        x = self.embedding(x)
        # Shape of input: (batch_size, sequence_length, hidden_size)
        x = self.linear_input(x)
        x = self.relu(x)
        # Shape of input: (batch_size, sequence_length, attention_dim)
        x = x.permute(0, 2, 1)
        # Shape of input: (batch_size, attention_dim, sequence_length)

        ######### THIS IS FOR THE CNN MODEL #########
        x = self.conv1(x)
        # Shape of input: (batch_size, attention_dim, sequence_length//2)
        x = self.relu(x)
        # Correct shape for transformer: (sequence_length//2, batch_size, attention_dim)
        x = x.permute(2, 0, 1)
        # Shape of input: (sequence_length//2, batch_size, attention_dim)
        x = self.transformer(x)
        # Shape of input: (sequence_length//2, batch_size, attention_dim)
        x = x.permute(1, 0, 2)
        # Shape of input: (batch_size, sequence_length//2, attention_dim)

        ######## THIS IS FOR THE NO CNN MODEL #########
        # # Correct shape for transformer: (sequence_length, batch_size, attention_dim)
        # x = x.permute(2, 0, 1)
        # # Shape of input: (sequence_length, batch_size, attention_dim)
        # x = self.transformer(x)
        # # Shape of input: (sequence_length, batch_size, attention_dim)
        # x = x.permute(1, 0, 2)
        # # Shape of input: (batch_size, sequence_length, attention_dim)


        # Average the attention vectors
        x = x.mean(dim=1)
        # Shape of input: (batch_size, attention_dim)

        reps = input[1]
        # Standardize the representations
        reps = reps / reps.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        x = torch.cat([x, reps], dim=1)
        # Shape of input: (batch_size, attention_dim + reps_dim)
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
                # assert input_values.is_cuda, "Input is not on GPU!"
                # assert labels.is_cuda, "Labels are not on GPU!"
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
            print(f"Saving model after epoch {epoch}")
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

class RepsDUseqsEmbedLinearClassifiereonAttentionLayers(PostEncoder):

    def __init__(self, vocab_size, num_classes, hidden_size, num_attention_layers, attention_dim, load_from_dir = False,  output_dir = None, batch_size = None, lr = None, num_epochs = None,\
                 reps_dim = None):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model'''
        if load_from_dir:
            model = torch.load(os.path.join(output_dir, f"reps-duseqs_attentions-{num_attention_layers}_classifier.pth"))
        else:
            model = RepsDUseqsEmbedLinearClassifiereonAttentionLayersModel(vocab_size=vocab_size, num_classes=num_classes, hidden_size=hidden_size, \
                                                            num_attention_layers=num_attention_layers, \
                                                            attention_dim=attention_dim, reps_dim=reps_dim)
        
        self.model = model.cuda()

        super().__init__(load_from_dir, output_dir, batch_size, lr, num_epochs)
        


    def train(self, train_dataset, dev_dataset, evaluate_steps = None):
        if "wav2vec2-base-layer8" in self.output_dir:
            kmeans_units = int(re.search(r"wav2vec2-base-layer8-(\d+)", self.output_dir).group(1))
        elif "wav2vec2-large-xlsr-53-layer21" in self.output_dir:
            kmeans_units = int(re.search(r"wav2vec2-large-xlsr-53-layer21-(\d+)", self.output_dir).group(1))
        else:
            kmeans_units = None
        wandb.init(project=f"train_lid_on_reps_duseqsembed_attentions-{self.model.num_attention_layers}", config={
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.lr,
            "num_attention_layers": self.model.num_attention_layers,
            "attention_dim": self.model.attention_dim,
            "hidden_size": self.model.hidden_size,
            "output_dir": self.output_dir,
            "kmeans_units": kmeans_units,
            "reps_dim": self.model.reps_dim
        })
        super().train(train_dataset, dev_dataset, collate_fn, evaluate_steps)


    def predict(self, test_dataset):
        return super().predict(test_dataset, collate_fn)


    def evaluate(self, test_dataset):
        return super().evaluate(test_dataset, collate_fn)
    
    def save(self):
        super().save(f"reps-duseqs_attentions-{self.model.num_attention_layers}_classifier.pth")



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Directory containing transcribed audio files")
    
    parser.add_argument("--transcriber_model", type=str, required=True, help="Model used to transcribe the audio files")
    parser.add_argument("--encoder_model", type=str, required=True, help="Model used to encode the audio files for acoustic representations")
    parser.add_argument("--ssl_model", type=str, required=True, help="Model used to get the SSL units")
    parser.add_argument("--kmeans_dir", type=str, required=True, help="Directory containing the kmeans units")
    parser.add_argument("--save_dataset_dir", type=str, required=True, help="Directory to save the dataset")

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
    parser.add_argument("--eval_dataset_name", type=str, default=None, help="Directory containing evaluation dataset")
    parser.add_argument("--save_eval_dataset_dir", type=str, default=None, help="Directory to save the evaluation dataset")
    return parser.parse_args()


def get_lang2idx_map(dataset_dir):
    '''Get the mapping from language to index'''

    global logger

    ### CHANGE THIS TO DATASET_DIR
    # langs = os.listdir(training_units_dir)

    if dataset_dir == "vl107":
        langs_dir = "/exp/jvillalba/corpora/voxlingua107"
    elif dataset_dir == "fleurs":
        langs_dir = "/export/common/data/corpora/fleurs/metadata"
    langs = sorted(os.listdir(langs_dir))

    #### For now, we'll just use langs that have non-empty directories in the output_dir ####
    # langs = [lang for lang in langs if len(os.listdir(os.path.join(training_units_dir, lang))) > 0]

    lang2idx = {lang: idx for idx, lang in enumerate(langs)}
    idx2lang = {idx: lang for lang, idx in lang2idx.items()}
    # langs = ["ar", "hi", "en"]
    logger.info(f"Number of languages: {len(langs)}")
    return lang2idx, idx2lang, langs


def get_dataset_splits(lid_dataset, dev_size = 0.05, test_size = 0.1):
    '''Split the dataset into train and dev'''
    global logger
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



def load_computed_dataset(transcriber_model, encoder_model, ssl_model, kmeans_dir, dataset_name, \
                                   langs, per_lang = None, batch_size = 1, output_dir = None, \
                                    log_file = None):

    all_lang_datasets = []

    ## FOR DEBUGGING #######
    # for lang in langs[:5]:
    for lang in langs:
        logger.info(f"Processing lang: {lang}")
        lang_dataset = load_or_compute_encode_transcribe_ssl_dataset(transcriber_model, encoder_model, ssl_model, kmeans_dir, dataset_name, per_lang=per_lang, lang=lang, \
            batch_size=batch_size, output_dir=output_dir, log_file=None)
        if lang_dataset is not None:
            all_lang_datasets.append(lang_dataset)

    lid_dataset = concatenate_datasets(all_lang_datasets)
    
    return lid_dataset



def collate_fn(batch):
    '''Collate function for the dataloader. Batch comes from lid_dataset. Labels are integers representing the language'''
    global lang2idx

    ssl_inputs = [torch.tensor(item["sequences"]) for item in batch]
    ssl_inputs = torch.stack(ssl_inputs)
    # Note that we do not need to pad because the sequences are already of the same length

    reps = [torch.tensor(item["reps"]) for item in batch]
    reps = torch.stack(reps)

    # Input is a tuple of (padded_sequences, reps)
    input_values = (ssl_inputs.cuda(), reps.cuda())
    
    # Prepare the labels
    # batch["label"] = [lang2idx[l] for l in batch["lang"]]
    labels = [lang2idx[item["lang"]] for item in batch]
    labels = torch.tensor(labels).cuda()

    # Prepare the accents
    accents = [item["accent"] for item in batch]  
    audio_files = [item["audio_file"] for item in batch]  

    # Input is a tuple of (ssl_units, reps)
    # Note that ssl_units are already embedded, of shape (batch_size, num_units, dim)
    # reps are of shape (batch_size, reps_dim)
    return {"input_values": input_values,
            "labels": labels, \
            "accents": accents, \
                "audio_files": audio_files}



def main():

    global logger, centroids, lang2idx

    args = parse_args()
    dataset_name = args.dataset_name

    transcriber_model = args.transcriber_model
    encoder_model = args.encoder_model
    ssl_model = args.ssl_model
    kmeans_dir = args.kmeans_dir
    save_dataset_dir = args.save_dataset_dir

    per_lang = args.per_lang
    lid_model_type = args.lid_model_type
    output_dir = args.output_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    evaluate_steps = args.evaluate_steps
    load_trained_from_dir = args.load_trained_from_dir
    num_attention_layers = args.num_attention_layers

    only_eval = args.only_eval
    eval_dataset_name = args.eval_dataset_name
    save_eval_dataset_dir = args.save_eval_dataset_dir

    logger = get_logger(args.logfile)

    lang2idx, idx2lang, langs = get_lang2idx_map(dataset_name)

    if not only_eval:
        # Load dataset
        lid_dataset = load_computed_dataset(transcriber_model=transcriber_model, encoder_model=encoder_model, ssl_model=ssl_model, kmeans_dir=kmeans_dir, \
            dataset_name=dataset_name, langs=langs, per_lang=per_lang, batch_size=batch_size, output_dir=save_dataset_dir, \
                log_file=None)

        print(f"Number of samples in dataset: {len(lid_dataset)}")
        # lid_dataset = lid_dataset.select(range(5000)) # For debugging
        
        # Split the dataset into training and validation
        train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset, dev_size = 1000, test_size = 10000)
        ### CHANGE THIS
        # train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset, dev_size = 50, test_size = 50)
    

    assert lid_model_type in ["attentions-linear"], "Invalid LID model type"
    if only_eval:
        assert load_trained_from_dir, "If only evaluating, you must load the model from a directory"

    logger.info(f"Type of LID model: {lid_model_type}")
    
    if lid_model_type == "attentions-linear":

        vocab_size = 1000 # Number of kmeans centroids. We will learn embeddings for them.
        hidden_size = 256
        reps_dim = 256 # train_dataset[0]["reps"].shape[0]
        num_classes = len(idx2lang)
        attention_dim = 128
        logger.info(f"Num classes: {num_classes}")
        lid_model = RepsDUseqsEmbedLinearClassifiereonAttentionLayers(vocab_size=vocab_size, num_classes=num_classes, hidden_size=hidden_size, num_attention_layers=num_attention_layers, \
                                                                 attention_dim=attention_dim, load_from_dir=load_trained_from_dir, output_dir=output_dir, \
                                                                    batch_size=batch_size, lr=lr, num_epochs=num_epochs, reps_dim=reps_dim)

    
    
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
            json.dump({f"{dataset_name}_test_accuracy": accuracy}, f)

    # Evaluate the model on eval dataset if provided

    if eval_dataset_name:
        logger.info(f"Evaluating model on eval dataset...")
        eval_dataset = load_computed_dataset(transcriber_model=transcriber_model, encoder_model=encoder_model, ssl_model=ssl_model, kmeans_dir=kmeans_dir, \
            dataset_name=eval_dataset_name, langs=langs, per_lang=per_lang, batch_size=batch_size, output_dir=save_eval_dataset_dir, \
                log_file=None)

        # Get the mapping from language to index
        ## Convert the phonetic sequences to integers, and map the languages to integers
        # eval_dataset = eval_dataset.map(map_tokenize_phoneme_labels, fn_kwargs={"phoneme2idx": phoneme2idx, "lang2idx": lang2idx}, \
        #     batched=True, \
        #     batch_size = 1000)
        # with open(os.path.join(kmeans_dir, f"kmeans_model_centroids.npy"), "rb") as f:
        #     centroids = np.load(f)
        
        # eval_dataset.set_transform(map_get_codevectors_reps)
        
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

        with open(os.path.join(output_dir, f"{eval_dataset_name}_predictions.pkl"), "wb") as f:
            # pkl.dump({"audio_files": audio_files_test, "preds": preds, "labels": labels}, f)
            pkl.dump({"preds": preds, "labels": labels, "accents": accents, "audio_files": audio_files}, f)
        
        # Save accuracy to JSON file

        if os.path.exists(os.path.join(output_dir, f"eval_accuracy.json")):
            results = json.load(open(os.path.join(output_dir, f"eval_accuracy.json")))
        else:
            results = {}
        
        results[f"{eval_dataset_name}_accuracy"] = accuracy

        with open(os.path.join(output_dir, f"eval_accuracy.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":

    main()
    