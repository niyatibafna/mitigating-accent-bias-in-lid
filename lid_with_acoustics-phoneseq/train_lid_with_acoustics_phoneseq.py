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
from encode_and_transcribe_dataset import main as load_or_compute_encode_and_transcribe_dataset

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


class RepsPhoneseqsAttentionsLinearModel(torch.nn.Module):

    def __init__(self, vocab_size, num_classes, hidden_size, num_attention_layers, attention_dim, reps_dim, padding_idx):

        '''
        vocab_size: Size of the phone vocabulary
        num_classes: Number of languages
        hidden_size: Embeddings dimension
        num_attention_layers: Number of transformer layers
        attention_dim: Dimension of the attention vectors
        reps_dim: Dimension of the acoustic representations of the sequence (fixed length).
        padding_idx: Index of the padding token. Note that padding is only applied to the phone sequences.
        Input to this model:
        {"input_values": (padded_sequences, reps),
            "labels": labels, \
            "accents": accents}
        padded_sequences: [batch_size, seq_len], reps: [batch_size, reps_dim]
        labels: [batch_size], (mapped to integers using lang2idx)
        accents: [batch_size] (strings)
        '''

        super(RepsPhoneseqsAttentionsLinearModel, self).__init__()
        # Model architecture: input --> embedding --> linear --> transformer layers --> linear
        self.padding_idx = padding_idx
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
    
        self.hidden_size = hidden_size
        self.num_attention_layers = num_attention_layers
        self.attention_dim = attention_dim
        self.reps_dim = reps_dim

        self.linear_input = torch.nn.Linear(hidden_size, attention_dim)

        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=attention_dim, 
                                                                   nhead=8, 
                                                                   dim_feedforward=256,
                                                                    dropout=0.1,
                                                                    activation='relu')
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=num_attention_layers)
        
        # Linear layer to output the class probabilities. We concatenate the attention vectors with the acoustic representations 
        ## to create the input to the linear layer
        self.linear_output = torch.nn.Linear(attention_dim + reps_dim, num_classes)
    

    def forward(self, input):
        ''' x is a tuple of (padded_sequences, reps) where padded_sequences is a padded tensor of phone sequences and reps is a tensor of acoustic representations
        phone_input_values: [batch_size, seq_len], reps_input_values: [batch_size, hidden_size]
        x : [batch_size, seq_len]'''
        
        x = input[0]
        ## We need to ignore padded tokens
        padding_mask = x == self.padding_idx

        # Embedding layer
        x = self.embedding(x)
        # x: [batch_size, seq_len, hidden_size]
        x = self.linear_input(x)
        # x: [batch_size, seq_len, attention_dim]
        x = x.permute(1, 0, 2)
        # x: [seq_len, batch_size, attention_dim]
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # x: [seq_len, batch_size, attention_dim]
        x = x.permute(1, 0, 2)
        # x: [batch_size, seq_len, attention_dim]
        # Average the attention vectors, ignoring the padding tokens
        mask = ~padding_mask
        mask = mask.unsqueeze(-1)
        x = x * mask
        x = x.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        # x: [batch_size, attention_dim]

        # Now, we concatenate the acoustic representations with the attention vectors
        reps = input[1]
        # Standardize the representations 
        reps = reps / reps.norm(dim=1, keepdim=True)
        # Standardize the attention vectors
        x = x / x.norm(dim=1, keepdim=True)

        x = torch.cat([x, reps], dim=1)
        # x: [batch_size, attention_dim + reps_dim]
        x = self.linear_output(x)

        return x



class PostEncoder:

    def __init__(self, load_from_dir = False,  output_dir = None, batch_size = None, lr = None, num_epochs = None,\
        padding_idx = None):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model        
        '''

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.padding_idx = padding_idx
        

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
                # logger.info(f"Steps: {steps}")
                optimizer.zero_grad()
                input_values = batch["input_values"]
                labels = batch["labels"]
                assert self.model is not None and next(self.model.parameters()).is_cuda, "Model is not on GPU!"
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
                    _, _, _, accuracy = self.evaluate(dev_dataset)
                    wandb.log({"loss": loss.item()})
                    wandb.log({"accuracy": accuracy, "epoch": epoch})
                    logger.info(f"Epoch: {epoch}, Steps: {steps}, Accuracy: {accuracy}")


            if not evaluate_steps:
                # Log per epoch if evaluate_steps is not provided
                wandb.log({"loss": loss.item()})
                # Evaluate
                _, _, _, accuracy = self.evaluate(dev_dataset)
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
        for batch in test_loader:
            input_values = batch["input_values"]
            with torch.no_grad():
                outputs = self.model(input_values)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_labels.append(batch["labels"])
                all_accents.extend(batch["accents"])
        return torch.cat(all_preds), torch.cat(all_labels), all_accents


    def evaluate(self, test_dataset, collate_fn):
        preds, labels, accents = self.predict(test_dataset)

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
        return preds, labels, accents, correct/total
    
    def save(self, model_filename):
        torch.save(self.model, os.path.join(self.output_dir, model_filename))


class RepsPhoneseqsLinearClassifiereonAttentionLayers(PostEncoder):

    def __init__(self, vocab_size, num_classes, hidden_size, num_attention_layers, attention_dim, load_from_dir = False,  output_dir = None, batch_size = None, lr = None, num_epochs = None, \
        reps_dim = None, padding_idx = 0):
        '''If load_from_dir is True, load the model from output_dir. Else, initialize a new model'''
        if load_from_dir:
            model = torch.load(os.path.join(output_dir, f"reps-phoneseqs_attentions-{num_attention_layers}_classifier.pth"))
        else:
            model = RepsPhoneseqsAttentionsLinearModel(vocab_size=vocab_size, num_classes=num_classes, \
                                          hidden_size=hidden_size, \
                                            num_attention_layers=num_attention_layers, \
                                            attention_dim=attention_dim, \
                                            reps_dim=reps_dim, padding_idx=padding_idx)
        
        self.model = model.cuda()
        super().__init__(load_from_dir, output_dir, batch_size, lr, num_epochs, padding_idx)
        


    def train(self, train_dataset, dev_dataset, evaluate_steps = None):

        wandb.init(project=f"train_lid_on_reps_phoneseqs_attentions", config={
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.lr,
            "num_attention_layers": self.model.num_attention_layers,
            "attention_dim": self.model.attention_dim,
            "hidden_size": self.model.hidden_size,
            "reps_dim": self.model.reps_dim,
            "output_dir": self.output_dir,
        })
        super().train(train_dataset, dev_dataset, collate_fn, evaluate_steps)


    def predict(self, test_dataset):
        return super().predict(test_dataset, collate_fn)


    def evaluate(self, test_dataset):
        return super().evaluate(test_dataset, collate_fn)
    
    def save(self):
        super().save(f"reps-phoneseqs_attentions-{self.model.num_attention_layers}_classifier.pth")



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Directory containing transcribed audio files")
    
    parser.add_argument("--transcriber_model", type=str, required=True, help="Model used to transcribe the audio files")
    parser.add_argument("--encoder_model", type=str, required=True, help="Model used to encode the audio files for acoustic representations")
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


def map_tokenize_phoneme_labels(batch, phoneme2idx, lang2idx):
    '''Tokenize the phoneme labels and convert labels to integers
    batch: {
    'lang': str, 'accent': str, \
    'audio_file': str, \
    'reps': torch.tensor, \
    'phone_sequence': list}

    Add 
    'sequence': list, 
    'label': int
    '''
    phoneme_sequences = batch["phone_sequence"]
    coded_phoneme_sequences = []
    for phoneme_sequence in phoneme_sequences:
        ###### THIS IS THE SPLIT-PS VERSION #######
        phoneme_sequence = phoneme_sequence.split()
        #----------------
        coded_phoneme_sequence = [phoneme2idx[phoneme] for phoneme in phoneme_sequence \
            if phoneme in phoneme2idx]
        coded_phoneme_sequences.append(coded_phoneme_sequence)
    batch["sequence"] = coded_phoneme_sequences

    # HACK to handle the case where eval dataset codes don't have training dataset codes
    ## Since we're only evaluating on English, this only applies to English
    ## For training languages, this will not be an issue
    eng_eqv = [lang for lang in lang2idx if "en" in lang][0] # There should be only one
    for i, lang in enumerate(batch["lang"]):
        if lang not in lang2idx and "en" in lang:
            # Replace with the equivalent
            batch["lang"][i] = eng_eqv
    batch["label"] = [lang2idx[lang] for lang in batch["lang"]]

    return batch


def collate_fn(batch):
    '''Pad the phoneme sequences to the maximum length.
    The inputs are a tuple of (padded_sequences, reps) where padded_sequences is a padded tensor of phone sequences and reps is a tensor of acoustic representations
    '''

    # Preparing padded phone sequences
    model_max_len = 256
    padding_idx = 0
    sequences = [torch.tensor(item["sequence"]) for item in batch]
    truncated_sequences = [seq[:model_max_len] for seq in sequences]
    ## Pad the sequences with padding_idx=0
    padded_sequences = torch.nn.utils.rnn.pad_sequence(truncated_sequences, batch_first=True, padding_value=padding_idx)

    # Preparing acoustic representations
    reps = [torch.tensor(item["reps"]) for item in batch]
    reps = torch.stack(reps)

    # Input is a tuple of (padded_sequences, reps)
    input_values = (padded_sequences.cuda(), reps.cuda())
    
    # Prepare the labels
    labels = [item["label"] for item in batch]
    labels = torch.tensor(labels).cuda()

    # Prepare the accents
    accents = [item["accent"] for item in batch]    

    # Now we have a 256-dim tensor for reps, and a 256-length phoneme sequence tensor
    ## Note that each phone needs to be embedded before use
    return {"input_values": input_values,
            "labels": labels, \
            "accents": accents}



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



def load_reps_and_phoneseq_dataset(transcriber_model, encoder_model, dataset_name, \
                                   langs, per_lang = None, batch_size = 1, output_dir = None, \
                                    log_file = None):

    all_lang_datasets = []
    for lang in langs:
        logger.info(f"Processing lang: {lang}")
        lang_dataset = load_or_compute_encode_and_transcribe_dataset(transcriber_model, encoder_model, dataset_name, per_lang=per_lang, lang=lang, \
            batch_size=batch_size, output_dir=output_dir, log_file=None)
        if lang_dataset is not None:
            all_lang_datasets.append(lang_dataset)

    lid_dataset = concatenate_datasets(all_lang_datasets)

    return lid_dataset

def main():

    global logger

    args = parse_args()
    dataset_name = args.dataset_name

    transcriber_model = args.transcriber_model
    encoder_model = args.encoder_model
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

    # Tokenize the phonetic sequences and labels
    phoneme2idx_file = "/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/wav2vec2phoneme_map/wav2vec2phoneme_map.json"
    with open(phoneme2idx_file, "r") as f:
        phoneme2idx = json.load(f)
    idx2phoneme = {idx: phoneme for phoneme, idx in phoneme2idx.items()}

    if not only_eval:
        # Load dataset
        lid_dataset = load_reps_and_phoneseq_dataset(transcriber_model=transcriber_model, encoder_model=encoder_model, \
            dataset_name=dataset_name, langs=langs, per_lang=per_lang, batch_size=batch_size, output_dir=save_dataset_dir, \
                log_file=None)

        # lid_dataset = lid_dataset.select(range(5000)) # For debugging

        lid_dataset = lid_dataset.filter(lambda x: len(x["phone_sequence"]) > 0)
        # Get the mapping from language to index
        ## Convert the phonetic sequences to integers, and map the languages to integers
        lid_dataset = lid_dataset.map(map_tokenize_phoneme_labels, fn_kwargs={"phoneme2idx": phoneme2idx, "lang2idx": lang2idx}, \
            batched=True, \
            batch_size = 1000)
        
        # Split the dataset into training and validation
        train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset, dev_size = 1000, test_size = 10000)
        ### CHANGE THIS
        # train_dataset, dev_dataset, test_dataset = get_dataset_splits(lid_dataset, dev_size = 500, test_size = 500)
    

    assert lid_model_type in ["attentions-linear"], "Invalid LID model type"
    if only_eval:
        assert load_trained_from_dir, "If only evaluating, you must load the model from a directory"

    logger.info(f"Type of LID model: {lid_model_type}")
    
    if lid_model_type == "attentions-linear":

        vocab_size = len(phoneme2idx)
        hidden_size = 256
        reps_dim = 256 # train_dataset[0]["reps"].shape[0]
        num_classes = len(idx2lang)
        logger.info(f"Num classes: {num_classes}")
        lid_model = RepsPhoneseqsLinearClassifiereonAttentionLayers(vocab_size=vocab_size, num_classes=num_classes, \
                                hidden_size=hidden_size, num_attention_layers=num_attention_layers, \
                                    attention_dim=128, \
                                    load_from_dir = load_trained_from_dir, output_dir = output_dir, \
                                    batch_size = batch_size, lr = lr, num_epochs = num_epochs, reps_dim=reps_dim, padding_idx=0)

    
    
    # Train the model
    if not only_eval:
        logger.info(f"Training model...")
        lid_model.train(train_dataset, dev_dataset, evaluate_steps = evaluate_steps)
        lid_model.save()

        # Evaluate the model
        logger.info(f"Evaluating model on test split of train dataset...")
        preds, labels, accents, accuracy = lid_model.evaluate(test_dataset)
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
            pkl.dump({"preds": preds, "labels": labels, "accents": accents}, f)

        with open(os.path.join(output_dir, f"eval_accuracy.json"), "w") as f:
            json.dump({f"{dataset_name}_test_accuracy": accuracy}, f)

    # Evaluate the model on eval dataset if provided

    if eval_dataset_name:
        logger.info(f"Evaluating model on eval dataset...")
        eval_dataset = load_reps_and_phoneseq_dataset(transcriber_model=transcriber_model, encoder_model=encoder_model, \
            dataset_name=eval_dataset_name, langs=langs, per_lang=per_lang, batch_size=batch_size, output_dir=save_eval_dataset_dir, \
                log_file=None)

        # Get the mapping from language to index
        ## Convert the phonetic sequences to integers, and map the languages to integers
        eval_dataset = eval_dataset.map(map_tokenize_phoneme_labels, fn_kwargs={"phoneme2idx": phoneme2idx, "lang2idx": lang2idx}, \
            batched=True, \
            batch_size = 1000)

        logger.info(f"Evaluating model on eval dataset...")
        preds, labels, accents, accuracy = lid_model.evaluate(eval_dataset)
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
            pkl.dump({"preds": preds, "labels": labels, "accents": accents}, f)
        
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
    