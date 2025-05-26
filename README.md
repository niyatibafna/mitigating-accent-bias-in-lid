# Mitigating Accent Bias in Speech Language Identification

This is the code for our paper **LID Models are Actually Accent Classifiers: Implications and Solutions for LID on Accented Speech** (Accepted at Interspeech 2025). 
We show that accent-language confusion is a major cause of error on accented speech in speech LID systems, and explore a number of ways of integrating sequence-level information about the speech into an ECAPA-TDNN-based LID classifier without relying on monolingual ASR systems.
This results in much higher performance on accented speech. 

## Preprocessing

### Dataset formats

See `utils/dataloading/dataset_loader.py` for a wrapper script for loading audio files and language labels from the datasets we used into a Huggingface dataset with a unified set of language labels. Datasets: CommonVoice, FLEURS, NISTLRE, VoxLingua-107, and the Edinburgh International Accents of English Corpus.

### Phoneme transcription of datasets

See `lid_with_phoneseqs/transcribe_data.py` for transcription of data into phoneme sequences. This is just inference with [facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft).

### Extracting clustered SSL units

We also prepare sequence representation of the speech that consists of clustered SSL representations, extracted from [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base).
See `lid_with_ssl_units/extracting_units_from_training_data.py` for extracting and clustering SSL representations.

### Extracting ECAPA-TDNN last layer representations 

See `lid_with_acoustics-phoneseq/encode_and_transcribe_dataset.py` and `lid_with_reps-sslunits/encode_transcribe_and_get_ssl_units.py`. This is just inference with the ECAPA-TDNN LID model: [speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa).

### Extracting output label distribution of ECAPA-TDNN 

We also extract output probability distributions over language labels for each sample from the ECAPA-TDNN model here: `lid_with_acoustics-phoneseq/et_dists_and_transcribe_dataset.py`. 

## Our Approaches

### `ET+phoneseqs-train` (Best performing)

We use ECAPA-TDNN last layer representations and phoneme sequences to train our best-performing LID model as described in the paper. See here: `lid_with_acoustics-phoneseq/train_lid_with_acoustics_phoneseq.py`.

### `phoneseqs`

We train an LID classifier on top of phoneme sequences. See `lid_with_phoneseqs/train_lid_on_phoneseqs.py`.

### `duseqs`

We train LID on top of discrete unit sequences obtained from clustering SSL representations as prepared above. See `lid_with_ssl_units/train_lid.py`.

### `ET+phoneseqs`

We perform a system combination of ECAPA-TDNN output probability distribution with output probability distribution of the `phoneseqs` module. See here: `lid_with_acoustics-phoneseq/system_combination_reps_phoneseqs.py`.

### `ET+duseqs-train` and `ET+duseqsembed-train`

We use ECAPA-TDNN last layer representations and discrete unit sequences to train an LID model. We either reuse the learnt centroid representations obtained from KMeans to embed the discrete units in the `duseqs` module, or train from-scratch embeddings, as described in the paper. See `lid_with_ssl_units/train_lid.py`.


If you use our code, please cite:
```
TODO
```










