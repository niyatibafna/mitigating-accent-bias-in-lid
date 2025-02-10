"""
Training:
This is for taking the segment embeddings and running kmeans on them to get centroids for phoneme SSL units.
We save the trained kmeans model to disk.
Inference:
This class is also for taking the segment embeddings and running a trained kmeans model on them to get the phoneme SSL units.
"""

import os
import numpy as np
from sklearn.cluster import KMeans
import faiss
from argparse import ArgumentParser
import torch
import pickle as pkl

class KMeansOnUnits:
    def __init__(self, n_clusters = 100, output_dir = None, dim = 768):
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        self.trained = False
        self.dim = dim
        self.init_kmeans_model()

    def init_kmeans_model(self):
        '''Load the trained kmeans model from disk'''
        if self.output_dir is not None and os.path.exists(os.path.join(self.output_dir, f"kmeans_model_centroids.npy")):
            with open(os.path.join(self.output_dir, f"kmeans_model_centroids.npy"), "rb") as f:
                print(f"Loading trained kmeans model from {self.output_dir}")
                centroids = np.load(f)
                n_centroids = centroids.shape[0]
                # We don't use self.n_clusters here because we want to be able to load a model with a different number of clusters
                self.kmeans = faiss.Kmeans(self.dim, n_centroids, niter=20, verbose=True, gpu=False, nredo=10)
                self.kmeans.index = faiss.IndexFlatL2(self.dim)  # Flat L2 index
                print(f"Number of centroids: {n_centroids}")
                # Check that n_clusters is the same as the number of centroids
                if n_centroids != self.n_clusters:
                    print(f"WARNING: Number of centroids in the trained model ({n_centroids}) is different from the number of clusters ({self.n_clusters})")
                self.kmeans.index.add(centroids)  # Add the centroids to the index
                self.trained = True
                self.dim = centroids.shape[1]

        else:
            print(f"Trained kmeans model not found in {self.output_dir}. Initializing a new model...")
            # self.kmeans = KMeans(n_clusters = self.n_clusters, random_state=0)
            self.kmeans = faiss.Kmeans(self.dim, self.n_clusters, niter=20, verbose=True, gpu=False, nredo=10)


    def train(self, all_segment_reps):
        '''Train kmeans on the full representations'''
        # all_segment_reps: List of tensors of shape (sequence_length, hidden_size)
        print(f"Training kmeans on segment representations...")
        # Flatten the segment representations and send to CPU
        reps = [segment_rep.cpu() for segment_reps in all_segment_reps for segment_rep in segment_reps]
        print(f"Number of segment representations: {len(reps)}")
        # print(f"Shape of segment representations: {np.array(reps).shape}")
        
        # Train kmeans
        # This is for sklearn kmeans
        # self.kmeans.fit(reps)
        # This is for faiss kmeans
        self.kmeans.train(reps)
        self.trained = True
    
    def save_model(self):
        # Save the model
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            print(f"Saving trained kmeans model to {self.output_dir}")
            with open(os.path.join(self.output_dir, f"kmeans_model_centroids.npy"), "wb") as f:
                np.save(f, self.kmeans.centroids)
    

    def predict(self, all_segment_reps):
        '''Predict phoneme units on the segment representations'''
        # all_segment_reps: List of tensors of shape (sequence_length, hidden_size)
        # We want to return the phoneme units for each segment: (sequence_length,)

        kmeans_reps = []
        for segment_reps in all_segment_reps:
            # kmeans_reps.append(self.kmeans.predict(segment_reps))
            # This is for faiss kmeans
            # Only move to CPU if tensor
            if not type(segment_reps) == np.ndarray:
                # Move to CPU and convert to numpy
                segment_reps = segment_reps.cpu().numpy()
            segment_reps = segment_reps.reshape(-1, self.dim)
            D, I = self.kmeans.index.search(segment_reps, 1)
            kmeans_reps.append(I.reshape(-1))
        
        return kmeans_reps
    
    def save_centroid_sequences(self, kmeans_reps, reps_output_dir):
        '''Save the phoneme units to disk'''
        # all_segment_reps: List of tensors of shape (sequence_length, hidden_size)
        # all_audio_files: List of audio file paths

        if reps_output_dir is not None:
            if not os.path.exists(reps_output_dir):
                os.makedirs(reps_output_dir)
            
            with open(os.path.join(reps_output_dir, f"all_segment_reps.pkl"), "wb") as f:
                pkl.dump(kmeans_reps, f)
        
        
def get_segment_reps(segment_rep_dir):
    '''Load the segment representations for all languages from disk'''
    all_segment_reps = []
    for lang in os.listdir(segment_rep_dir):
        lang_dir = os.path.join(segment_rep_dir, lang)
        with open(os.path.join(lang_dir, f"all_segment_reps.pkl"), "rb") as f:
            segment_reps = pkl.load(f)
            all_segment_reps.extend(segment_reps)
    return all_segment_reps
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters for kmeans")
    parser.add_argument("--segment_rep_dir", type=str, default=None, help="Directory containing the segment representations for all languages")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save the trained kmeans model and units")
    return parser.parse_args()

def main():
    # If we have the segment representations, we can run main() to train the kmeans model
    args = parse_args()

    kmeans_on_units = KMeansOnUnits(n_clusters = args.n_clusters, output_dir = args.output_dir)

    # Load the segment representations
    print(f"Loading segment representations from {args.output_dir}")
    all_segment_reps = get_segment_reps(args.segment_rep_dir)

    
    # Train kmeans and save the units
    print(f"Training kmeans on segment representations")
    kmeans_on_units.train(all_segment_reps)
    kmeans_on_units.save_model()
    print(f"Computing and saving representations")
    kmeans_reps = kmeans_on_units.predict(all_segment_reps)
    kmeans_on_units.save_centroid_sequences(kmeans_reps= kmeans_reps, reps_output_dir = args.output_dir)

if __name__ == "__main__":
    main()