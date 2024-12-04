import logomaker
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


def create_combined_sequence_logo(sequence_dict):
    """
    Creates sequence logos for each trajectory in the input dictionary.

    Args:
        sequence_dict (dict): Dictionary where keys are trajectory names and values are lists of sequences

    Returns:
        dict: Dictionary of matplotlib figures, with trajectory names as keys
    """
    all_sequences = []
    for sequences in tqdm(sequence_dict.values()):
        all_sequences.extend(sequences)

    counts_matrix = []
    for pos in range(len(all_sequences[0])):
        pos_counts = {}
        for seq in all_sequences:
            char = seq[pos]
            pos_counts[char] = pos_counts.get(char, 0) + 1
        counts_matrix.append(pos_counts)

    df = pd.DataFrame(counts_matrix)
    df = df.fillna(0)

    fig, ax = plt.subplots(figsize=(10, 3))
    logo = logomaker.Logo(df, ax=ax)

    ax.set_title("Combined Sequence Logo")
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")

    return fig


def create_single_sequence_logo(sequences):
    """
    Creates a PSSM-based sequence logo from a list of sequences.

    Args:
        sequences (list): List of sequences to create logo from

    Returns:
        matplotlib.figure.Figure: Figure containing the sequence logo
    """
    counts_matrix = []
    for pos in range(len(sequences[0])):
        pos_counts = {}
        total = len(sequences)
        for seq in sequences:
            char = seq[pos]
            pos_counts[char] = pos_counts.get(char, 0) + 1

        for char in pos_counts:
            pos_counts[char] = pos_counts[char] / total

        counts_matrix.append(pos_counts)

    df = pd.DataFrame(counts_matrix)
    df = df.fillna(0)

    fig, ax = plt.subplots(figsize=(10, 3))
    logo = logomaker.Logo(df, ax=ax, weight_type="information", center_values=False)

    ax.set_title("PSSM Sequence Logo")
    ax.set_xlabel("Position")
    ax.set_ylabel("Information Content")

    return fig


def plot_sequence_order_pca(sequences_dict):
    """
    Perform PCA on sequences from a dictionary, visualize them in 2D, and color by sequence order.
    
    Parameters:
    sequences_dict (dict): A dictionary where keys are names and values are lists of sequences.

    Returns:
    None: Displays a scatter plot.
    """
    def hamming_distance(seq1, seq2):
        """Compute the normalized Hamming distance between two sequences."""
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)

    for name, sequences in sequences_dict.items():
        sequence_array = np.array(sequences).reshape(-1, 1)
        pairwise_distances = pdist(sequence_array, metric=lambda u, v: hamming_distance(u[0], v[0]))
        distance_matrix = squareform(pairwise_distances)

        pca = PCA(n_components=2)
        sequence_coords = pca.fit_transform(distance_matrix)

        sequence_order_colors = np.linspace(0, 1, len(sequences))

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(sequence_coords[:, 0], sequence_coords[:, 1], c=sequence_order_colors, cmap='bwr', alpha=0.7)

        cbar = plt.colorbar(scatter, label='Sequence Order')
        cbar.set_ticks(np.linspace(0, 1, num=len(sequences) // 100 + 1))
        cbar.set_ticklabels(np.arange(0, len(sequences) + 1, 100))

        plt.title(f'2D Scatter Plot of {name} Sequences Colored by Sequence Order (Blue to Red)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        return plt
