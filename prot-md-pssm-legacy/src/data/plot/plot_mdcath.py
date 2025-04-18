import logomaker
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def hamming_distance(seq1, seq2):
    """Calculate Hamming distance between two sequences"""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def plot_sequence_order_pca_plotly(sequences_dict, numbering=False, arrows=False, average_arrows=False):
    """
    Perform PCA on sequences from a dictionary and visualize them in 2D using Plotly.
    
    Args:
        sequences_dict (dict): Dictionary where keys are names and values are lists of sequences
        numbering (bool): Whether to show sequence numbers
        arrows (bool): Whether to show arrows between consecutive points
        average_arrows (bool): Whether to show arrows between points and average of next n points
        
    Returns:
        plotly.graph_objs.Figure: Interactive Plotly figure
    """
    
    for name, sequences in sequences_dict.items():
        sequence_array = np.array(sequences).reshape(-1, 1)
        pairwise_distances = pdist(sequence_array, metric=lambda u, v: hamming_distance(u[0], v[0]))
        distance_matrix = squareform(pairwise_distances)
        
        pca = PCA(n_components=2)
        sequence_coords = pca.fit_transform(distance_matrix)
        
        sequence_order_colors = np.linspace(0, 1, len(sequences))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=sequence_coords[:, 0],
                y=sequence_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=sequence_order_colors,
                    colorscale='Viridis',
                    showscale=True,
                    opacity=0.7
                ),
                text=[f"Sequence {i}" for i in range(len(sequences))],
                hoverinfo='text'
            )
        )

        # Add numbering if requested
        count_numbers = 20
        count_arrows = 20
        if numbering:
            n = len(sequence_coords) // count_numbers
            n = max(1, n)
            for i in range(0, len(sequence_coords), n):
                fig.add_trace(
                    go.Scatter(
                        x=[sequence_coords[i, 0]],
                        y=[sequence_coords[i, 1]],
                        mode='text',
                        text=[str(i)],
                        textposition="top center",
                        showlegend=False,
                        textfont=dict(size=10)
                    )
                )

        # Update layout
        fig.update_layout(
            title=f"2D Scatter Plot of {name} Tokenized Trajectory Sequences",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            hovermode='closest',
            showlegend=False
        )

        return fig
