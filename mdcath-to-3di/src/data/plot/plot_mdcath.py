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
    """Compute the normalized Hamming distance between two sequences."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)


def plot_sequence_order_pca_matplotlib(sequences_dict, numbering=False, arrows=False, average_arrows=False):
    """
    Perform PCA on sequences from a dictionary, visualize them in 2D, and color by sequence order.

    Parameters:
    sequences_dict (dict): A dictionary where keys are names and values are lists of sequences.

    Returns:
    None: Displays a scatter plot.
    """

    for name, sequences in sequences_dict.items():
        sequence_array = np.array(sequences).reshape(-1, 1)
        pairwise_distances = pdist(sequence_array, metric=lambda u, v: hamming_distance(u[0], v[0]))
        distance_matrix = squareform(pairwise_distances)

        pca = PCA(n_components=2)
        sequence_coords = pca.fit_transform(distance_matrix)

        sequence_order_colors = np.linspace(0, 1, len(sequences))

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            sequence_coords[:, 0], sequence_coords[:, 1], c=sequence_order_colors, cmap=plt.cm.cool, alpha=0.7
        )

        count_numbers = 20
        count_arrows = 20
        if numbering:
            n = len(sequence_coords) // count_numbers  # Show ~20 numbers
            n = max(1, n)  # Ensure n is at least 1
            for i, (x, y) in enumerate(sequence_coords):
                if i % n == 0:  # Only show every nth number
                    plt.annotate(
                        str(i // count_numbers), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=8, alpha=0.7
                    )

        if arrows:
            for i, (x, y) in enumerate(sequence_coords[:-1]):
                next_x, next_y = sequence_coords[i + 1]
                plt.arrow(
                    x,
                    y,
                    next_x - x,
                    next_y - y,
                    head_width=0.01,
                    head_length=0.05,
                    fc="red",
                    ec="gray",
                    alpha=0.2,
                    length_includes_head=True,
                )

        if average_arrows:
            n = len(sequences) // count_arrows  # Use ~20 arrows total
            n = max(1, n)  # Ensure n is at least 1
            for i in range(0, len(sequence_coords) - n, n):
                current_pos = sequence_coords[i]
                # Calculate average position of next n points
                next_positions = sequence_coords[i + 1 : i + n + 1]
                if len(next_positions) > 0:  # Only draw if there are next positions
                    avg_next_pos = np.mean(next_positions, axis=0)
                    # Get color from sequence order
                    color = plt.cm.cool(i / len(sequence_coords))
                    plt.arrow(
                        current_pos[0],
                        current_pos[1],
                        avg_next_pos[0] - current_pos[0],
                        avg_next_pos[1] - current_pos[1],
                        head_width=0.02,
                        head_length=0.05,
                        fc=color,
                        ec=color,
                        alpha=0.6,
                        length_includes_head=True,
                    )

        cbar = plt.colorbar(scatter, label="Sequence Order")
        cbar.set_ticks(np.linspace(0, 1, num=len(sequences) // 100 + 1))
        cbar.set_ticklabels(np.arange(0, len(sequences) + 1, 100))

        plt.title(f"2D Scatter Plot of {name} Tokenized Trajectory Sequences Colored by Sequence Order (Blue to Purple)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        return plt


def plot_sequence_order_pca_plotly(sequences_dict, numbering=False, arrows=False, average_arrows=False):
    """
    Perform PCA on sequences from a dictionary, visualize them in 2D using Plotly, and color by sequence order.

    Parameters:
    sequences_dict (dict): A dictionary where keys are names and values are lists of sequences.

    Returns:
    plotly.graph_objects.Figure: Interactive Plotly figure
    """
    colorscale = "cool"
    for name, sequences in sequences_dict.items():
        sequence_array = np.array(sequences).reshape(-1, 1)
        pairwise_distances = pdist(sequence_array, metric=lambda u, v: hamming_distance(u[0], v[0]))
        distance_matrix = squareform(pairwise_distances)

        pca = PCA(n_components=2)
        sequence_coords = pca.fit_transform(distance_matrix)

        # Create base scatter plot
        fig = go.Figure()

        # Add main scatter points
        fig.add_trace(
            go.Scatter(
                x=sequence_coords[:, 0],
                y=sequence_coords[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=np.arange(len(sequences)),
                    colorscale=[[i/255, f"rgb{tuple(int(x * 255) for x in plt.cm.cool(i/255)[:3])}"] for i in range(256)],
                    showscale=True,
                    colorbar=dict(
                        title="Sequence Order",
                        tickmode="array",
                        ticktext=list(range(0, len(sequences) + 1, 100)),
                        tickvals=list(range(0, len(sequences) + 1, 100)),
                    ),
                ),
                name="Sequences",
            )
        )

        count_numbers = len(sequences)
        # count_arrows = len(sequences) // 20
        count_arrows = len(sequences)

        # Add numbering if requested
        if numbering:
            n = len(sequences) // count_numbers
            n = max(1, n)
            for i in range(0, len(sequence_coords), n):
                fig.add_trace(
                    go.Scatter(
                        x=[sequence_coords[i, 0]],
                        y=[sequence_coords[i, 1]],
                        mode="text",
                        text=[str(i)],
                        textposition="top center",
                        showlegend=False,
                        textfont=dict(size=10)
                    )
                )

        # Add arrows if requested
        if arrows:
            n = len(sequences) // count_arrows
            n = max(1, n)
            for i in range(0, len(sequence_coords) - 1, n):
                current_pos = sequence_coords[i]
                next_pos = sequence_coords[i + 1]
                color = f"rgb{tuple(int(x * 255) for x in plt.cm.cool(i / len(sequence_coords))[:3])}"
                
                # Draw the line segment
                fig.add_trace(
                    go.Scatter(
                        x=[current_pos[0], next_pos[0]],
                        y=[current_pos[1], next_pos[1]],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=2
                        ),
                        opacity=0.6,
                        showlegend=False
                    )
                )
                
                # Add arrow at the end point
                fig.add_trace(
                    go.Scatter(
                        x=[next_pos[0]],
                        y=[next_pos[1]],
                        mode="markers",
                        marker=dict(
                            symbol="arrow",
                            size=10,
                            angle=np.degrees(np.arctan2(
                                next_pos[1] - current_pos[1],
                                next_pos[0] - current_pos[0]
                            )),
                            color=color
                        ),
                        opacity=0.6,
                        showlegend=False
                    )
                )

        # Add average arrows if requested
        if average_arrows:
            n = len(sequences) // count_arrows
            n = max(1, n)
            for i in range(0, len(sequence_coords) - n, n):
                current_pos = sequence_coords[i]
                next_positions = sequence_coords[i + 1 : i + n + 1]
                if len(next_positions) > 0:
                    avg_next_pos = np.mean(next_positions, axis=0)
                    color = f"rgb{tuple(int(x * 255) for x in plt.cm.cool(i / len(sequence_coords))[:3])}"
                    fig.add_trace(
                        go.Scatter(
                            x=[current_pos[0], avg_next_pos[0]],
                            y=[current_pos[1], avg_next_pos[1]],
                            mode="lines+markers",
                            line=dict(color=color, width=2),
                            opacity=0.6,
                            showlegend=False,
                        )
                    )

        # Update layout
        fig.update_layout(
            title=f"2D Scatter Plot of {name} Tokenized Trajectory Sequences",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            hovermode="closest",
            showlegend=False,
        )

        return fig


def combine_plots(plots, max_cols):
    """Combines multiple plotly plots into a single figure with subplots

    Args:
        plots (list): List of plotly figure objects to combine
        max_cols (int): Maximum number of columns in subplot grid
        max_rows (int): Maximum number of rows in subplot grid

    Returns:
        plotly.graph_objs.Figure: Combined figure with subplots
    """
    # Create subplot figure with max_rows x max_cols grid
    fig = make_subplots(
        rows=-(-len(plots) // max_cols),
        cols=max_cols,
        subplot_titles=[plot.layout.title.text for plot in plots],
        horizontal_spacing=0.05,  # Decrease horizontal padding
        vertical_spacing=0.08,  # Decrease vertical padding
    )

    # Decrease font size of subplot titles
    fig.update_annotations(font_size=11)

    # Add each plot as a subplot
    for i, plot in enumerate(plots):
        row = (i // max_cols) + 1
        col = (i % max_cols) + 1

        # Extract traces from individual plots and add to subplots
        for trace in plot.data:
            fig.add_trace(trace, row=row, col=col)

        # Copy layout properties for each subplot
        fig.update_xaxes(title_text=plot.layout.xaxis.title.text, row=row, col=col)
        fig.update_yaxes(title_text=plot.layout.yaxis.title.text, row=row, col=col)

    # Update overall layout
    fig.update_layout(height=900, width=1600, showlegend=False)

    return fig
