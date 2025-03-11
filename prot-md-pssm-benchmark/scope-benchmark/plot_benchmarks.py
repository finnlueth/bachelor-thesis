import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_scope_benchmarks(file_paths: list[str]):
    """
    Plots SCOPe benchmark sensitivity up to the first false positive for multiple .rocx files.
    
    Parameters:
        file_paths (list): List of file paths to .rocx files.
    """
    
    plt.rcParams['figure.dpi'] = 300
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["Family", "Superfamily", "Fold"]
    y_labels = ["FAM", "SFAM", "FOLD"]

    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange']

    for idx, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path, sep="\t")

        fraction_queries = np.linspace(0, 1, len(df))

        for i, ylabel in enumerate(y_labels):
            df_sorted = df.sort_values(by=ylabel, ascending=False)
            axes[i].plot(
                fraction_queries, df_sorted[ylabel],
                label=file_path, 
                color=colors[idx % len(colors)], 
                linewidth=0.8,
                linestyle='-'
            )
            axes[i].set_xlabel("Fraction of Queries")
            axes[i].set_ylabel("Sensitivity up to the 1st FP")
            axes[i].set_title(f"SCOPe Benchmark - {titles[i]}")
            axes[i].grid()

    axes[-1].legend(title="File Path", loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()