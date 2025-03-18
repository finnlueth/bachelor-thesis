import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_scope_benchmarks(file_paths: dict[str, str]):
    """
    Plots SCOPe benchmark sensitivity up to the first false positive for multiple .rocx files.
    
    Parameters:
        file_paths (dict): Dictionary mapping display names to .rocx file paths.
    """
    
    plt.rcParams['figure.dpi'] = 300
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["Family", "Superfamily", "Fold"]
    y_labels = ["FAM", "SFAM", "FOLD"]

    colors = ['black', 'blue', 'red', 'orange', 'green', 'purple', 'brown', 'gray', 'olive', 'cyan', 'magenta', 'lime', 'teal', 'navy', 'maroon', 'gold', 'silver', 'indigo', 'turquoise', 'coral', 'khaki', 'lime', 'teal', 'navy', 'maroon', 'gold', 'silver', 'indigo', 'turquoise', 'coral', 'khaki']

    for idx, (name, path) in enumerate(file_paths.items()):
        df = pd.read_csv(path, sep="\t")

        fraction_queries = np.linspace(0, 1, len(df))

        for i, ylabel in enumerate(y_labels):
            df_sorted = df.sort_values(by=ylabel, ascending=False)
            axes[i].plot(
                fraction_queries, df_sorted[ylabel],
                label=name,
                color=colors[idx % len(colors)],
                linewidth=0.8,
                linestyle='-'
            )
            axes[i].set_xlabel("Fraction of Queries")
            axes[i].set_ylabel("Sensitivity up to the 1st FP")
            axes[i].set_title(f"SCOPe Benchmark - {titles[i]}")
            axes[i].grid()

    axes[-1].legend(title="Method", loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()