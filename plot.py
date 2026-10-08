import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _plot_metric(k_values, values, label, output_path):
    plt.figure()
    plt.plot(k_values, values, marker="o")
    plt.xticks(k_values)
    plt.xlabel("k")
    plt.ylabel(f"{label}@k")
    plt.title(f"{label} vs k")
    plt.grid(True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_results(k_values, precisions, recalls, output_dir=RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    _plot_metric(k_values, precisions, "Precision", os.path.join(output_dir, "precision.png"))
    _plot_metric(k_values, recalls, "Recall", os.path.join(output_dir, "recall.png"))
