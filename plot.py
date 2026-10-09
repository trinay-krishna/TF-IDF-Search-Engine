import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# Default y-axis ranges so plots from different runs can be compared by eye.
# They are widened automatically (see _fit_ylim) if the data falls outside.
PRECISION_YLIM = (0.15, 0.30)
RECALL_YLIM = (0.0, 0.15)


def _fit_ylim(values, default, step=0.05):
    """Returns `default`, widened on whichever side the data would not fit,
    snapped outward to a multiple of `step` with at least one step of headroom."""
    lo = min(default[0], (math.ceil(round(min(values) / step, 9)) - 1) * step)
    hi = max(default[1], (math.floor(round(max(values) / step, 9)) + 1) * step)
    return round(max(lo, 0), 10), round(hi, 10)


def _plot_metric(k_values, values, label, output_path, default_ylim):
    plt.figure()
    plt.plot(k_values, values, marker="o")
    plt.ylim(*_fit_ylim(values, default_ylim))
    plt.xticks(k_values)
    plt.xlabel("k")
    plt.ylabel(f"{label}@k")
    plt.title(f"{label} vs k")
    plt.grid(True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_results(k_values, precisions, recalls, output_dir=RESULTS_DIR, prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    _plot_metric(k_values, precisions, "Precision", os.path.join(output_dir, f"{stem}precision.png"), PRECISION_YLIM)
    _plot_metric(k_values, recalls, "Recall", os.path.join(output_dir, f"{stem}recall.png"), RECALL_YLIM)
