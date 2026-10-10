import csv
import os

from plot import RESULTS_DIR, plot_alpha_sweep, plot_overlay

K_VALUES = [5, 10, 20]
CSV_PATH = os.path.join(RESULTS_DIR, "all_runs.csv")

# CSV method name -> label shown in the README and the charts.
MAIN_METHODS = {
    "tfidf": "TF-IDF (from scratch)",
    "tfidf_stopwords": "+ stopwords",
    "tfidf_sklearn": "TF-IDF (sklearn)",
    "embeddings": "Sentence embeddings",
}
HYBRID_METHODS = {
    "hybrid_tfidf": "Hybrid, TF-IDF (from scratch + stopwords)",
    "hybrid_sklearn": "Hybrid, TF-IDF (sklearn)",
}


def make_figures(path=CSV_PATH):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    precision = {}
    recall = {}
    for row in rows:
        if row["method"] in MAIN_METHODS:
            label = MAIN_METHODS[row["method"]]
            precision[label] = [float(row[f"p{k}"]) for k in K_VALUES]
            recall[label] = [float(row[f"r{k}"]) for k in K_VALUES]
    plot_overlay(K_VALUES, precision, "Precision", os.path.join(RESULTS_DIR, "overview_precision.png"))
    plot_overlay(K_VALUES, recall, "Recall", os.path.join(RESULTS_DIR, "overview_recall.png"))

    for column, ylabel, filename in (("p10", "P@10", "hybrid_alpha_p10.png"),
                                     ("r10", "R@10", "hybrid_alpha_r10.png")):
        sweep = {label: [] for label in HYBRID_METHODS.values()}
        for row in rows:
            if row["method"] in HYBRID_METHODS:
                sweep[HYBRID_METHODS[row["method"]]].append((float(row["alpha"]), float(row[column])))
        for points in sweep.values():
            points.sort()
        alphas = [alpha for alpha, _ in next(iter(sweep.values()))]
        series = {label: [value for _, value in points] for label, points in sweep.items()}
        plot_alpha_sweep(alphas, series, os.path.join(RESULTS_DIR, filename), ylabel=ylabel)
    print("Saved overview_precision.png, overview_recall.png, hybrid_alpha_p10.png, hybrid_alpha_r10.png")


if __name__ == "__main__":
    make_figures()
