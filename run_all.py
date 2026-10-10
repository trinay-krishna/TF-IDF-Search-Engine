import csv
import os
from functools import partial

from embeddings import embedding_search
from eval import compute_metrics
from hybrid import hybrid_search
from plot import RESULTS_DIR
from search import TF_IDF, TF_IDF_NO_STOPWORDS
from sklearn_search import sklearn_search

K_VALUES = (5, 10, 20)
ALPHAS = tuple(round(i / 10, 1) for i in range(11))   # 0.0 (embeddings only) .. 1.0 (TF-IDF only)
OUTPUT_PATH = os.path.join(RESULTS_DIR, "all_runs.csv")


def build_runs():
    """Returns (method name, alpha or None, search function) for every run."""
    runs = [
        ("tfidf", None, TF_IDF_NO_STOPWORDS),
        ("tfidf_stopwords", None, TF_IDF),
        ("tfidf_sklearn", None, sklearn_search),
        ("embeddings", None, embedding_search),
    ]
    # The hybrid always uses the stopword-removed hand-written TF-IDF.
    for name, tfidf_search in (("hybrid_tfidf", TF_IDF), ("hybrid_sklearn", sklearn_search)):
        for alpha in ALPHAS:
            runs.append((name, alpha, partial(hybrid_search, alpha=alpha, tfidf_search=tfidf_search)))
    return runs


def run_all():
    header = ["method", "alpha"] + [f"p{k}" for k in K_VALUES] + [f"r{k}" for k in K_VALUES]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for name, alpha, search in build_runs():
            precisions, recalls, _, _ = compute_metrics(search, K_VALUES)
            row = [name, "" if alpha is None else alpha]
            row += [f"{value:.4f}" for value in precisions + recalls]
            writer.writerow(row)
            print(",".join(map(str, row)))

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    run_all()
