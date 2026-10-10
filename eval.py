from parsing import parse_file
from plot import plot_results
from search import get_search_function


def load_relevance(path="CISI.REL"):
    """Maps each query id to the set of document ids judged relevant to it."""
    relevance = {}
    with open(file=path, mode="r", encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if len(fields) < 2:
                continue
            relevance.setdefault(int(fields[0]), set()).add(int(fields[1]))
    return relevance


def precision_recall(retrieved_ids, relevant_ids):
    hits = len(set(retrieved_ids) & relevant_ids)
    precision = hits / len(retrieved_ids) if retrieved_ids else 0
    recall = hits / len(relevant_ids)
    return precision, recall


def compute_metrics(search, k_values=(5, 10, 20)):
    """Runs `search` on every judged query. Returns (precisions, recalls, evaluated, skipped),
    with precisions/recalls averaged over queries, in k_values order."""
    queries = parse_file("CISI.QRY", "query")
    relevance = load_relevance()

    totals = {k: [0.0, 0.0] for k in k_values}
    evaluated = 0

    for query in queries:
        relevant_ids = relevance.get(query.id)
        if not relevant_ids:
            continue

        ranked = [doc_id for doc_id, _ in search(query.content)]
        evaluated += 1

        for k in k_values:
            precision, recall = precision_recall(ranked[:k], relevant_ids)
            totals[k][0] += precision
            totals[k][1] += recall

    precisions = [totals[k][0] / evaluated for k in k_values]
    recalls = [totals[k][1] / evaluated for k in k_values]
    return precisions, recalls, evaluated, len(queries) - evaluated


def evaluate(k_values=(5, 10, 20), method="tfidf"):
    """method: "tfidf", "sklearn", "embeddings" or "hybrid". Plots are saved as <method>_precision.png / <method>_recall.png."""
    precisions, recalls, evaluated, skipped = compute_metrics(get_search_function(method), k_values)

    print(f"Method: {method}")
    print(f"Evaluated {evaluated} queries, skipped {skipped} without judgments.")

    print(f"{'k':>4}  {'precision':>9}  {'recall':>7}")
    for k, precision, recall in zip(k_values, precisions, recalls):
        print(f"{k:>4}  {precision:>9.4f}  {recall:>7.4f}")

    plot_results(list(k_values), precisions, recalls, prefix=method)

if __name__ == "__main__":
    evaluate(method="hybrid")  # "tfidf", "sklearn", "embeddings" or "hybrid"
