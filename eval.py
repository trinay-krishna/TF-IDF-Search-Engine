from parsing import parse_file
from plot import plot_results
from search import TF_IDF


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


def evaluate(k_values=(5, 10, 20)):
    queries = parse_file("CISI.QRY", "query")
    relevance = load_relevance()

    totals = {k: [0.0, 0.0] for k in k_values}
    evaluated = 0

    for query in queries:
        relevant_ids = relevance.get(query.id)
        if not relevant_ids:
            continue

        ranked = [doc_id for doc_id, _ in TF_IDF(query.content)]
        evaluated += 1

        for k in k_values:
            precision, recall = precision_recall(ranked[:k], relevant_ids)
            totals[k][0] += precision
            totals[k][1] += recall

    print(f"Evaluated {evaluated} queries, skipped {len(queries) - evaluated} without judgments.")
    precisions = [totals[k][0] / evaluated for k in k_values]
    recalls = [totals[k][1] / evaluated for k in k_values]

    print(f"{'k':>4}  {'precision':>9}  {'recall':>7}")
    for k, precision, recall in zip(k_values, precisions, recalls):
        print(f"{k:>4}  {precision:>9.4f}  {recall:>7.4f}")

    plot_results(list(k_values), precisions, recalls)

if __name__ == "__main__":
    evaluate()
