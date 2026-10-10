from embeddings import embedding_search
from search import TF_IDF
from sklearn_search import sklearn_search

# Which TF-IDF scores the hybrid blends with embeddings: "tfidf" (hand-written) or "sklearn".
TFIDF_METHOD = "sklearn"

# Weight of the TF-IDF score; the embedding score gets (1 - ALPHA).
ALPHA = 0.5

_TFIDF_SEARCHES = {"tfidf": TF_IDF, "sklearn": sklearn_search}
if TFIDF_METHOD not in _TFIDF_SEARCHES:
    raise ValueError(f"Unknown TFIDF_METHOD {TFIDF_METHOD!r}; expected 'tfidf' or 'sklearn'")
_tfidf_search = _TFIDF_SEARCHES[TFIDF_METHOD]


def _normalize(scores):
    """Min-max scales a {doc_id: score} dict to [0, 1]."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: (score - lo) / (hi - lo) for doc_id, score in scores.items()}


def hybrid_search(query, alpha=ALPHA, tfidf_search=None):
    """tfidf_search overrides the TFIDF_METHOD choice (e.g. for comparing runs)."""
    tfidf_search = tfidf_search or _tfidf_search
    tfidf_norm = _normalize(dict(tfidf_search(query)))
    emb_norm = _normalize(dict(embedding_search(query)))

    # The hand-written TF-IDF only returns docs with a non-zero score; a missing doc counts as 0.
    final = {
        doc_id: alpha * tfidf_norm.get(doc_id, 0.0) + (1 - alpha) * emb_score
        for doc_id, emb_score in emb_norm.items()
    }
    return sorted(final.items(), key=lambda item: item[1], reverse=True)
