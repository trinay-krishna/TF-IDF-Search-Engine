import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parsing import parse_file
from search import STOPWORDS

documents = parse_file("CISI.txt", "document")
doc_ids = [d.id for d in documents]

# Same tokenization and stopword list as the hand-written TF-IDF in search.py.
vectorizer = TfidfVectorizer(
    token_pattern=r"[a-zA-Z]+",
    stop_words=list(STOPWORDS),
)
doc_matrix = vectorizer.fit_transform([d.content for d in documents])


def sklearn_search(query):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, doc_matrix).ravel()   # one score per document
    order = np.argsort(-sims, kind="stable")          # best first, ties keep document order
    return [(doc_ids[i], float(sims[i])) for i in order]
