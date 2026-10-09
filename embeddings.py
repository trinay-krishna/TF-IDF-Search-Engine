from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from parsing import parse_file

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE = Path(__file__).resolve().parent / "cache"

model = SentenceTransformer(MODEL_NAME)

documents = parse_file("CISI.txt", "document")
doc_ids = [d.id for d in documents]


def _load_or_build_vectors():
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"doc_vectors_{MODEL_NAME}.npy"
    if path.exists():
        return np.load(path)

    texts = [d.title + ". " + d.content for d in documents]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(path, vectors)
    return vectors


doc_vectors = _load_or_build_vectors()


def embedding_search(query):
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = doc_vectors @ q                 # cosine similarity (vectors are unit length)
    order = np.argsort(-sims)              # best first
    return [(doc_ids[i], float(sims[i])) for i in order]
