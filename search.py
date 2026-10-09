import math
import re
from collections import Counter

from parsing import parse_file

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself
yourselves
""".split())

documents = parse_file("CISI.txt")

document_term_frequency_index = {}
vocabulary = {}

for document in documents:
    words = [w for w in re.findall(r"[a-zA-Z]+", document.content.lower()) if w not in STOPWORDS]
    word_counts = Counter(words)
    document_term_frequency_index[document.id] = dict(word_counts)

    for word in word_counts:
        vocabulary[word] = vocabulary.get(word, 0) + 1

idf_index = {word: math.log(len(documents) / df) for word, df in vocabulary.items()}

def term_frequency(word, document):
    word_counts = document_term_frequency_index[document.id]
    total_doc_word_count = sum(word_counts.values())

    if total_doc_word_count == 0:
        return 0

    doc_word_count = word_counts.get(word, 0)

    return doc_word_count/total_doc_word_count

def TF_IDF(query):
    words = set(re.findall(r"[a-zA-Z]+", query.lower()))
    scores = {}
    for document in documents:
        score = 0
        for word in words:
            if word not in idf_index:
                continue
            score += term_frequency(word, document) * idf_index[word]
        if score > 0:
            scores[document.id] = score

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def get_search_function(method):
    """Returns the search function for "tfidf" or "embeddings". The embeddings
    module is imported lazily so TF-IDF runs never load the model."""
    if method == "tfidf":
        return TF_IDF
    if method == "embeddings":
        from embeddings import embedding_search
        return embedding_search
    raise ValueError(f"Unknown method {method!r}; expected 'tfidf' or 'embeddings'")
