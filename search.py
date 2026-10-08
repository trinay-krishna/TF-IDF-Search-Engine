import math
import re
from collections import Counter

from parsing import parse_file

documents = parse_file("CISI.txt")

document_term_frequency_index = {}
vocabulary = {}

for document in documents:
    words = re.findall(r"[a-zA-Z]+", document.content.lower())
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
