import math
import re
from collections import Counter

from document import Document

with open(file="CISI.txt", mode="r", encoding="utf-8") as file:
    text = file.read()

documents = []

#Converts the raw dataset text to structured documents.
records = text.split(".I ")[1:]

for record in records:
    doc_id = int(record.split("\n", 1)[0].strip())

    title_match = re.search(r"\.T[ \t]*\n(.*?)\n\.A", record, re.DOTALL)
    author_match = re.search(r"\.A[ \t]*\n(.*?)\n\.W", record, re.DOTALL)
    content_match = re.search(r"\.W[ \t]*\n(.*?)\n\.X", record, re.DOTALL)

    title = title_match.group(1).strip() if title_match else ""
    author = author_match.group(1).strip() if author_match else ""
    content = content_match.group(1).strip() if content_match else ""

    documents.append(Document(id=doc_id, title=title, author=author, content=content))

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


query = input("Enter your query:")

docs_by_id = {document.id: document for document in documents}

results = TF_IDF(query=query)
if not results:
    print("No results found.")
for doc_id, score in results[:10]:
    print(f"{score:.4f}  [{doc_id}] {docs_by_id[doc_id].title}")


