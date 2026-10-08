from search import documents, TF_IDF


def main():
    query = input("Enter your query:")

    docs_by_id = {document.id: document for document in documents}

    results = TF_IDF(query=query)
    if not results:
        print("No results found.")
    for doc_id, score in results[:10]:
        print(f"{score:.4f}  [{doc_id}] {docs_by_id[doc_id].title}")


if __name__ == "__main__":
    main()
