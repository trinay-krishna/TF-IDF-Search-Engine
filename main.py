from parsing import parse_file
from search import get_search_function

# Which search to use: "tfidf" or "embeddings". Change here as needed.
METHOD = "tfidf"


def main(method=METHOD):
    query = input("Enter your query:")

    docs_by_id = {document.id: document for document in parse_file("CISI.txt")}

    results = get_search_function(method)(query)
    if not results:
        print("No results found.")
    for doc_id, score in results[:10]:
        print(f"{score:.4f}  [{doc_id}] {docs_by_id[doc_id].title}")


if __name__ == "__main__":
    main()
