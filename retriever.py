# retriever.py — similarity-search retriever over the shared vector store
from ingest import get_store


def build_retriever(k: int = 5):
    """
    Retriever over the shared (singleton) vector store.
    k = number of most-similar chunks returned per query.
    """
    return get_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    retriever = build_retriever(k=5)
    results = retriever.invoke("What are the main points covered in these documents?")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])
