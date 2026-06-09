from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR  = "chroma_store"
COLLECTION  = "medical_knowledge"

def build_retriever(k: int = 5):
    #k=5 return the 5 most similar chunks to the query.
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",  
        search_kwargs={"k": k},
    )
    return retriever



if __name__ == "__main__":
    retriever = build_retriever(k=5)
    results = retriever.invoke("What are the symptoms of diabetes?")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:300])