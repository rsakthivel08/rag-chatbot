# ingest.py  —  Stage 1: Load documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_documents(data_dir: str = "data"):
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {data_dir}/")
    return documents

def chunk_documents(documents):
    """
    chunk_size=800   : each chunk is ~800 characters
    chunk_overlap=150: 150 chars of overlap between consecutive chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["source_type"] = "medical_guide"
    
    print(f"Split into {len(chunks)} chunks")
    return chunks

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR  = "chroma_store"
COLLECTION  = "medical_knowledge"

def embed_and_store(chunks):
    print("Loading embedding model (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    print("Embedding chunks and writing to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    count = vectorstore._collection.count()
    print(f"Done. {count} vectors stored in '{CHROMA_DIR}/'")
    return vectorstore


if __name__ == "__main__":
    docs   = load_documents("data")
    chunks = chunk_documents(docs)
    embed_and_store(chunks)