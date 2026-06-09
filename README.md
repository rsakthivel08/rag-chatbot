# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for medical knowledge queries,
built with LangChain, ChromaDB, and Llama 3.1 via Groq.

## Tech Stack
- **LLM:** Llama 3.1 8B via Groq (free)
- **Embeddings:** all-MiniLM-L6-v2 (local)
- **Vector DB:** ChromaDB
- **Framework:** LangChain
- **UI:** Streamlit

## Setup
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Add your Groq API key to `.env`: `GROQ_API_KEY=your_key`
6. Add medical PDFs to `data/`
7. Run ingestion: `python ingest.py`
8. Launch app: `streamlit run app.py`

## Project Structure
```
medical-rag-chatbot/
├── data/              # Put your medical PDFs here
├── ingest.py          # Chunk, embed, store documents
├── retriever.py       # Similarity search
├── rag_chain.py       # Prompt + LLM chain
├── app.py             # Streamlit UI
├── requirements.txt
└── .env.example
```