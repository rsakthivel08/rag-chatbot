# Document RAG Chatbot

A general-purpose Retrieval-Augmented Generation (RAG) chatbot that answers
questions from any collection of documents, built with LangChain, ChromaDB,
and GPT-OSS 120B via Groq.

## Tech Stack
- **LLM:** GPT-OSS 120B via Groq (`openai/gpt-oss-120b`, free tier)
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
6. Add your documents to `data/` — supported formats: **PDF, DOCX, TXT, MD, CSV, HTML**
7. Launch app: `streamlit run app.py`

   The sidebar uploader indexes documents **instantly into the knowledge base**
   — nothing to run, no folder needed. The Documents panel lists everything
   indexed and deletes a document (all its content) with one click. Optionally,
   files can also be dropped into `data/` and picked up via the Rescan button
   (deleting a data/ file from the UI removes its content but keeps the file).
   `python ingest.py` does a manual data/ sync; `--rebuild` wipes and re-indexes
   data/ from scratch (uploads are cleared — re-upload them afterwards).

## Project Structure
```
medical-rag-chatbot/
├── data/              # Put your documents here (PDF, DOCX, TXT, MD, CSV, HTML)
├── ingest.py          # Chunk, embed, store documents
├── retriever.py       # Similarity search
├── rag_chain.py       # Prompt + LLM chain
├── app.py             # Streamlit UI
├── requirements.txt
└── .env.example
```