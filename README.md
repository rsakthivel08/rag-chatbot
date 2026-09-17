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
5. Add your API keys to `.env` (see `.env.example`): `GROQ_API_KEY`, `SARVAM_API_KEY`, optionally `HF_TOKEN`
6. Launch app: `streamlit run app.py`

   The sidebar uploader indexes documents **instantly into the knowledge base**
   — nothing to run, no folder needed. The Documents panel lists everything
   indexed and deletes a document (all its content) with one click. Optionally,
   files can also be dropped into `data/` and picked up via the Rescan button
   (deleting a data/ file from the UI removes its content but keeps the file).
   `python ingest.py` does a manual data/ sync; `--rebuild` wipes and re-indexes
   data/ from scratch (uploads are cleared — re-upload them afterwards).

## Project Structure
```
rag-chatbot/
├── app.py             # Streamlit UI
├── ingest.py          # Chunk, embed, store documents
├── retriever.py       # Similarity search
├── rag_chain.py       # Prompt + LLM chain
├── sarvam_client.py   # Sarvam AI (STT / translation / TTS)
├── requirements.txt
└── .env.example
```

## Deploy to Streamlit Community Cloud (free)

1. Make sure this repo is on GitHub (it is: `rsakthivel08/rag-chatbot`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** (or **New app**) → **Paste GitHub URL** → select
   `rsakthivel08/rag-chatbot`, branch `main`, main file `app.py`.
4. Click **Advanced settings** → paste your secrets:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   SARVAM_API_KEY = "your_sarvam_key"
   HF_TOKEN = "your_huggingface_token"
   ```
   (These replace the local `.env` — the app merges them automatically.)
5. Click **Deploy**.

**What to expect**
- **First boot is slow (5–10 min)**: the free instance downloads PyTorch and
  the MiniLM embedding model. Later boots are faster; inactivity reboots
  repeat this cold start.
- **The knowledge base starts empty** — `chroma_store/` is not in git. Upload
  documents through the sidebar after deployment; they index in seconds.
- Free tier has ~1 GB RAM; the embedding model fits comfortably.
- Voice input works out of the box (Cloud serves the app over HTTPS, which
  browsers require for microphone access).
- If the app crashes on boot, check **Manage app → Logs** — usually a missing
  or misspelled secret.