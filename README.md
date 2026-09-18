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

## Implementation

The app is a five-stage pipeline: **ingest → embed → retrieve → generate → speak**.

**1. Document ingestion (`ingest.py`)**
- Supports **PDF, DOCX, TXT, MD, CSV, HTML**. Every format is loaded through a
  matching LangChain loader (from memory for UI uploads, from disk for `data/`),
  and blank/scanned pages with no extractable text are dropped.
- Text is split with a `RecursiveCharacterTextSplitter` (800 chars, 150 overlap),
  which keeps paragraphs and sentences intact so chunks stay readable.
- Each document gets a **fingerprint** — a content hash for uploads, name+size+mtime
  for `data/` files — that becomes the prefix of every chunk ID. This makes the
  store self-managing: re-uploading the same file is a no-op, uploading a new
  version of a filename replaces the old chunks, and deletion is a precise
  fingerprint match rather than a guess.
- The vector store is the **single source of truth**. Uploads index straight into
  ChromaDB from the sidebar — no folder workflow required — while `data/` remains
  an optional extra managed by an incremental sync that never touches uploads.

**2. Embedding & storage**
- Chunks are embedded locally with **all-MiniLM-L6-v2** (fast, domain-agnostic,
  no API cost) and persisted in **ChromaDB** on disk, so the knowledge base
  survives restarts. The embedding model and store are process-wide singletons,
  loaded and synced exactly once per app launch.

**3. Retrieval (`retriever.py`)**
- A question is embedded with the same model and the **top-5 most similar
  chunks** are pulled by cosine similarity per query.

**4. Generation (`rag_chain.py`)**
- Retrieved chunks are formatted into a labeled context block and sent to
  **GPT-OSS 120B on Groq** (`temperature=0`, `reasoning_effort="low"`) with a
  strict system prompt: answer only from context, cite the file with page/section,
  keep it short, and say honestly when the knowledge base doesn't contain the answer.
- Citations are format-aware: PDFs carry real page numbers from the loader, while
  non-paginated formats (DOCX, TXT, …) fall back to section locators derived from
  chunk order — so references stay meaningful for every file type.

**5. Multilingual voice layer (`sarvam_client.py` + `app.py`)**
- **STT** (Saaras v3) transcribes recorded speech, **translation** (Mayura v1)
  moves questions to English and answers back, and **TTS** (Bulbul v3) reads
  answers aloud. All three are called through one guarded client with timeouts,
  chunked long-text translation, and graceful fallbacks — an API failure degrades
  the experience but never crashes the app.
- API calls happen **only** for STT, TTS, and actual translation; English and
  `en-IN` paths short-circuit to zero Sarvam usage.

**UI (`app.py`)** — Streamlit chat interface with sample questions, streaming
answers, grouped source-citation expanders, a Documents panel (per-file delete),
language/input-mode settings, and automatic environment handling for both local
`.env` and Streamlit Cloud secrets.

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