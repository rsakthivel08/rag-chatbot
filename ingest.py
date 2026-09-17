# ingest.py  —  Document ingestion + incremental syncing
"""
The vector store is the single source of truth for uploaded documents.

- `get_store()`       : process-wide singleton store (created once per launch).
- `index_upload()`    : chunk + embed an uploaded file straight into the store.
                        Nothing is written to disk; no data/ folder required.
- `delete_document()` : remove a document's chunks from the store instantly.
- `list_documents()`  : per-document chunk counts for UI display.
- `sync_data_dir()`   : OPTIONAL — indexes files placed in data/ (add/update/
                        remove). Never touches uploaded documents.

Run `python ingest.py` for a manual data/ sync (optional); `--rebuild` wipes
the collection and re-indexes data/ from scratch (uploads are cleared — the
only operation that removes them).
"""
import hashlib
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".htm"}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR  = "chroma_store"
COLLECTION  = "documents"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
ADD_BATCH_SIZE = 4000  # Chroma rejects single adds above ~5461 vectors


# ── Loading ──────────────────────────────────────────────────────────────────
def _loader_for(suffix: str):
    """Return a LangChain loader factory for the file type, or None."""
    suffix = suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader
    if suffix == ".docx":
        return Docx2txtLoader
    if suffix == ".csv":
        return lambda p: CSVLoader(p, encoding="utf-8")
    if suffix in {".html", ".htm"}:
        return BSHTMLLoader
    if suffix == ".md":
        return UnstructuredMarkdownLoader
    if suffix in {".txt", ".log"}:
        return lambda p: TextLoader(p, encoding="utf-8")
    return None


def _load_bytes(data: bytes, name: str) -> list[Document]:
    """Load a document from raw bytes; keep only pages with real text."""
    suffix = Path(name).suffix
    factory = _loader_for(suffix)
    if factory is None:
        raise ValueError(
            f"Unsupported file type: {name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    # Loaders work on paths; write a temp file with the right extension
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        docs = factory(tmp.name).load()
    finally:
        os.unlink(tmp.name)

    kept: list[Document] = []
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue  # scanned/blank page — indexing it only pollutes results
        d.page_content = text
        d.metadata["source"] = name  # short, citation-friendly
        kept.append(d)
    return kept


# ── Fingerprints (stable chunk-ID prefix → no duplicates, precise deletes) ──
def _upload_fingerprint(name: str, data: bytes) -> str:
    """Content hash — same file re-uploaded maps to the same fingerprint."""
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"H:{name}:{digest}"


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _collect_files(data_dir: str) -> dict[str, Path]:
    """Map fingerprint -> path for every supported file under data_dir."""
    data_path = Path(data_dir)
    files: dict[str, Path] = {}
    if not data_path.exists():
        return files
    for path in sorted(data_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files[_file_fingerprint(path)] = path
    return files


def _chunk_ids(fingerprint: str, n_chunks: int) -> list[str]:
    return [f"FP:{fingerprint}|{i}" for i in range(n_chunks)]


def _stored_fingerprints(vectorstore: Chroma) -> set[str]:
    """File fingerprints present in the store (parsed from chunk IDs)."""
    stored = vectorstore._collection.get(include=[])
    return {
        chunk_id.split("|")[0][3:]
        for chunk_id in stored["ids"]
        if chunk_id.startswith("FP:")
    }


def _fingerprint_name(fingerprint: str) -> str:
    """Document name from a fingerprint: 'H:<name>:<hash>' or '<name>:<size>:<mtime>'."""
    if fingerprint.startswith("H:"):
        return fingerprint[2:].rsplit(":", 1)[0]
    return fingerprint.split(":")[0]


# ── Citations: human-friendly locator for a chunk ───────────────────────────
def chunk_locator(chunk: Document) -> str:
    """
    Locate a chunk inside its document for citations.

    - PDF pages carry a 'page' metadata key (0-based from PyPDFLoader)
      → "page 3" (displayed 1-based).
    - Other formats (DOCX, TXT, MD, CSV, HTML) have no real pagination, so
      fall back to the chunk's ordinal position → "section 7".
    """
    metadata = (chunk if isinstance(chunk, dict) else getattr(chunk, "metadata", None)) or {}
    page = metadata.get("page")
    if isinstance(page, int) and page >= 0:
        return f"page {page + 1}"
    idx = metadata.get("chunk_index")
    if isinstance(idx, int) and idx >= 0:
        return f"section {idx + 1}"
    return ""


# ── Splitting ────────────────────────────────────────────────────────────────
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = chunk.page_content.strip()
        chunk.metadata.setdefault("source_type", "document")
    return [c for c in chunks if c.page_content]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Backwards-compatible wrapper (adds global chunk_index)."""
    chunks = split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


# ── Process-wide singletons (load model + store exactly once per launch) ────
_EMBEDDINGS = None
_STORE = None
_SYNCED = False


def get_embeddings() -> HuggingFaceEmbeddings:
    """Embedding model singleton — heavy model loads only once per process."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMBEDDINGS


def get_store() -> Chroma:
    """Vector store singleton. Syncs optional data/ once per process."""
    global _STORE, _SYNCED
    if _STORE is None:
        _STORE = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        # Run the once-per-process sync here, inside creation, so that
        # sync_data_dir() calling back into get_store() cannot recurse
        # (the store already exists by then).
        if not _SYNCED:
            try:
                sync_data_dir("data")
            except Exception as e:
                print(f"! optional data/ sync failed: {e}")
            _SYNCED = True
    return _STORE


def new_store(embeddings=None) -> Chroma:
    """Kept for backwards compatibility — returns the shared store."""
    return get_store()


# ── Core: add/remove chunks in the store ─────────────────────────────────────
def _add_documents(vectorstore: Chroma, fingerprint: str, docs: list[Document]) -> int:
    chunks = split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    if chunks:
        ids = _chunk_ids(fingerprint, len(chunks))
        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start : start + ADD_BATCH_SIZE]
            vectorstore.add_documents(batch, ids=ids[start : start + ADD_BATCH_SIZE])
    return len(chunks)


def _delete_fingerprints(vectorstore: Chroma, fingerprints: set[str]) -> int:
    """Delete all chunks whose fingerprint matches; returns number removed."""
    if not fingerprints:
        return 0
    all_ids = vectorstore._collection.get(include=[])["ids"]
    ids_to_delete = [cid for cid in all_ids if cid.split("|")[0][3:] in fingerprints]
    if ids_to_delete:
        vectorstore._collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def _fingerprints_for_name(vectorstore: Chroma, name: str) -> set[str]:
    """All stored fingerprints belonging to a document filename."""
    return {
        fp for fp in _stored_fingerprints(vectorstore)
        if _fingerprint_name(fp) == name
    }


# ── Public API: upload from the UI (straight into the store) ────────────────
def index_upload(uploaded_file) -> dict:
    """
    Chunk + embed a Streamlit UploadedFile directly into the vector store.
    Nothing is written to disk — no data/ folder involved.

    Uploading a file with an existing name replaces the old version (older
    chunks are removed). Re-uploading identical content is a no-op.

    Returns {"name", "replaced", "chunks", "total"}.
    """
    name = Path(uploaded_file.name).name  # strip any path components
    data = uploaded_file.getvalue()

    vectorstore = get_store()
    new_fp = _upload_fingerprint(name, data)
    existing = _fingerprints_for_name(vectorstore, name)

    if new_fp in existing:
        return {"name": name, "replaced": False, "chunks": 0,
                "total": vectorstore._collection.count(), "unchanged": True}

    # Replace any older version of this filename
    stale = existing - {new_fp}
    if stale:
        _delete_fingerprints(vectorstore, stale)

    n_chunks = _add_documents(vectorstore, new_fp, _load_bytes(data, name))
    print(f"+ indexed {name} ({n_chunks} chunks)")

    return {"name": name, "replaced": bool(stale), "chunks": n_chunks,
            "total": vectorstore._collection.count(), "unchanged": False}


# ── Public API: delete a document from the store ────────────────────────────
def delete_document(name: str, data_dir: str = "data") -> int:
    """
    Remove a document completely from the knowledge base (all its chunks).

    For uploads: content is gone from the app (nothing exists on disk).
    For data/ files: chunks are removed but the file on disk is NOT touched
    (it will be re-indexed on the next sync — delete the file yourself if you
    want it gone for good).

    Returns the number of chunks removed.
    """
    vectorstore = get_store()
    n = _delete_fingerprints(vectorstore, _fingerprints_for_name(vectorstore, name))
    print(f"- removed {n} chunks for {name}")
    return n


# ── Public API: list documents for the UI ────────────────────────────────────
def list_documents(data_dir: str = "data") -> list[dict]:
    """
    Per-document summary for the sidebar Documents panel:
    [{"name", "chunks", "origin"}], sorted by name.
    origin is "upload" (content-hash fingerprint) or "data/" file.
    """
    vectorstore = get_store()
    counts: dict[str, int] = {}
    origins: dict[str, str] = {}
    for cid in vectorstore._collection.get(include=[])["ids"]:
        if cid.startswith("FP:"):
            fp = cid.split("|")[0][3:]
            name = _fingerprint_name(fp)
            counts[name] = counts.get(name, 0) + 1
            origins[name] = "upload" if fp.startswith("H:") else "data/"

    return [
        {"name": name, "chunks": n, "origin": origins.get(name, "data/")}
        for name, n in sorted(counts.items())
    ]


# ── Public API: optional sync of data/ (never touches uploads) ──────────────
def sync_data_dir(data_dir: str = "data", force: bool = False) -> dict:
    """
    OPTIONAL: incrementally sync data/ into the vector store.

    - Runs at most once per process unless force=True (sidebar Rescan / CLI).
    - New or modified files  → chunked + embedded
    - Deleted files          → their chunks are removed
    - Uploaded documents     → NEVER touched (they live only in the store)
    - Unchanged files        → untouched (fast, can never duplicate)

    Returns {"added": [...], "removed": [...], "total": <chunk count>}
    """
    global _SYNCED
    if _SYNCED and not force:
        return {"added": [], "removed": [], "total": get_store()._collection.count()}

    vectorstore = get_store()
    present = _collect_files(data_dir)
    stored  = _stored_fingerprints(vectorstore)

    # Remove chunks of data/-files that no longer exist on disk.
    # Upload fingerprints ("H:...") can never match a data-file fingerprint
    # ("<name>:<size>:<mtime>"), so uploads are structurally safe here.
    removed = [
        fp for fp in sorted(stored)
        if not fp.startswith("H:") and fp not in present
    ]
    if removed:
        _delete_fingerprints(vectorstore, set(removed))

    # Add files that are new or modified
    added: list[str] = []
    for fp, path in present.items():
        if fp in stored:
            continue
        try:
            _add_documents(vectorstore, fp, _load_bytes(path.read_bytes(), path.name))
            added.append(path.name)
            print(f"+ indexed {path.name}")
        except Exception as e:
            print(f"! failed to index {path.name}: {e}")

    _SYNCED = True
    return {
        "added":   added,
        "removed": [_fingerprint_name(fp) for fp in removed],
        "total":   vectorstore._collection.count(),
    }


# ── Full rebuild (CLI --rebuild) — the only op that clears uploads ──────────
def full_rebuild(data_dir: str = "data") -> int:
    """
    Wipe the collection and re-index everything in data/ from scratch.
    WARNING: uploaded documents are cleared by a rebuild (they only exist in
    the store). Re-upload them afterwards if needed.
    """
    global _STORE, _SYNCED
    _SYNCED = True  # we manage the store manually here; skip auto-sync
    try:
        get_store().delete_collection()
        print(f"Cleared existing collection '{COLLECTION}'")
    except Exception:
        pass  # nothing stored yet
    _STORE = None
    get_store()  # recreate empty store

    files = _collect_files(data_dir)
    docs: list[Document] = []
    for fp, path in files.items():
        try:
            docs.extend(_load_bytes(path.read_bytes(), path.name))
        except Exception as e:
            print(f"! failed to load {path.name}: {e}")
    print(f"Loaded {len(docs)} pages/sections from {data_dir}/")
    if not docs:
        print(
            "No supported documents found in data/ "
            "(supported: PDF, DOCX, TXT, MD, CSV, HTML)"
        )
        return 0

    chunks = chunk_documents(docs)
    print("Embedding chunks and writing to ChromaDB...")

    # Per-file stable IDs so later syncs can manage these chunks
    counts: dict[str, int] = {}
    for c in chunks:
        name = Path(c.metadata.get("source", "")).name
        counts[name] = counts.get(name, 0) + 1
    fp_by_name = {Path(p).name: fp for fp, p in files.items()}

    all_ids: list[str] = []
    idx_by_name: dict[str, int] = {}
    for c in chunks:
        name = Path(c.metadata.get("source", "")).name
        i = idx_by_name.get(name, 0)
        all_ids.append(f"FP:{fp_by_name[name]}|{i}")
        idx_by_name[name] = i + 1

    store = get_store()
    for start in range(0, len(chunks), ADD_BATCH_SIZE):
        store.add_documents(
            chunks[start : start + ADD_BATCH_SIZE],
            ids=all_ids[start : start + ADD_BATCH_SIZE],
        )
    count = store._collection.count()
    print(f"Done. {count} vectors stored in '{CHROMA_DIR}/'")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index documents into ChromaDB")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="wipe the collection and re-index everything in data/ (clears uploads!)",
    )
    args = parser.parse_args()

    if args.rebuild:
        full_rebuild("data")
    else:
        result = sync_data_dir("data", force=True)
        print(
            f"Sync complete: {len(result['added'])} added, "
            f"{len(result['removed'])} removed, "
            f"{result['total']} chunks total."
        )
