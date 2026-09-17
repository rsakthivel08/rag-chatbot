import os
import logging
from typing import Optional
 
# ── Third-party: env must be loaded BEFORE anything that reads env vars ───────
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ immediately
 
# Suppress noisy HuggingFace logs before model loads
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
 
# ── Streamlit (import after env is ready) ────────────────────────────────────
import streamlit as st
 
# ── Internal modules ──────────────────────────────────────────────────────────
from langchain_core.documents import Document
from rag_chain import build_chain
from ingest import (
    get_store,
    sync_data_dir,
    index_upload,
    delete_document,
    list_documents,
    chunk_locator,
)
from sarvam_client import (
    SUPPORTED_LANGUAGES,
    speech_to_text,
    translate_to_english,
    translate_from_english,
    text_to_speech,
)
from streamlit_mic_recorder import mic_recorder
 
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
 
# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS: list[str] = [
    "What are the main points covered in these documents?",
    "Summarize the key findings and conclusions.",
    "What topics or chapters does the document set cover?",
    "List the definitions or terms introduced in the documents.",
    "What recommendations does the document make?",
    "What data, statistics or numbers appear in the documents?",
    "What are the most important dates, names, or events mentioned?",
    "Compare the viewpoints presented across the documents.",
    "What questions do the documents leave unanswered?",
    "Give me a 5-bullet executive summary of the documents.",
]
 
# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="VIOLA — Document Knowledge Assistant",
    page_icon="📚",
    layout="centered",
)
 
 
# ── Cached resource loaders (loaded once per session) ─────────────────────────
@st.cache_resource
def get_chain():
    """Build and cache the RAG chain. Loads embedding model + connects to ChromaDB."""
    logger.info("Building RAG chain...")
    return build_chain()
 
 
@st.cache_resource
def get_vectorstore():
    """
    Shared process-wide vector store. First call per app launch runs the
    incremental data/ sync — exactly once, no matter how many browser
    sessions connect or actions are taken.
    """
    return get_store()


@st.cache_resource
def get_retriever():
    """Build and cache the retriever. Separate from chain for source citation fetching."""
    logger.info("Building retriever...")
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
 
 
# ── Session state initialisation ─────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all session state keys once per browser session."""
    defaults = {
        "messages": [],
        "pending":  None,
        "upload_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
 
 
init_session_state()
 
 
# ── Helper: fetch source citations ────────────────────────────────────────────
def fetch_sources(query: str) -> list[dict]:
    """
    Retrieves top-K source chunks for a query and formats them for display.
 
    Args:
        query: English-language query string.
 
    Returns:
        List of dicts with keys: file, page, preview.
    """
    retriever  = get_retriever()
    source_docs: list[Document] = retriever.invoke(query)
    if not source_docs:
        st.warning(
            "⚠️ Nothing matched in the knowledge base — if you expected results, "
            "upload documents in the sidebar (or drop them into `data/` and hit Rescan)."
        )
    seen: set[tuple[str, str, str]] = set()
    sources: list[dict] = []
    for doc in source_docs:
        locator = chunk_locator(doc)          # "page 3" or "section 7"
        preview = (doc.page_content or "").strip()[:300]
        key = (doc.metadata.get("source", "unknown"), locator, preview)
        if key in seen:                       # identical excerpt — skip
            continue
        seen.add(key)
        sources.append(
            {
                "file":    doc.metadata.get("source", "unknown"),
                "locator": locator,
                "preview": preview + ("..." if len((doc.page_content or "").strip()) > 300 else ""),
            }
        )
    return sources
 
 
# ── Helper: safe Sarvam STT ───────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, language_code: str) -> Optional[str]:
    """
    Wraps speech_to_text with error handling so a Sarvam API failure
    does not crash the Streamlit app.
 
    Returns:
        Transcript string, or None on failure.
    """
    try:
        transcript = speech_to_text(audio_bytes, language_code)
        if not transcript:
            st.warning("Could not transcribe audio. Please speak clearly and try again.")
            return None
        return transcript
    except Exception as e:
        logger.error(f"STT failed: {e}")
        st.error(f"Speech transcription failed: {e}")
        return None
 
 
# ── Helper: safe Sarvam translation ──────────────────────────────────────────
def safe_translate_to_english(text: str, language_code: str) -> str:
    """Translates to English. Falls back to original text on failure."""
    if language_code == "en-IN":
        return text
    try:
        return translate_to_english(text, language_code)
    except Exception as e:
        logger.error(f"Translation to EN failed: {e}")
        st.warning("Translation to English failed — querying in original language.")
        return text
 
 
def safe_translate_from_english(text: str, language_code: str) -> str:
    """Translates from English. Falls back to English text on failure."""
    if language_code == "en-IN":
        return text
    try:
        return translate_from_english(text, language_code)
    except Exception as e:
        logger.error(f"Translation from EN failed: {e}")
        st.warning("Translation back to your language failed — showing English answer.")
        return text
 
 
# ── Helper: safe Sarvam TTS ───────────────────────────────────────────────────
def speak_text(text: str, language_code: str) -> Optional[bytes]:
    """
    Wraps text_to_speech with error handling.
    Truncates to 500 chars (Sarvam bulbul limit).
 
    Returns:
        Audio bytes (WAV), or None on failure.
    """
    try:
        truncated = text[:500] + ("..." if len(text) > 500 else "")
        return text_to_speech(truncated, language_code)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        st.warning(f"Audio generation failed: {e}")
        return None
 
 
# ── Helper: render source citations ──────────────────────────────────────────
def render_sources(sources: list[dict]) -> None:
    """
    Renders the source-citation expander, grouped by file so repeated chunks
    from the same document collapse under one heading.
    Call inside a st.chat_message block.
    """
    if not sources:
        return
    by_file: dict[str, list[dict]] = {}
    for src in sources:
        by_file.setdefault(src["file"], []).append(src)
    with st.expander(f"📚 View sources ({len(sources)} excerpt{'s' if len(sources) != 1 else ''})"):
        for file, excerpts in by_file.items():
            locators = [e["locator"] for e in excerpts if e["locator"]]
            heading = f"📄 **{file}**"
            if locators:
                heading += f" — {', '.join(locators)}"
            st.markdown(heading)
            for e in excerpts:
                st.markdown(e["preview"])
            st.divider()
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 VIOLA")
    st.caption("Powered by RAG · GPT-OSS 120B on Groq")
    st.divider()
 
    # Sample questions
    st.markdown("**Try a sample question**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state.pending = q
 
    st.divider()
 
    # Language settings
    st.markdown("**Language settings**")
    selected_language: str = st.selectbox(
        "Select your language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=0,
    )
    language_code: str = SUPPORTED_LANGUAGES[selected_language]
 
    input_mode: str = st.radio(
        "Input mode",
        options=["Text", "Voice"],
        horizontal=True,
    )
    enable_tts: bool = st.toggle("Read answer aloud", value=False)
 
    st.divider()
 
    # Display controls
    show_sources: bool = st.toggle("Show source citations", value=True)

    st.divider()

    # Document management — upload and delete entirely from the UI
    st.markdown("**Documents**")
    st.caption(
        "Uploads are indexed instantly into the knowledge base "
    )

    uploaded = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md", "csv", "html", "htm"],
        key=f"doc_uploader_{st.session_state['upload_counter']}",
    )
    if uploaded:
        results: list[dict] = []
        errors: list[str] = []
        with st.spinner("Indexing uploaded documents..."):
            for uf in uploaded:
                try:
                    results.append(index_upload(uf))
                except Exception as e:
                    logger.error(f"Upload indexing failed for {uf.name}: {e}")
                    errors.append(f"{uf.name}: {e}")
        if results:
            replaced_ct = sum(1 for r in results if r["replaced"])
            msg = f"Indexed {len(results)} document(s)"
            if replaced_ct:
                msg += f" — {replaced_ct} replaced existing version(s)"
            st.toast(msg, icon="📚")
        for err in errors:
            st.error(f"Indexing failed — {err}")
        # Reset the uploader by rotating its key: a fresh widget key makes
        # Streamlit recreate the uploader empty, so this block cannot re-run
        # on later interactions (deleting the session-state key instead
        # restored the files and re-indexed on every question).
        st.session_state.upload_counter += 1
        st.rerun()

    # Indexed documents panel — with per-document delete
    try:
        indexed_docs = list_documents()
    except Exception as e:
        logger.error(f"Listing documents failed: {e}")
        indexed_docs = []
    if not indexed_docs:
        st.caption("Nothing indexed yet — upload a document above.")
    for doc in indexed_docs:
        name_col, del_col = st.columns([4, 1])
        with name_col:
            st.markdown(f"📄 **{doc['name']}**  \n{doc['chunks']} chunks · {doc['origin']}")
        if del_col.button("🗑️", key=f"del_{doc['name']}", use_container_width=True,
                          help=f"Remove {doc['name']} from the knowledge base"):
            with st.spinner(f"Deleting {doc['name']}..."):
                try:
                    delete_document(doc["name"])
                    st.toast(f"Deleted {doc['name']}", icon="🗑️")
                except Exception as e:
                    logger.error(f"Delete failed for {doc['name']}: {e}")
                    st.error(f"Delete failed: {e}")
            st.rerun()

    if st.button("🔄 Rescan data/ folder", use_container_width=True):
        with st.spinner("Scanning data/..."):
            try:
                rescan = sync_data_dir("data", force=True)
                if rescan["added"] or rescan["removed"]:
                    st.toast(
                        f"Indexed {len(rescan['added'])}, "
                        f"removed {len(rescan['removed'])}",
                        icon="🔄",
                    )
                else:
                    st.toast("Knowledge base already up to date", icon="✅")
            except Exception as e:
                logger.error(f"Rescan failed: {e}")
                st.error(f"Rescan failed: {e}")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
 
    st.divider()
    st.caption(
        "⚠️ This assistant answers only from your document collection. "
        "Always verify important information against the original documents."
    )
 
 
# ── Main area header ──────────────────────────────────────────────────────────
st.title("VIOLA 📚")
st.caption(
    "Hello! I am Viola, your friendly document assistant. "
    "Upload files in the sidebar (PDF, DOCX, TXT, MD, CSV, HTML) "
    "then ask me anything about their contents."
)
 
# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (
            msg["role"] == "assistant"
            and show_sources
            and msg.get("sources")
        ):
            render_sources(msg["sources"])
 
 
# ── Resolve input: text typed, sidebar button, or voice recording ─────────────
question: Optional[str] = None
 
if input_mode == "Voice":
    st.caption(f"🎙️ Speak in **{selected_language}** — press the button to record")
    audio = mic_recorder(
        start_prompt="🎙️ Start speaking",
        stop_prompt="⏹️ Stop recording",
        key="viola_mic",
    )
    if audio and audio.get("bytes"):
        with st.spinner("Transcribing your speech..."):
            question = transcribe_audio(audio["bytes"], language_code)
        if question:
            st.info(f"**You said:** {question}")
else:
    # Text input — check sidebar button first, then chat input
    placeholder = (
        "Ask me anything..." if language_code == "en-IN"
        else f"Ask in {selected_language} or English..."
    )
    typed = st.chat_input(placeholder)
    if st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None
    elif typed:
        question = typed
 
 
# ── Handle new question ───────────────────────────────────────────────────────
if question:
    # 1. Display and store user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    # 2. Translate question to English for RAG (RAG chain only works in English)
    with st.spinner("Processing..."):
        english_question = safe_translate_to_english(question, language_code)
    logger.info(f"English query: {english_question}")
 
    # 3. Fetch source citations using the English query
    sources = fetch_sources(english_question)
 
    # 4. Generate answer (stream directly in English first)
    with st.chat_message("assistant"):
        chain = get_chain()
 
        try:
            english_answer: str = st.write_stream(
                chain.stream(english_question)
            )
        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            st.error(f"Failed to generate answer: {e}")
            english_answer = ""
 
        # 5. Translate answer back to user's language if needed
        if english_answer and language_code != "en-IN":
            with st.spinner(f"Translating to {selected_language}..."):
                translated_answer = safe_translate_from_english(
                    english_answer, language_code
                )
            st.markdown(f"**{selected_language} translation:**")
            st.markdown(translated_answer)
        else:
            translated_answer = english_answer
 
        # 6. Show source citations
        if show_sources:
            render_sources(sources)
 
        # 7. Text-to-speech (runs on translated answer, falls back to English)
        if enable_tts and translated_answer:
            with st.spinner("Generating audio..."):
                audio_bytes = speak_text(translated_answer, language_code)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
 
    # 8. Persist assistant message to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": english_answer,          # store English for consistency
        "sources": sources,
    })