# app.py  —  Streamlit web interface
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"   # suppress HuggingFace noise

import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_chain
from retriever import build_retriever
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(
    page_title="Pediatrics Knowledge Assistant",
    page_icon="🏥",
    layout="centered",
)

SAMPLE_QUESTIONS = [
    "What are the warning signs of dehydration in children?",
    "How is acute otitis media diagnosed and treated in pediatric patients?",
    "What is the recommended vaccination schedule for infants?",
    "How do you assess growth and development milestones in a 2-year-old child?",
    "What are the common causes of fever in newborns?",
    "How is pediatric asthma diagnosed and managed?",
    "What are the symptoms and treatment options for bronchiolitis?",
    "When should a child with diarrhea be referred for emergency care?",
    "What are the clinical features of neonatal jaundice?",
    "How is iron deficiency anemia managed in children?",
]

@st.cache_resource
def get_chain():
    return build_chain()

@st.cache_resource
def get_retriever():
    return build_retriever(k=5)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.title("🏥 VIOLA")
    st.caption("Powered by RAG · Llama 3.1 on Groq")
    st.divider()

    st.markdown("**Try a sample question**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

    st.divider()
    show_sources = st.toggle("Show source citations", value=True)
    
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []

    st.divider()
    st.caption(
        "⚠️ This assistant is for informational purposes only. "
        "Always consult a qualified healthcare professional for medical advice."
    )

st.title("VIOLA🩺")
st.caption("HELLO! I am Viola your friendly pediatric knowledge assistant. Ask me any questions regarding symptoms, treatments, medications, and more.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources if stored and toggle is on
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("View sources"):
                for src in msg["sources"]:
                    st.markdown(f"📄 **{src['file']}** — Page {src['page']}")
                    st.caption(src["preview"])

question = st.chat_input("Ask a medical question...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chain     = get_chain()
        retriever = get_retriever()
        
        source_docs: list[Document] = retriever.invoke(question)
        sources = [
            {
                "file":    doc.metadata.get("source", "unknown"),
                "page":    doc.metadata.get("page", "?"),
                "preview": doc.page_content[:200] + "...",
            }
            for doc in source_docs
        ]
        
        response = st.write_stream(chain.stream(question))
        
        if show_sources and sources:
            with st.expander("View sources"):
                for src in sources:
                    st.markdown(f"📄 **{src['file']}** — Page {src['page']}")
                    st.caption(src["preview"])

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response,
        "sources": sources,
    })