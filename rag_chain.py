# rag_chain.py  —  Stage 5+6: Prompt construction and LLM generation
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from retriever import build_retriever
from ingest import chunk_locator


SYSTEM_PROMPT = """You are a knowledgeable, friendly and joyful Document Assistant.
Your role is to help users understand the contents of their documents clearly and accurately.

IMPORTANT RULES:
- Base your answer on the context provided below, which contains excerpts from
  the user's document collection. Never invent facts that are not there.
- If the context is partially relevant, answer with what it does contain and
  note the parts you could not find. Do not refuse just because the context
  is incomplete.
- Cite the source file with its page or section when the answer comes from a
  specific chunk, e.g. (report.pdf, page 12) or (notes.docx, section 4).
  Use exactly the page/section shown in the chunk's [Source: ...] label.
- Only if the context is clearly unrelated to the question, say:
  "I don't have enough information in my knowledge base to answer this
  confidently." and briefly suggest what to check or add.
- Keep the answers short and clear, answering the user's query in bullet point
  format with main points only.
- Be friendly wherever possible.

Context retrieved from the document collection:
{context}
"""

def _format_docs(docs: list[Document]) -> str:
    """
    Formats retrieved chunks into a readable context block.
    Labels each chunk with its source file and a human locator
    ("page 3" for PDFs, "section 7" for formats without pagination).
    """
    sections = []
    for doc in docs:
        source  = doc.metadata.get("source", "unknown")
        locator = chunk_locator(doc)
        label   = f"{source}, {locator}" if locator else source
        sections.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    """
    LLM: openai/gpt-oss-120b on Groq (llama-3.1-8b-instant was deprecated)
      - temperature=0
      - reasoning_effort='low' keeps latency low; remove if your account
        or SDK version does not support it
    """
    retriever = build_retriever(k=5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  "{question}"),
    ])

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1024,
        max_retries=2,
        reasoning_effort="low",   # GPT-OSS is a reasoning model; low = fastest
    )


    chain = (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    chain = build_chain()
    question = "What are the main points covered in these documents?"
    print(f"Q: {question}\n")
    print("A: ", end="", flush=True)
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print()