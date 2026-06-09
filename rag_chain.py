# rag_chain.py  —  Stage 5+6: Prompt construction and LLM generation
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from retriever import build_retriever


SYSTEM_PROMPT = """You are a knowledgeable,friendly and joyful Pediatric Knowledge Assistant.
Your role is to help users understand medical information clearly and accurately and in a friendly manner.

IMPORTANT RULES:
- Answer ONLY using the context provided below. Do not use outside knowledge.
- If the context does not contain enough information, say:
  "I don't have enough information in my knowledge base to answer this confidently.
   Please consult a qualified healthcare professional."
- Be playful and friendly wherever possible.
- Always remind users that your answers are for informational purposes only
  and do not replace professional medical advice.

Context retrieved from medical knowledge base:
{context}
"""

def _format_docs(docs: list[Document]) -> str:
    """
    Formats retrieved chunks into a readable context block.
    Labels each chunk with its source file and page number.
    """
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        sections.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    """
    LLM: llama-3.1-8b-instant on Groq
      - temperature=0
    """
    retriever = build_retriever(k=5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  "{question}"),
    ])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",   
        temperature=0,                 
        max_tokens=1024,                 
        max_retries=2,
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
    question = "What are the common symptoms of hypertension?"
    print(f"Q: {question}\n")
    print("A: ", end="", flush=True)
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print()