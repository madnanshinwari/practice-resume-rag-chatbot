from retriever import retrieve
from llm import ask


def chat(question: str) -> str:
    """
    Resume Q&A pipeline:
      1. Retrieve relevant resume chunks from ChromaDB
      2. Build a grounded prompt and ask Gemini
    """
    chunks   = retrieve(question)
    response = ask(question, chunks)
    return response
