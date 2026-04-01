import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_LLM_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            raise ValueError("[LLM] GEMINI_API_KEY is not set. Add it to your .env file.")
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(model_name=GEMINI_LLM_MODEL)
        print(f"[LLM] Gemini ready: {GEMINI_LLM_MODEL}")
    return _model


def build_prompt(question: str, chunks: list[str]) -> str:
    """
    Build a grounded prompt from retrieved resume chunks + the user question.
    """
    system = (
        "You are a helpful assistant that answers questions strictly based on "
        "the resume context provided below. "
        "If the answer is not found in the context, say so honestly. "
        "Do not make up information."
    )

    if chunks:
        context_block = "\n\n---\n".join(chunks)
        context_section = f"\n\n=== Resume Context ===\n{context_block}\n====================="
    else:
        context_section = "\n\n[No relevant context found in resume.]"

    return f"{system}{context_section}\n\nQuestion: {question}\nAnswer:"


def ask(question: str, chunks: list[str]) -> str:
    """Send the grounded prompt to Gemini and return the response."""
    prompt = build_prompt(question, chunks)
    return get_model().generate_content(prompt).text.strip()
