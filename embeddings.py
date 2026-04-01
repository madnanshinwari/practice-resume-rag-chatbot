from config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL

# Track which backend is active
_backend      = None    # "gemini" | "huggingface"
_hf_model     = None
_genai_client = None


def _get_genai_client():
    """Initialise the new google-genai client (once)."""
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")
            _genai_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            raise RuntimeError(f"[Embeddings] Failed to init Gemini client: {e}")
    return _genai_client


def _get_hf_model():
    """Load HuggingFace SentenceTransformer model (once)."""
    global _hf_model
    if _hf_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embeddings] Loading HuggingFace fallback: {FALLBACK_EMBEDDING_MODEL}")
        _hf_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
        print("[Embeddings] HuggingFace model loaded.")
    return _hf_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Tries Gemini first; falls back to HuggingFace on any error.
    """
    global _backend

    if _backend != "huggingface":
        try:
            client  = _get_genai_client()
            vectors = []
            for text in texts:
                result = client.models.embed_content(
                    model    = GEMINI_EMBEDDING_MODEL,
                    contents = text,
                )
                vectors.append(result.embeddings[0].values)
            _backend = "gemini"
            return vectors
        except Exception as e:
            print(f"[Embeddings] Gemini failed ({e}). Switching to HuggingFace fallback.")
            _backend = "huggingface"

    # HuggingFace fallback
    return _get_hf_model().encode(texts, convert_to_numpy=True).tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Tries Gemini first; falls back to HuggingFace on any error.
    """
    global _backend

    if _backend != "huggingface":
        try:
            client = _get_genai_client()
            result = client.models.embed_content(
                model    = GEMINI_EMBEDDING_MODEL,
                contents = query,
            )
            _backend = "gemini"
            return result.embeddings[0].values
        except Exception as e:
            print(f"[Embeddings] Gemini query embed failed ({e}). Using HuggingFace fallback.")
            _backend = "huggingface"

    # HuggingFace fallback
    return _get_hf_model().encode(query, convert_to_numpy=True).tolist()


def active_backend() -> str:
    return _backend or "not initialised yet"