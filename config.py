import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Models ───────────────────────────────────────────────────────────────────
GEMINI_LLM_MODEL       = "gemini-2.5-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"            # Gemini embedding
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"             # HuggingFace fallback

# ── PDF ──────────────────────────────────────────────────────────────────────
RESUME_PDF_PATH = "./resume.pdf"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR      = "./chroma_db"
CHROMA_COLLECTION_NAME  = "resume"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 5   # number of chunks to retrieve per query