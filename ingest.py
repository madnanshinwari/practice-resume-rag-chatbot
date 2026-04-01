import os
import uuid
import chromadb
from chromadb import PersistentClient

from config import (
    RESUME_PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
)
from embeddings import embed_texts


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Resume PDF not found at '{pdf_path}'. "
            "Drop your resume.pdf in the project root."
        )

    reader = PdfReader(pdf_path)
    pages  = [page.extract_text() or "" for page in reader.pages]
    text   = "\n".join(pages).strip()

    if not text:
        raise ValueError("No text extracted from PDF. Make sure it is not a scanned image.")

    print(f"[Ingest] Extracted {len(text)} characters from {len(reader.pages)} page(s).")
    return text


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character-level chunks.
    Tries to split on newlines to keep sentences intact.
    """
    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size

        # Prefer splitting at a newline or space near the boundary
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at == -1:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap   # step back by overlap for context continuity

    print(f"[Ingest] Created {len(chunks)} chunks "
          f"(size≈{chunk_size} chars, overlap={overlap}).")
    return chunks


# ── ChromaDB Storage ──────────────────────────────────────────────────────────

def get_collection(reset: bool = False):
    """Return (and optionally reset) the ChromaDB collection."""
    client = PersistentClient(path=CHROMA_PERSIST_DIR)

    if reset and CHROMA_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"[Ingest] Existing collection '{CHROMA_COLLECTION_NAME}' deleted.")

    collection = client.get_or_create_collection(
        name     = CHROMA_COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(chunks: list[str], collection) -> None:
    """Embed chunks and upsert them into ChromaDB."""
    print(f"[Ingest] Embedding {len(chunks)} chunks...")
    vectors = embed_texts(chunks)

    ids       = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids        = ids,
        embeddings = vectors,
        documents  = chunks,
        metadatas  = metadatas,
    )
    print(f"[Ingest] Stored {len(chunks)} chunks in ChromaDB.")


# ── Main ingest entry point ───────────────────────────────────────────────────

def ingest(pdf_path: str = RESUME_PDF_PATH, reset: bool = True) -> None:
    """
    Full ingestion pipeline:
      1. Extract text from PDF
      2. Chunk the text
      3. Embed chunks
      4. Store in ChromaDB
    """
    print(f"\n{'='*50}")
    print(f"[Ingest] Starting ingestion: {pdf_path}")
    print(f"{'='*50}")

    text       = extract_text_from_pdf(pdf_path)
    chunks     = chunk_text(text)
    collection = get_collection(reset=reset)
    store_chunks(chunks, collection)

    print(f"[Ingest] ✅ Done. Collection '{CHROMA_COLLECTION_NAME}' "
          f"now has {collection.count()} chunk(s).\n")


if __name__ == "__main__":
    ingest()
