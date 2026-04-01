import chromadb
from chromadb import PersistentClient

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K
from embeddings import embed_query

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client      = PersistentClient(path=CHROMA_PERSIST_DIR)
        collections = [c.name for c in client.list_collections()]

        if CHROMA_COLLECTION_NAME not in collections:
            raise RuntimeError(
                f"Collection '{CHROMA_COLLECTION_NAME}' not found. "
                "Run ingest.py first to index your resume."
            )

        _collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    return _collection


def retrieve(query: str, n_results: int = TOP_K) -> list[str]:
    """
    Embed the query and return the top-n most relevant resume chunks.
    """
    col = get_collection()

    if col.count() == 0:
        return []

    vector  = embed_query(query)
    results = col.query(
        query_embeddings = [vector],
        n_results        = min(n_results, col.count()),
        include          = ["documents"],
    )
    return results["documents"][0]
