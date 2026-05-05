"""
ChromaDB Store
==============
Handles all vector storage and retrieval operations.
Stores embeddings with metadata for source filtering.
"""

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chromadb")
COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION", "knowledge_os")

_client = None
_collection = None


def get_client():
    """Get or create ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMADB_PATH)
        print(f"✅ ChromaDB connected at: {CHROMADB_PATH}")
    return _client


def get_collection():
    """Get or create the knowledge OS collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection: {COLLECTION_NAME} ({_collection.count()} chunks)")
    return _collection


def search(query_embedding: list[float], n_results: int = 5, source_filter: str = None) -> list[dict]:
    """
    Search ChromaDB for similar chunks.

    Args:
        query_embedding: The embedding vector to search with
        n_results: Number of results to return
        source_filter: Optional filter by source ('pdf', 'notion', 'obsidian', 'gmail')

    Returns:
        List of dicts with 'document', 'metadata', and 'distance'
    """
    collection = get_collection()

    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "document": doc,
            "metadata": meta,
            "distance": dist,
            "relevance": round(1 - dist, 3)
        })

    return chunks


def get_stats() -> dict:
    """Get collection statistics."""
    collection = get_collection()
    count = collection.count()

    # Count by source
    sources = {}
    if count > 0:
        all_meta = collection.get(include=["metadatas"])
        for meta in all_meta["metadatas"]:
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

    return {
        "total_chunks": count,
        "by_source": sources,
        "collection": COLLECTION_NAME,
        "path": CHROMADB_PATH,
    }


def delete_by_source(source: str):
    """Delete all chunks from a specific source (for re-ingestion)."""
    collection = get_collection()
    collection.delete(where={"source": source})
    print(f"🗑️  Deleted all chunks from source: {source}")


if __name__ == "__main__":
    stats = get_stats()
    print(f"\n📊 ChromaDB Stats")
    print(f"=================")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"By source:")
    for src, count in stats["by_source"].items():
        print(f"  {src}: {count} chunks")
