"""
core/vector_store.py
Manages the ChromaDB vector store: adding documents, similarity search,
and Cohere reranking for higher retrieval quality.
"""

from typing import List, Optional
import cohere
import chromadb
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    COHERE_API_KEY,
    TOP_K_RETRIEVAL,
    TOP_K_RERANKED,
)
from core.embeddings import get_embeddings



@st.cache_resource
def get_vector_store() -> Chroma:
    """Returns (or creates) the persistent ChromaDB vector store."""
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_documents(docs: List[Document]) -> int:
    """
    Adds a list of Document chunks to ChromaDB.
    Deduplicates by chunk_id so re-uploading the same PDF is safe.
    Returns the number of new chunks added.
    """
    store = get_vector_store()

    # Collect existing chunk IDs to avoid duplicates
    existing_ids: set = set()
    try:
        collection = store._collection
        existing = collection.get(include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass

    new_docs = [d for d in docs if d.metadata.get("chunk_id") not in existing_ids]
    if not new_docs:
        return 0

    ids = [d.metadata["chunk_id"] for d in new_docs]
    store.add_documents(new_docs, ids=ids)
    return len(new_docs)


def retrieve_and_rerank(query: str, k_fetch: int = TOP_K_RETRIEVAL, k_keep: int = TOP_K_RERANKED) -> List[Document]:
    """
    Two-stage retrieval:
      1. ChromaDB similarity search (k_fetch candidates)
      2. Cohere reranker picks the best k_keep

    Falls back to raw retrieval if Cohere key is missing.
    """
    store = get_vector_store()
    candidates = store.similarity_search(query, k=k_fetch)

    if not candidates:
        return []

    # ── Cohere reranking ─────────────────────────────────────────────────────
    if COHERE_API_KEY:
        try:
            co = cohere.Client(COHERE_API_KEY)
            response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=[d.page_content for d in candidates],
                top_n=k_keep,
            )
            reranked = [candidates[r.index] for r in response.results]
            return reranked
        except Exception:
            pass  # fallback to raw retrieval

    return candidates[:k_keep]


def list_indexed_sources() -> List[str]:
    """Returns filenames of all PDFs currently in the vector store."""
    store = get_vector_store()
    try:
        collection = store._collection
        data = collection.get(include=["metadatas"])
        sources = {m.get("source", "Unknown") for m in data.get("metadatas", [])}
        return sorted(sources)
    except Exception:
        return []


def clear_collection() -> None:
    """Wipes the entire ChromaDB collection. Use with care."""
    store = get_vector_store()
    try:
        store.delete_collection()
    except Exception:
        pass
    st.cache_resource.clear()
