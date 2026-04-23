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


# 1. ONLY cache the core ChromaDB client to prevent connection crashes. 
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


# 2. DO NOT cache this function. LangChain will safely wrap the cached client.
def get_vector_store() -> Chroma:
    """Returns the LangChain wrapper around the persistent ChromaDB client."""
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
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
    """Wipes the ChromaDB collection without breaking the connection."""
    client = get_chroma_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass


def delete_source(filename: str) -> None:
    """Deletes all chunks associated with a specific filename from ChromaDB."""
    client = get_chroma_client()
    try:
        # Get the collection and delete chunks where the source matches the filename
        collection = client.get_collection(CHROMA_COLLECTION)
        collection.delete(where={"source": filename})
    except Exception:
        pass
