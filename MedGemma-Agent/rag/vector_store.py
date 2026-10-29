"""
ChromaDB vector store interface.

One collection per patient: "ehr_{patient_id}"
PHI is NOT stored in ChromaDB — chunk text is de-identified before insertion.
The compliance/phi_handler.py layer handles de-identification upstream.
"""

from __future__ import annotations
from functools import lru_cache
import chromadb

from config.settings import get_settings
from rag.embedder import embed_texts, embed_query

settings = get_settings()


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
    )


def _collection_name(patient_id: str) -> str:
    # ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens
    safe = patient_id.replace("_", "-").replace(" ", "-")[:50]
    return f"ehr-{safe}"


def upsert_chunks(patient_id: str, chunks: list) -> None:
    """
    Embed and upsert EHR chunks into the patient's ChromaDB collection.
    `chunks` is a list of EHRChunk objects (from ehr_processor.py).
    Text must already be de-identified before calling this function.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=_collection_name(patient_id),
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"{patient_id}-chunk-{c.chunk_index}" for c in chunks]
    metadatas = [
        {
            "patient_id": patient_id,
            "chunk_index": c.chunk_index,
            "section": c.section or "unknown",
            "page": c.page or 0,
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


def query_patient_ehr(patient_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the top-k most relevant EHR chunks for a patient.
    Returns list of dicts with keys: text, section, page, distance.
    """
    client = get_chroma_client()
    name = _collection_name(patient_id)

    try:
        collection = client.get_collection(name=name)
    except Exception:
        return []  # No EHR ingested for this patient

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "section": meta.get("section", "unknown"),
            "page": meta.get("page", 0),
            "distance": round(dist, 4),
        })
    return chunks


def patient_ehr_exists(patient_id: str) -> bool:
    """Check whether EHR chunks exist for a patient."""
    client = get_chroma_client()
    try:
        col = client.get_collection(name=_collection_name(patient_id))
        return col.count() > 0
    except Exception:
        return False


def delete_patient_ehr(patient_id: str) -> None:
    """Remove all EHR data for a patient from ChromaDB."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=_collection_name(patient_id))
    except Exception:
        pass
