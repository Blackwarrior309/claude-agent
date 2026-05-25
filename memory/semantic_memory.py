"""
Semantic vector memory using ChromaDB for similarity search.
Falls back to keyword search if ChromaDB is unavailable.
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SemanticMemory:
    def __init__(self, persist_dir: str = "data/embeddings"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._fallback_store: list[dict] = []
        self._init()

    def _init(self):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name="business_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("SemanticMemory: ChromaDB initialized")
        except Exception as e:
            logger.warning(f"SemanticMemory: ChromaDB unavailable ({e}), using keyword fallback")

    def add(self, doc_id: str, text: str, metadata: dict | None = None):
        meta = metadata or {}
        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[meta],
                )
                return
            except Exception as e:
                logger.warning(f"SemanticMemory add failed: {e}")
        self._fallback_store = [e for e in self._fallback_store if e["id"] != doc_id]
        self._fallback_store.append({"id": doc_id, "text": text, "metadata": meta})

    def search(self, query: str, n: int = 5) -> list[dict]:
        if self._collection is not None:
            try:
                results = self._collection.query(query_texts=[query], n_results=min(n, max(1, self._collection.count())))
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                return [
                    {"text": d, "metadata": m, "score": round(1 - dist, 4)}
                    for d, m, dist in zip(docs, metas, distances)
                ]
            except Exception as e:
                logger.warning(f"SemanticMemory search failed: {e}")

        # keyword fallback
        query_words = set(query.lower().split())
        scored = []
        for entry in self._fallback_store:
            text_words = set(entry["text"].lower().split())
            overlap = len(query_words & text_words)
            if overlap:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": e["text"], "metadata": e["metadata"], "score": s} for s, e in scored[:n]]

    def count(self) -> int:
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._fallback_store)
