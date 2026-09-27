import os
from dataclasses import asdict
from typing import List, Tuple

import chromadb
import numpy as np

from app import config
from app.services.chunking import Chunk

COLLECTION_NAME = "nusantaracare_chunks"


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


class VectorStore:
    def __init__(self, index_dir=None):
        self.index_dir = index_dir or config.INDEX_DIR
        self.client = None
        self.collection = None
        self.chunks = []

    def _get_collection(self):
        if self.collection is None:
            os.makedirs(self.index_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.index_dir)
            
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self.collection

    def build(self, chunks, embeddings):
        self.chunks = chunks
        matrix = np.array(embeddings, dtype="float32")
        matrix = normalize(matrix)

        collection = self._get_collection()
        
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])

        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=matrix.tolist(),
            metadatas=[self._chunk_to_metadata(c) for c in chunks],
            documents=[c.text for c in chunks],
        )

    @property
    def index(self):
        return self.collection if self.chunks else None

    def save(self):
        pass

    def load(self):
        collection = self._get_collection()
        existing = collection.get(include=["metadatas", "documents"])
        ids = existing.get("ids") or []
        if not ids:
            return False

        self.chunks = [
            self._metadata_to_chunk(chunk_id, metadata, document)
            for chunk_id, metadata, document in zip(
                ids, existing["metadatas"], existing["documents"]
            )
        ]
        return True

    def search(self, query_embedding, top_k=None, active_only=True):
        top_k = top_k or config.TOP_K
        collection = self._get_collection()

        count = collection.count()
        if count == 0:
            return []

        query = np.array([query_embedding], dtype="float32")
        query = normalize(query)

        fetch_k = min(count, max(top_k * 4, top_k))
        raw = collection.query(
            query_embeddings=query.tolist(),
            n_results=fetch_k,
            include=["metadatas", "documents", "distances"],
        )

        ids = raw["ids"][0]
        metadatas = raw["metadatas"][0]
        documents = raw["documents"][0]
        distances = raw["distances"][0]

        candidates = []
        for chunk_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
            chunk = self._metadata_to_chunk(chunk_id, metadata, document)
            
            score = 1.0 - float(distance)
            candidates.append((chunk, score))

        results = []
        for chunk, score in candidates:
            if active_only and not chunk.is_active:
                continue
            results.append((chunk, score))
            if len(results) >= top_k:
                break

        if not results:
            results = candidates[:top_k]

        return results

    @staticmethod
    def _chunk_to_metadata(chunk: Chunk):
        data = asdict(chunk)
        data.pop("text", None)
        
        metadata = data.pop("metadata", {}) or {}
        flat = {k: v for k, v in data.items()}
        for k, v in metadata.items():
            flat[f"meta_{k}"] = v
        return flat

    @staticmethod
    def _metadata_to_chunk(chunk_id, metadata, document) -> Chunk:
        metadata = dict(metadata or {})
        nested = {}
        for key in list(metadata.keys()):
            if key.startswith("meta_"):
                nested[key[len("meta_"):]] = metadata.pop(key)
        return Chunk(
            chunk_id=chunk_id,
            text=document,
            doc_id=metadata.get("doc_id", ""),
            doc_title=metadata.get("doc_title", ""),
            doc_version=metadata.get("doc_version", ""),
            is_active=bool(metadata.get("is_active", True)),
            section_title=metadata.get("section_title", ""),
            metadata=nested,
            line_start=int(metadata.get("line_start", 0) or 0),
            line_end=int(metadata.get("line_end", 0) or 0),
        )


vector_store = VectorStore()
