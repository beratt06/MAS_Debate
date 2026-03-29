"""PDF parcalarini indekslemek icin ChromaDB tabanli vector store araclari."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from config import CHROMA_PERSIST_DIR
from retrieval.chunker import ChunkRecord
from retrieval.embeddings import EmbeddingService


COLLECTION_NAME = "pdf_chunks"
INDEX_BATCH_SIZE = 100


class ChromaVectorStore:
    """Parca kayitlari icin kalici vector store."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Kalici Chroma istemcisini ve koleksiyon ayarlarini baslatir."""

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_service = embedding_service or EmbeddingService()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))

    def get_collection(self) -> Collection:
        """Parca koleksiyonunu olusturur veya mevcut olani dondurur."""

        return self.client.get_or_create_collection(name=self.collection_name)

    def index_chunks(self, chunks: list[ChunkRecord]) -> int:
        """Parca kayitlarini indeksler ve zaten var olan kayitlari atlar."""

        if not chunks:
            return 0

        collection = self.get_collection()
        new_chunks = self._filter_new_chunks(collection, chunks)
        if not new_chunks:
            return 0

        for start in range(0, len(new_chunks), INDEX_BATCH_SIZE):
            batch = new_chunks[start : start + INDEX_BATCH_SIZE]
            collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[self._build_metadata(chunk) for chunk in batch],
                embeddings=self.embedding_service.embed_texts(
                    [chunk["text"] for chunk in batch]
                ),
            )

        return len(new_chunks)

    def _filter_new_chunks(
        self, collection: Collection, chunks: list[ChunkRecord]
    ) -> list[ChunkRecord]:
        """Yalnizca kimligi henuz kayitli olmayan parcalari dondurur."""

        new_chunks: list[ChunkRecord] = []

        for start in range(0, len(chunks), INDEX_BATCH_SIZE):
            batch = chunks[start : start + INDEX_BATCH_SIZE]
            batch_ids = [chunk["chunk_id"] for chunk in batch]
            existing = collection.get(ids=batch_ids)
            existing_ids = set(existing.get("ids", []))

            for chunk in batch:
                if chunk["chunk_id"] not in existing_ids:
                    new_chunks.append(chunk)

        return new_chunks

    def _build_metadata(self, chunk: ChunkRecord) -> dict[str, Any]:
        """Her parcayla birlikte saklanacak metaveri yukunu olusturur."""

        return {
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
        }
