"""Yerel Chroma koleksiyonundan ilgili PDF parcalarini getiren retriever."""

from typing import Any, TypedDict

from config import RETRIEVAL_TOP_K
from retrieval.embeddings import EmbeddingService
from retrieval.vector_store import ChromaVectorStore


class RetrievalResult(TypedDict):
    """Retriever sonucunun kararli ve acik semasi."""

    text: str
    document: str
    page: int | None
    chunk_id: str
    distance: float | None


class ChromaRetriever:
    """Turkce sorgular icin Chroma tabanli parca getirici."""

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Retriever bagimliliklarini baslatir."""

        self.vector_store = vector_store or ChromaVectorStore(
            embedding_service=embedding_service
        )
        self.embedding_service = embedding_service or self.vector_store.embedding_service

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[RetrievalResult]:
        """Verilen sorgu icin en ilgili parcalari dondurur."""

        if not query.strip() or top_k <= 0:
            return []

        collection = self.vector_store.get_collection()
        if collection.count() == 0:
            return []

        query_embedding = self.embedding_service.embed_query(query)
        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._build_results(raw_results)

    def _build_results(self, raw_results: dict[str, Any]) -> list[RetrievalResult]:
        """Ham Chroma sonucunu kararli retrieval semasina donusturur."""

        ids = self._first_list(raw_results.get("ids"))
        documents = self._first_list(raw_results.get("documents"))
        metadatas = self._first_list(raw_results.get("metadatas"))
        distances = self._first_list(raw_results.get("distances"))

        if not ids:
            return []

        results: list[RetrievalResult] = []

        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document_text = documents[index] if index < len(documents) and documents[index] else ""
            distance = distances[index] if index < len(distances) else None

            page_value = metadata.get("page")
            page = page_value if isinstance(page_value, int) else None

            results.append(
                {
                    "text": document_text,
                    "document": self._safe_string(metadata.get("document")),
                    "page": page,
                    "chunk_id": self._safe_string(metadata.get("chunk_id")) or chunk_id,
                    "distance": self._safe_float(distance),
                }
            )

        return results

    def _first_list(self, value: Any) -> list[Any]:
        """Chroma'nin katmanli sonucundan ilk listeyi guvenli sekilde alir."""

        if isinstance(value, list) and value:
            first_item = value[0]
            if isinstance(first_item, list):
                return first_item
        return []

    def _safe_string(self, value: Any) -> str:
        """Metin alanlarini guvenli sekilde dizeye cevirir."""

        return value if isinstance(value, str) else ""

    def _safe_float(self, value: Any) -> float | None:
        """Mesafe degerini mumkunse float olarak dondurur."""

        if isinstance(value, (int, float)):
            return float(value)
        return None
