"""Yerel retrieval hatti icin Ollama tabanli embedding araclari."""

from ollama import Client

from config import EMBEDDING_MODEL_NAME


class EmbeddingService:
    """Chroma ile uyumlu embeddingler icin kucuk bir Ollama sarmalayicisi."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, host: str | None = None) -> None:
        """Ollama istemcisini ve embedding model adini baslatir."""

        self.model_name = model_name
        self.client = Client(host=host) if host else Client()

    def embed_query(self, query: str) -> list[float]:
        """Tek bir sorgu metnini embeddinge donusturur."""

        response = self.client.embed(model=self.model_name, input=query)
        return self._extract_single_embedding(response["embeddings"])

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Bir metin listesini embeddinglere donusturur."""

        if not texts:
            return []

        response = self.client.embed(model=self.model_name, input=texts)
        return [list(embedding) for embedding in response["embeddings"]]

    def _extract_single_embedding(self, embeddings: list[list[float]] | list[float]) -> list[float]:
        """Tek metinli embedding yanitlarini duz bir listeye normalize eder."""

        if not embeddings:
            return []

        first_item = embeddings[0]
        if isinstance(first_item, list):
            return list(first_item)

        return list(embeddings)
