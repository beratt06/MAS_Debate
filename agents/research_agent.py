"""PDF baglamina dayali Turkce arastirma uretimi yapan agent."""

import json
from typing import Any, TypedDict

from ollama import Client

from config import OLLAMA_MODEL_NAME, RETRIEVAL_TOP_K
from retrieval.retriever import ChromaRetriever, RetrievalResult


class SourceRecord(TypedDict):
    """Kaynak bilgisinin kararli semasi."""

    document: str
    page: int | None
    chunk_id: str


class ResearchOutput(TypedDict):
    """Arastirma agentinin dondurdugu cikti semasi."""

    topic_summary: str
    key_facts: list[str]
    sources: list[SourceRecord]
    retrieved_context: list[str]


SYSTEM_PROMPT = """
Sen PDF tabanli bir arastirma asistanisin.
Yalnizca sana verilen retrieval baglamini kullan.
Baglam disinda hicbir iddia, yorum veya ek bilgi uretme.
Eger baglam bir bilgi icermiyorsa bunu acikca belirt.
Her zaman Turkce yaz.
Kisa, net ve resmi bir Turkce kullan.
topic_summary alani kisa, tarafsiz ve ozetleyici olsun.
key_facts alani sadece kisa, dogrudan ve olgusal cumlelerden olussun.
key_facts icinde "Parca 1", "Parca 2", "chunk", "bolum", "baglam", "yukaridaki metin" gibi atiflar kullanma.
Metinde gecmeyen hicbir bilgiyi ekleme ve desteklenmeyen sonuc cikarma.
Yanitini yalnizca gecerli JSON olarak ver.
JSON semasi:
{
  "topic_summary": "string",
  "key_facts": ["string", "string"]
}
""".strip()


class ResearchAgent:
    """Retriever sonucunu kullanarak Turkce arastirma cikti ureten agent."""

    def __init__(
        self,
        retriever: ChromaRetriever | None = None,
        model_name: str = OLLAMA_MODEL_NAME,
        host: str | None = None,
    ) -> None:
        """Retriever ve Ollama istemcisini baslatir."""

        self.retriever = retriever or ChromaRetriever()
        self.model_name = model_name
        self.client = Client(host=host) if host else Client()

    def research(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> ResearchOutput:
        """Soru icin retrieval yapar ve PDF baglamina dayali arastirma ciktisi uretir."""

        cleaned_question = question.strip()
        if not cleaned_question:
            return self._empty_output("Arastirma icin gecerli bir soru saglanmadi.")

        retrieved_results = self.retriever.retrieve(cleaned_question, top_k=top_k)
        if not retrieved_results:
            return self._empty_output(
                "Ilgili PDF baglami bulunamadi; bu nedenle arastirma ozeti uretilemedi."
            )

        prompt = self._build_prompt(cleaned_question, retrieved_results)
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
        )

        parsed_response = self._parse_model_response(
            response.get("message", {}).get("content", "")
        )

        topic_summary = self._safe_summary(parsed_response.get("topic_summary"))
        key_facts = self._safe_key_facts(parsed_response.get("key_facts"))

        return {
            "topic_summary": topic_summary,
            "key_facts": key_facts,
            "sources": self._build_sources(retrieved_results),
            "retrieved_context": self._build_retrieved_context(retrieved_results),
        }

    def _build_prompt(self, question: str, retrieved_results: list[RetrievalResult]) -> str:
        """Model icin kullanilacak baglamli istemi olusturur."""

        context_blocks: list[str] = []

        for index, result in enumerate(retrieved_results, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"Parca {index}",
                        f"Belge: {result['document'] or 'Bilinmiyor'}",
                        f"Sayfa: {result['page'] if result['page'] is not None else 'Bilinmiyor'}",
                        f"Parca Kimligi: {result['chunk_id'] or 'Bilinmiyor'}",
                        f"Mesafe: {result['distance'] if result['distance'] is not None else 'Bilinmiyor'}",
                        "Metin:",
                        result["text"],
                    ]
                )
            )

        context_text = "\n\n".join(context_blocks)

        return (
            f"Soru:\n{question}\n\n"
            "Yalnizca asagidaki retrieval baglamini kullan.\n"
            "Baglam yeterli degilse bunu topic_summary icinde acikca belirt.\n"
            "topic_summary alaninda kisa ve tarafsiz bir ozet yaz.\n"
            "key_facts alaninda sadece baglamdan dogrudan desteklenen kisa, acik ve olgusal maddeler yaz.\n"
            "key_facts icinde Parca numarasi, chunk etiketi veya metin parcasi referansi kullanma.\n"
            "Desteklenmeyen hicbir iddia ekleme.\n\n"
            f"Baglam:\n{context_text}"
        )

    def _parse_model_response(self, content: str) -> dict[str, Any]:
        """Model yanitini guvenli sekilde JSON olarak ayrisir."""

        cleaned_content = content.strip()
        if not cleaned_content:
            return {}

        try:
            parsed = json.loads(cleaned_content)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            start = cleaned_content.find("{")
            end = cleaned_content.rfind("}")
            if start == -1 or end == -1 or start >= end:
                return {}

            try:
                parsed = json.loads(cleaned_content[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

    def _safe_summary(self, value: Any) -> str:
        """Ozet alanini guvenli bir dizeye donusturur."""

        if isinstance(value, str) and value.strip():
            return value.strip()
        return "Saglanan PDF baglamina gore kisa bir arastirma ozeti uretilemedi."

    def _safe_key_facts(self, value: Any) -> list[str]:
        """Olgu listesini guvenli ve temiz bicime donusturur."""

        if not isinstance(value, list):
            return []

        facts: list[str] = []
        banned_fragments = [
            "parca ",
            "parca:",
            "chunk ",
            "chunk:",
            "bolum ",
            "bolum:",
        ]
        for item in value:
            if isinstance(item, str):
                cleaned_item = item.strip()
                normalized_item = cleaned_item.casefold()
                if cleaned_item and not any(
                    fragment in normalized_item for fragment in banned_fragments
                ):
                    facts.append(cleaned_item)

        return facts

    def _build_sources(self, retrieved_results: list[RetrievalResult]) -> list[SourceRecord]:
        """Retrieval sonucundan tekrar etmeyen kaynak listesini olusturur."""

        sources: list[SourceRecord] = []
        seen: set[tuple[str, int | None, str]] = set()

        for result in retrieved_results:
            source_key = (result["document"], result["page"], result["chunk_id"])
            if source_key in seen:
                continue

            seen.add(source_key)
            sources.append(
                {
                    "document": result["document"],
                    "page": result["page"],
                    "chunk_id": result["chunk_id"],
                }
            )

        return sources

    def _build_retrieved_context(
        self, retrieved_results: list[RetrievalResult]
    ) -> list[str]:
        """Retrieval sonucundaki metinleri sirali baglam listesine donusturur."""

        return [result["text"] for result in retrieved_results if result["text"]]

    def _empty_output(self, summary: str) -> ResearchOutput:
        """Bos sonuc durumlari icin guvenli cikti dondurur."""

        return {
            "topic_summary": summary,
            "key_facts": [],
            "sources": [],
            "retrieved_context": [],
        }
