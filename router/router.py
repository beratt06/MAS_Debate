""" soru geldikten sonra hangi agent’in calisacagini belirleyen ince kontrol katmani. Su an mimari basit oldugu icin sadece ResearchAgent cagiriyor ve sonucu DebateState icine yerlestiriyor. Bunu kullanmamizin nedeni, ileride farkli agent’ler eklense bile giris noktasi ayni kalsin."""

from agents.research_agent import ResearchAgent
from config import RETRIEVAL_TOP_K
from debate_state import DebateState


class DebateRouter:
    """Soru alip DebateState olusturan ve arastirma sonucuyla dolduran router."""

    def __init__(self, research_agent: ResearchAgent | None = None) -> None:
        """Router bagimliliklarini baslatir."""

        self.research_agent = research_agent or ResearchAgent()

    def route(self, question: str, top_k: int | None = None) -> DebateState:
        """Soruyu isler ve doldurulmus DebateState dondurur."""

        cleaned_question = question.strip()
        state = DebateState(question=cleaned_question)

        if not cleaned_question:
            state.research_summary = "Lutfen gecerli bir soru giriniz."
            return state

        resolved_top_k = top_k if isinstance(top_k, int) and top_k > 0 else RETRIEVAL_TOP_K
        research_output = self.research_agent.research(cleaned_question, top_k=resolved_top_k)
        state.research_summary = research_output["topic_summary"]
        state.facts = research_output["key_facts"]
        state.sources = research_output["sources"]
        state.retrieved_context = research_output["retrieved_context"]

        return state
