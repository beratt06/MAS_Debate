""" soru geldikten sonra hangi agent’in calisacagini belirleyen ince kontrol katmani. Su an mimari basit oldugu icin sadece ResearchAgent cagiriyor ve sonucu DebateState icine yerlestiriyor. Bunu kullanmamizin nedeni, ileride farkli agent’ler eklense bile giris noktasi ayni kalsin."""

from agents.research_agent import ResearchAgent
from models.debate_state import DebateState


class DebateRouter:
    """Soru alip DebateState olusturan ve arastirma sonucuyla dolduran router."""

    def __init__(self, research_agent: ResearchAgent | None = None) -> None:
        """Router bagimliliklarini baslatir."""

        self.research_agent = research_agent or ResearchAgent()

    def route(self, question: str) -> DebateState:
        """Soruyu isler ve doldurulmus DebateState dondurur."""

        cleaned_question = question.strip()
        state = DebateState(question=cleaned_question)

        if not cleaned_question:
            state.research_summary = "Lutfen gecerli bir soru giriniz."
            return state

        research_output = self.research_agent.research(cleaned_question)
        state.research_summary = research_output["topic_summary"]
        state.facts = research_output["key_facts"]
        state.sources = research_output["sources"]
        state.retrieved_context = research_output["retrieved_context"]

        return state
