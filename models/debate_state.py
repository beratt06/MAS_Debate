"""Router ve research agent akisi icin durum modeli."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DebateState:
    """sistemin o anki sonucunu ve tasinan verileri tutan veri yapisidir"""

    question: str
    research_summary: str = ""
    facts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
