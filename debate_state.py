"""Router and research flow state container."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DebateState:
    """Holds mutable state for question routing and research output."""

    question: str
    research_summary: str = ""
    facts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
