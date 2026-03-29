"""
Debate Loop — Orchestrates multi-round debates between Pro and Contra agents.

This module implements the DebateLoop class which:
1. Maintains an Argument Memory (chronological list of all debate entries).
2. Runs a configurable number of debate rounds (default: 3).
3. In each round, calls Pro Agent then Contra Agent with full history context.
4. Produces a structured DebateTranscript for downstream consumption
   (Evidence Verifier, Judge Agent).
"""

from __future__ import annotations

import json
from typing import Optional

from models import (
    ResearchInput,
    ProAgentOutput,
    ContraAgentOutput,
    DebateEntry,
    DebateRound,
    DebateTranscript,
)
from pro_agent import ProAgent
from contra_agent import ContraAgent


class DebateLoop:
    """Orchestrates a multi-round debate between a Pro Agent and a Contra Agent.

    The debate proceeds in rounds. In each round:
        1. The Pro Agent presents (or defends) arguments, aware of all prior history.
        2. The Contra Agent critiques those arguments, also aware of all prior history.

    All entries are appended to a shared **Argument Memory** so that subsequent
    rounds can reference and build upon earlier exchanges.

    Args:
        pro_agent: An initialized ``ProAgent`` instance.
        contra_agent: An initialized ``ContraAgent`` instance.
        max_rounds: Maximum number of debate rounds (default: ``3``).
    """

    def __init__(
        self,
        pro_agent: ProAgent,
        contra_agent: ContraAgent,
        max_rounds: int = 3,
    ) -> None:
        self.pro_agent = pro_agent
        self.contra_agent = contra_agent
        self.max_rounds = max_rounds

        # ── Argument Memory ───────────────────────────────────────────
        self.memory: list[DebateEntry] = []
        self._rounds: list[DebateRound] = []

    # ── Main Debate Loop ──────────────────────────────────────────────
    def run(
        self,
        research: ResearchInput,
        on_round_complete: Optional[callable] = None,
    ) -> DebateTranscript:
        """Execute the full debate loop.

        Args:
            research: Validated ``ResearchInput`` from the Research Agent.
            on_round_complete: Optional callback ``(round_number, pro_output, contra_output)``
                invoked after each round completes, useful for logging or streaming.

        Returns:
            A ``DebateTranscript`` containing the complete, structured debate history.
        """
        for round_num in range(1, self.max_rounds + 1):
            # ── Pro Agent's turn ──────────────────────────────────────
            # Round 1: no history → fresh arguments
            # Round 2+: full history → defend + rebut + new arguments
            history_for_pro = list(self.memory) if self.memory else None

            pro_output: ProAgentOutput = self.pro_agent.run(
                research=research,
                debate_history=history_for_pro,
            )

            pro_entry = DebateEntry(
                agent="PRO",
                round_number=round_num,
                content=pro_output.model_dump(),
            )
            self.memory.append(pro_entry)

            # ── Contra Agent's turn ───────────────────────────────────
            # Always has at least the current round's pro entry in history
            history_for_contra = list(self.memory)

            contra_output: ContraAgentOutput = self.contra_agent.run(
                research=research,
                pro_output=pro_output,
                debate_history=history_for_contra,
            )

            contra_entry = DebateEntry(
                agent="CONTRA",
                round_number=round_num,
                content=contra_output.model_dump(),
            )
            self.memory.append(contra_entry)

            # ── Record the round ──────────────────────────────────────
            debate_round = DebateRound(
                round_number=round_num,
                pro_entry=pro_entry,
                contra_entry=contra_entry,
            )
            self._rounds.append(debate_round)

            # ── Optional callback ─────────────────────────────────────
            if on_round_complete:
                on_round_complete(round_num, pro_output, contra_output)

        return self.get_transcript(research.topic_summary)

    # ── Transcript Builder ────────────────────────────────────────────
    def get_transcript(self, topic: str) -> DebateTranscript:
        """Build a ``DebateTranscript`` from the current memory state.

        Args:
            topic: The topic string for the transcript header.

        Returns:
            A validated ``DebateTranscript`` instance.
        """
        return DebateTranscript(
            topic=topic,
            total_rounds=len(self._rounds),
            rounds=self._rounds,
            full_history=list(self.memory),
        )

    # ── JSON Export ───────────────────────────────────────────────────
    def export_json(self, topic: str, indent: int = 2) -> str:
        """Export the full debate transcript as a formatted JSON string.

        This output is designed to be consumed by the Evidence Verifier
        and Judge Agent modules downstream.

        Args:
            topic: The topic string for the transcript header.
            indent: JSON indentation level (default: 2).

        Returns:
            A pretty-printed JSON string of the debate transcript.
        """
        transcript = self.get_transcript(topic)
        return transcript.model_dump_json(indent=indent)

    # ── State Inspection ──────────────────────────────────────────────
    @property
    def current_round(self) -> int:
        """Return the number of completed rounds."""
        return len(self._rounds)

    @property
    def history_size(self) -> int:
        """Return the total number of entries in Argument Memory."""
        return len(self.memory)

    def reset(self) -> None:
        """Clear the Argument Memory and round history for a fresh debate."""
        self.memory.clear()
        self._rounds.clear()

    def __repr__(self) -> str:
        return (
            f"DebateLoop(max_rounds={self.max_rounds}, "
            f"completed={self.current_round}, "
            f"memory_size={self.history_size})"
        )
