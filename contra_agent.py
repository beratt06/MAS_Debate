"""
Contra Agent — Challenges the Pro Agent's arguments in the AI Decision Debate System.

This module implements the ContraAgent class which:
1. Receives the original research data AND the Pro Agent's output.
2. Optionally receives debate history to build upon previous rounds.
3. Constructs a prompt that enforces a strictly critical stance.
4. Calls an LLM via LangChain and returns validated counter-arguments and risks.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from models import ResearchInput, ProAgentOutput, ContraAgentOutput, DebateEntry
from prompts import CONTRA_AGENT_SYSTEM_PROMPT


class ContraAgent:
    """Agent that generates structured counter-arguments and risks against the Pro Agent's position.

    Args:
        model_name: Ollama model identifier (default: ``gpt120bcloud``).
        temperature: Sampling temperature for the LLM (default: ``0.7``).
        base_url: Ollama server URL (default: ``http://localhost:11434``).
    """

    def __init__(
        self,
        model_name: str = "gpt120bcloud",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=base_url,
            format="json",
        )

    # ── History Formatting ────────────────────────────────────────────
    @staticmethod
    def _format_history(debate_history: list[DebateEntry]) -> str:
        """Format previous debate entries into a readable text block for the prompt."""
        if not debate_history:
            return ""

        sections: list[str] = []
        for entry in debate_history:
            content = entry.content
            agent_label = "🟢 PRO Agent" if entry.agent == "PRO" else "🔴 CONTRA Agent"
            header = f"### Round {entry.round_number} — {agent_label}"

            if entry.agent == "PRO":
                args_text = ""
                for i, arg in enumerate(content.get("arguments", []), 1):
                    args_text += f"  - **{arg['title']}**: {arg['explanation']}\n"
                body = f"{args_text}  Summary: {content.get('summary', '')}"
            else:  # CONTRA
                ca_text = ""
                for ca in content.get("counter_arguments", []):
                    ca_text += f"  - **[targets: {ca['target_argument']}]**: {ca['criticism']}\n"
                risk_text = ""
                for r in content.get("risks", []):
                    risk_text += f"  - **[{r['severity']}] {r['title']}**: {r['description']}\n"
                body = f"**Counter-Arguments:**\n{ca_text}**Risks:**\n{risk_text}  Summary: {content.get('summary', '')}"

            sections.append(f"{header}\n{body}")

        return "\n\n".join(sections)

    # ── Prompt Construction ───────────────────────────────────────────
    @staticmethod
    def build_prompt(
        research: ResearchInput,
        pro_output: ProAgentOutput,
        debate_history: Optional[list[DebateEntry]] = None,
    ) -> list:
        """Build the message list to send to the LLM.

        Args:
            research: Validated research data from the Research Agent.
            pro_output: The structured output produced by the Pro Agent (current round).
            debate_history: Optional list of previous debate entries for context.

        Returns:
            A list of LangChain ``BaseMessage`` instances (system + human).
        """
        # Format research facts
        facts_text = "\n".join(
            f"  - Fact {i}: {fact}" for i, fact in enumerate(research.facts, 1)
        )

        # Format current round's pro arguments
        pro_args_text = ""
        for i, arg in enumerate(pro_output.arguments, 1):
            supporting = (
                "\n".join(f"      • {f}" for f in arg.supporting_facts)
                if arg.supporting_facts
                else "      (none cited)"
            )
            pro_args_text += (
                f"  ### Argument {i}: {arg.title}\n"
                f"  **Explanation**: {arg.explanation}\n"
                f"  **Supporting Facts**:\n{supporting}\n\n"
            )

        human_content = (
            f"## Topic\n{research.topic_summary}\n\n"
            f"## Research Facts\n{facts_text}\n\n"
        )

        # Add debate history if available (round 2+)
        if debate_history:
            history_text = ContraAgent._format_history(debate_history)
            human_content += (
                f"## Previous Debate History\n{history_text}\n\n"
            )

        human_content += (
            f"## Pro Agent Arguments (Current Round)\n{pro_args_text}"
            f"## Pro Agent Summary\n{pro_output.summary}\n\n"
        )

        # Add round 2+ specific instructions
        if debate_history:
            human_content += (
                "IMPORTANT: The Pro Agent has responded to your previous criticisms. "
                "You MUST now:\n"
                "1. Identify if the Pro Agent successfully addressed your earlier criticisms or merely deflected them.\n"
                "2. Find NEW weaknesses in the Pro Agent's updated arguments.\n"
                "3. Escalate any risks that remain unaddressed from previous rounds.\n"
                "4. Directly reference the Pro Agent's rebuttals and explain why they are insufficient.\n\n"
            )

        human_content += (
            "Based on the topic, facts, debate history, and the Pro Agent's arguments above, "
            "generate your CONTRA counter-arguments and risks in the specified JSON format."
        )

        return [
            SystemMessage(content=CONTRA_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    # ── Main Execution ────────────────────────────────────────────────
    def run(
        self,
        research: ResearchInput,
        pro_output: ProAgentOutput,
        debate_history: Optional[list[DebateEntry]] = None,
    ) -> ContraAgentOutput:
        """Execute the Contra Agent pipeline.

        Args:
            research: Validated ``ResearchInput`` from the Research Agent.
            pro_output: The ``ProAgentOutput`` produced by the Pro Agent.
            debate_history: Optional list of previous ``DebateEntry`` objects
                for multi-round context.

        Returns:
            A ``ContraAgentOutput`` containing structured counter-arguments and risks.

        Raises:
            ValueError: If the LLM response cannot be parsed into valid JSON
                or does not conform to the ``ContraAgentOutput`` schema.
        """
        messages = self.build_prompt(research, pro_output, debate_history)

        response = self.llm.invoke(messages)

        try:
            raw_json = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response is not valid JSON:\n{response.content}"
            ) from exc

        try:
            output = ContraAgentOutput.model_validate(raw_json)
        except Exception as exc:
            raise ValueError(
                f"LLM JSON does not match ContraAgentOutput schema:\n{raw_json}"
            ) from exc

        return output

    # ── Convenience ───────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"ContraAgent(model={self.model_name!r}, "
            f"temperature={self.temperature})"
        )
