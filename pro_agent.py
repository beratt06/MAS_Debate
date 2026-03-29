"""
Pro Agent — Advocates the positive side of a topic in the AI Decision Debate System.

This module implements the ProAgent class which:
1. Receives structured research data from the Research Agent.
2. Optionally receives debate history to build upon previous rounds.
3. Constructs a prompt that enforces a strictly positive stance.
4. Calls an LLM via LangChain and returns validated, structured arguments.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from models import ResearchInput, ProAgentOutput, DebateEntry
from prompts import PRO_AGENT_SYSTEM_PROMPT


class ProAgent:
    """Agent that generates structured arguments **in favor of** a given topic.

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
                body = f"{args_text}  Ozet: {content.get('summary', '')}"
            else:  # CONTRA
                ca_text = ""
                for i, ca in enumerate(content.get("counter_arguments", []), 1):
                    ca_text += f"  - **[hedef: {ca['target_argument']}]**: {ca['criticism']}\n"
                risk_text = ""
                for r in content.get("risks", []):
                    risk_text += f"  - **[{r['severity']}] {r['title']}**: {r['description']}\n"
                body = f"**Karsi Argumanlar:**\n{ca_text}**Riskler:**\n{risk_text}  Ozet: {content.get('summary', '')}"

            sections.append(f"{header}\n{body}")

        return "\n\n".join(sections)

    # ── Prompt Construction ───────────────────────────────────────────
    @staticmethod
    def build_prompt(
        research: ResearchInput,
        debate_history: Optional[list[DebateEntry]] = None,
    ) -> list:
        """Build the message list to send to the LLM.

        Args:
            research: Validated research data from the Research Agent.
            debate_history: Optional list of previous debate entries for context.

        Returns:
            A list of LangChain ``BaseMessage`` instances (system + human).
        """
        facts_text = "\n".join(
            f"  - Olgu {i}: {fact}" for i, fact in enumerate(research.facts, 1)
        )

        human_content = f"## Konu\n{research.topic_summary}\n\n## Arastirma Olgulari\n{facts_text}\n\n"

        # Add debate history if available (round 2+)
        if debate_history:
            history_text = ProAgent._format_history(debate_history)
            human_content += (
                f"## Onceki Tartisma Gecmisi\n{history_text}\n\n"
                "ONEMLI: Contra Agent onceki argumanlarini elestirdi. "
                "Simdi SUNLARI yapmalisin:\n"
                "1. Onceki argumanlarini gelen elestirilere karsi savun.\n"
                "2. Tespit edilen zayifliklari gideren YENI ve DAHA GUCLU pro argumanlar sun.\n"
                "3. Onceki turlardaki belirli karsi argumanlara dogrudan atif yapip cevap ver.\n\n"
            )

        human_content += (
            "Yukaridaki konu, olgular ve varsa onceki tartisma gecmisine dayanarak "
            "PRO argumanlarini belirtilen JSON formatinda uret. "
            "JSON anahtarlari ayni kalsin; metin alanlarinin tamamini Turkce yaz."
        )

        return [
            SystemMessage(content=PRO_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    # ── Main Execution ────────────────────────────────────────────────
    def run(
        self,
        research: ResearchInput,
        debate_history: Optional[list[DebateEntry]] = None,
    ) -> ProAgentOutput:
        """Execute the Pro Agent pipeline.

        Args:
            research: Validated ``ResearchInput`` from the Research Agent.
            debate_history: Optional list of previous ``DebateEntry`` objects
                for multi-round context.

        Returns:
            A ``ProAgentOutput`` containing structured pro arguments.

        Raises:
            ValueError: If the LLM response cannot be parsed into valid JSON
                or does not conform to the ``ProAgentOutput`` schema.
        """
        messages = self.build_prompt(research, debate_history)

        response = self.llm.invoke(messages)

        raw_json = self._parse_json_response(response.content)

        try:
            output = ProAgentOutput.model_validate(raw_json)
        except Exception as exc:
            raise ValueError(
                f"LLM JSON does not match ProAgentOutput schema:\n{raw_json}"
            ) from exc

        return output

    def _parse_json_response(self, content: object) -> dict:
        """Parse model output into JSON even when wrapped with markdown fences."""

        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)

        text = str(content).strip()
        if not text:
            raise ValueError("LLM response is empty.")

        candidates: list[str] = [text]

        if "```json" in text:
            fenced = text.split("```json", 1)[1]
            fenced = fenced.split("```", 1)[0].strip()
            candidates.append(fenced)
        elif "```" in text:
            fenced = text.split("```", 1)[1]
            fenced = fenced.split("```", 1)[0].strip()
            candidates.append(fenced)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        raise ValueError(f"LLM response is not valid JSON:\n{text}")

    # ── Convenience ───────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"ProAgent(model={self.model_name!r}, "
            f"temperature={self.temperature})"
        )
