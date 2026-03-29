"""Unified multi-agent orchestration for research, debate, verification, and judging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.research_agent import ResearchAgent
from config import OLLAMA_MODEL_NAME, PDF_DIR, RETRIEVAL_TOP_K
from contra_agent import ContraAgent
from debate_loop import DebateLoop
from last_part import debate_scoring, evidence_verifier, judge_agent
from models import ResearchInput
from pro_agent import ProAgent
from retrieval.chunker import chunk_page_records
from retrieval.pdf_loader import load_pdf_pages
from retrieval.vector_store import ChromaVectorStore
from router.router import DebateRouter


@dataclass(slots=True)
class IndexBuildReport:
    """Summary of retrieval indexing operation."""

    pdf_count: int
    page_count: int
    chunk_count: int
    indexed_chunk_count: int


class MultiAgentDebateSystem:
    """Runs the full pipeline as one coherent multi-agent system."""

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL_NAME,
        max_rounds: int = 3,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
    ) -> None:
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.retrieval_top_k = retrieval_top_k

        self.router = DebateRouter(ResearchAgent(model_name=self.model_name))
        self.pro_agent = ProAgent(model_name=self.model_name)
        self.contra_agent = ContraAgent(model_name=self.model_name)

    def build_index(self) -> IndexBuildReport:
        """Load PDFs, chunk pages, and upsert vectors into Chroma."""

        pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
        pages = load_pdf_pages()
        chunks = chunk_page_records(pages)
        indexed_chunk_count = ChromaVectorStore().index_chunks(chunks)

        return IndexBuildReport(
            pdf_count=len(pdf_files),
            page_count=len(pages),
            chunk_count=len(chunks),
            indexed_chunk_count=indexed_chunk_count,
        )

    def run(self, question: str) -> dict[str, Any]:
        """Execute research, multi-round debate, verification, and final judgment."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        state = self.router.route(cleaned_question, top_k=self.retrieval_top_k)
        facts = self._resolve_facts(state.facts, state.retrieved_context, state.research_summary)

        research_input = ResearchInput(topic_summary=state.research_summary, facts=facts)

        debate_loop = DebateLoop(
            pro_agent=self.pro_agent,
            contra_agent=self.contra_agent,
            max_rounds=self.max_rounds,
        )
        transcript = debate_loop.run(research_input)

        claims_for_verification = self._collect_claims_for_verification(transcript.model_dump())
        verifier_logs = self._run_verification_and_scoring(claims_for_verification)

        transcript_text = self._to_debate_history_text(transcript.model_dump())
        judge_result = self._run_judge(
            question=cleaned_question,
            research={
                "topic_summary": state.research_summary,
                "facts": facts,
                "sources": state.sources,
            },
            debate_history_text=transcript_text,
            verifier_logs=verifier_logs,
        )

        return {
            "question": cleaned_question,
            "research": {
                "topic_summary": state.research_summary,
                "facts": facts,
                "sources": state.sources,
                "retrieved_context": state.retrieved_context,
            },
            "debate": transcript.model_dump(),
            "verifier_logs": verifier_logs,
            "judge": judge_result,
        }

    def _resolve_facts(
        self,
        key_facts: list[str],
        retrieved_context: list[str],
        summary: str,
    ) -> list[str]:
        """Ensure the debate stage receives at least one factual item."""

        if key_facts:
            return key_facts

        fallback_from_context = [text.strip() for text in retrieved_context[:5] if text.strip()]
        if fallback_from_context:
            return fallback_from_context

        return [summary or "No concrete facts were retrieved from the PDF corpus."]

    def _collect_claims_for_verification(self, debate: dict[str, Any]) -> list[str]:
        """Collect a bounded set of claims so verifier calls stay practical."""

        claims: list[str] = []

        for round_record in debate.get("rounds", []):
            pro_entry = round_record.get("pro_entry", {}).get("content", {})
            contra_entry = round_record.get("contra_entry", {}).get("content", {})

            for arg in pro_entry.get("arguments", []):
                explanation = arg.get("explanation", "").strip()
                if explanation:
                    claims.append(explanation)

            for counter in contra_entry.get("counter_arguments", []):
                criticism = counter.get("criticism", "").strip()
                if criticism:
                    claims.append(criticism)

            for risk in contra_entry.get("risks", []):
                description = risk.get("description", "").strip()
                if description:
                    claims.append(description)

        unique_claims: list[str] = []
        seen: set[str] = set()
        for claim in claims:
            if claim in seen:
                continue
            seen.add(claim)
            unique_claims.append(claim)

        return unique_claims[:8]

    def _run_verification_and_scoring(self, claims: list[str]) -> list[dict[str, Any]]:
        """Run Evidence Verifier and Debate Scoring for each selected claim."""

        logs: list[dict[str, Any]] = []

        for claim in claims:
            verifier_result = self._safe_call(evidence_verifier, claim)
            scoring_result = self._safe_call(debate_scoring, claim)
            logs.append(
                {
                    "claim": claim,
                    "evidence_verifier": verifier_result,
                    "debate_scoring": scoring_result,
                }
            )

        return logs

    def _run_judge(
        self,
        question: str,
        research: dict[str, Any],
        debate_history_text: str,
        verifier_logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the final judge agent with fully serialized context."""

        return self._safe_call(
            judge_agent,
            question,
            json.dumps(research, ensure_ascii=False),
            debate_history_text,
            json.dumps(verifier_logs, ensure_ascii=False),
        )

    def _to_debate_history_text(self, debate: dict[str, Any]) -> str:
        """Serialize debate rounds into concise plain text for judge context."""

        lines: list[str] = []

        for round_record in debate.get("rounds", []):
            round_number = round_record.get("round_number")
            pro_content = round_record.get("pro_entry", {}).get("content", {})
            contra_content = round_record.get("contra_entry", {}).get("content", {})

            lines.append(f"Round {round_number} - PRO Summary: {pro_content.get('summary', '')}")
            lines.append(
                f"Round {round_number} - CONTRA Summary: {contra_content.get('summary', '')}"
            )

        return "\n".join(lines)

    def _safe_call(self, fn: Any, *args: Any) -> dict[str, Any]:
        """Protect orchestration from downstream model or JSON parser failures."""

        try:
            result = fn(*args)
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as exc:
            return {"error": str(exc)}
