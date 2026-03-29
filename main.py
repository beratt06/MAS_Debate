"""CLI entrypoint for the unified multi-agent debate system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import OLLAMA_MODEL_NAME, RETRIEVAL_TOP_K
from multiagent_system import MultiAgentDebateSystem


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pipeline execution."""

    parser = argparse.ArgumentParser(
        description="Run the full multi-agent debate pipeline from terminal.",
    )
    parser.add_argument("question", help="Question to debate.")
    parser.add_argument("--model", default=OLLAMA_MODEL_NAME, help="Ollama model name.")
    parser.add_argument("--rounds", type=int, default=3, help="Debate round count.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help="Retriever top-k chunk count.",
    )
    parser.add_argument(
        "--output",
        default="multiagent_result.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build or refresh retrieval index before running debate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run complete pipeline and save structured result as JSON."""

    args = parse_args()
    system = MultiAgentDebateSystem(
        model_name=args.model,
        max_rounds=max(args.rounds, 1),
        retrieval_top_k=max(args.top_k, 1),
    )

    if args.build_index:
        report = system.build_index()
        print(
            "Index report: "
            f"pdfs={report.pdf_count}, pages={report.page_count}, "
            f"chunks={report.chunk_count}, newly_indexed={report.indexed_chunk_count}"
        )

    result = system.run(args.question)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Debate finished. Output saved to: {output_path}")
    judge = result.get("judge", {})
    if isinstance(judge, dict):
        print(f"Judge recommendation: {judge.get('Oneri', 'N/A')}")
        print(f"Judge confidence: {judge.get('Guven_Skoru', 'N/A')}")


if __name__ == "__main__":
    main()
