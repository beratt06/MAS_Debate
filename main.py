"""
Demo runner for the AI Decision Debate System.

Demonstrates the full multi-round debate pipeline:
1. Simulates Research Agent output.
2. DebateLoop orchestrates 3 rounds of Pro ↔ Contra debate.
3. Displays each round's arguments in real-time.
4. Exports the full debate transcript as structured JSON.
"""

from models import ResearchInput, ProAgentOutput, ContraAgentOutput
from pro_agent import ProAgent
from contra_agent import ContraAgent
from debate_loop import DebateLoop


def print_round(round_num: int, pro: ProAgentOutput, contra: ContraAgentOutput) -> None:
    """Callback: pretty-print a completed debate round."""
    print(f"\n{'═' * 65}")
    print(f"  ROUND {round_num}")
    print(f"{'═' * 65}")

    # ── Pro Agent ─────────────────────────────────────────────────
    print(f"\n  🟢 PRO AGENT\n")
    for i, arg in enumerate(pro.arguments, 1):
        print(f"    ✅ Argument {i}: {arg.title}")
        print(f"       {arg.explanation}")
        if arg.supporting_facts:
            for fact in arg.supporting_facts:
                print(f"         • {fact}")
        print()
    print(f"    📝 {pro.summary}\n")

    # ── Contra Agent ──────────────────────────────────────────────
    print(f"  🔴 CONTRA AGENT\n")
    for i, ca in enumerate(contra.counter_arguments, 1):
        print(f"    ❌ Counter {i} → \"{ca.target_argument}\"")
        print(f"       {ca.criticism}")
        if ca.evidence:
            for ev in ca.evidence:
                print(f"         • {ev}")
        print()

    for i, risk in enumerate(contra.risks, 1):
        icon = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}.get(risk.severity, "⚪")
        print(f"    {icon} Risk [{risk.severity}]: {risk.title}")
        print(f"       {risk.description}")
        print()

    print(f"    📝 {contra.summary}")


def main() -> None:
    # ── 1. Simulate Research Agent output ─────────────────────────
    research = ResearchInput(
        topic_summary="The use of Artificial Intelligence in healthcare diagnostics",
        facts=[
            "AI algorithms can analyze medical images (X-rays, MRIs) with accuracy rates exceeding 90%.",
            "Machine learning models can predict patient deterioration up to 48 hours in advance.",
            "AI-powered drug discovery has reduced early-stage research timelines by up to 40%.",
            "Natural Language Processing can extract structured data from unstructured clinical notes.",
            "AI chatbots handle over 60% of routine patient inquiries, freeing clinical staff time.",
        ],
    )

    # ── 2. Initialize agents and debate loop ──────────────────────
    pro_agent = ProAgent()
    contra_agent = ContraAgent()
    debate = DebateLoop(pro_agent, contra_agent, max_rounds=3)

    # ── 3. Run the debate ─────────────────────────────────────────
    print("╔" + "═" * 63 + "╗")
    print("║  AI DECISION DEBATE SYSTEM — Multi-Round Debate" + " " * 14 + "║")
    print("╚" + "═" * 63 + "╝")
    print(f"\n📋 Topic: {research.topic_summary}")
    print(f"🔄 Max Rounds: {debate.max_rounds}")

    transcript = debate.run(research, on_round_complete=print_round)

    # ── 4. Summary ────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  DEBATE COMPLETE")
    print(f"  Rounds: {transcript.total_rounds}  |  "
          f"Total entries: {len(transcript.full_history)}")
    print(f"{'─' * 65}")

    # ── 5. Export transcript JSON ─────────────────────────────────
    json_output = debate.export_json(research.topic_summary)
    print(f"\n📦 Transcript JSON ({len(json_output)} chars) — "
          "ready for Evidence Verifier & Judge Agent")

    # Optionally save to file
    with open("debate_transcript.json", "w", encoding="utf-8") as f:
        f.write(json_output)
    print("💾 Saved to debate_transcript.json")


if __name__ == "__main__":
    main()
