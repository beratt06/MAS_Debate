"""
Pydantic data models for the AI Decision Debate System.

Defines structured input/output schemas used across all debate agents.
"""

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Input Model  (Research Agent → Pro Agent)
# ──────────────────────────────────────────────
class ResearchInput(BaseModel):
    """Data produced by the Research Agent and consumed by debate agents.

    Attributes:
        topic_summary: A concise summary of the topic under debate.
        facts: A list of relevant facts or pieces of evidence gathered during research.
    """

    topic_summary: str = Field(
        ...,
        description="Concise summary of the topic under debate.",
        examples=["Artificial intelligence in healthcare"],
    )
    facts: list[str] = Field(
        ...,
        min_length=1,
        description="Relevant facts / evidence gathered by the Research Agent.",
        examples=[
            [
                "AI can detect diseases from medical images with over 90% accuracy.",
                "Machine learning models can predict patient outcomes.",
            ]
        ],
    )


# ──────────────────────────────────────────────
# Output Models  (Pro Agent → Orchestrator)
# ──────────────────────────────────────────────
class Argument(BaseModel):
    """A single structured argument in favor of the topic.

    Attributes:
        title: Short, descriptive headline for the argument.
        explanation: Detailed reasoning that supports the argument.
        supporting_facts: Facts from the research that back this argument.
    """

    title: str = Field(
        ...,
        description="Short descriptive headline for the argument.",
    )
    explanation: str = Field(
        ...,
        description="Detailed reasoning that supports the argument.",
    )
    supporting_facts: list[str] = Field(
        default_factory=list,
        description="Facts from the research input that back this argument.",
    )


class ProAgentOutput(BaseModel):
    """Structured output produced by the Pro Agent.

    Attributes:
        stance: The position taken — always 'PRO' for this agent.
        arguments: A list of at least three structured arguments.
        summary: A brief concluding statement summarising the pro position.
    """

    stance: str = Field(
        default="PRO",
        description="The position taken by this agent. Always 'PRO'.",
    )
    arguments: list[Argument] = Field(
        ...,
        min_length=3,
        description="Structured arguments in favor of the topic (minimum 3).",
    )
    summary: str = Field(
        ...,
        description="Brief concluding statement summarising the pro position.",
    )


# ──────────────────────────────────────────────
# Output Models  (Contra Agent → Orchestrator)
# ──────────────────────────────────────────────
class CounterArgument(BaseModel):
    """A structured counter-argument that challenges a specific pro argument.

    Attributes:
        target_argument: The title of the pro argument being challenged.
        criticism: Detailed critique explaining why the pro argument is weak, flawed, or incomplete.
        evidence: Supporting evidence or reasoning that backs the criticism.
    """

    target_argument: str = Field(
        ...,
        description="Title of the pro argument being challenged.",
    )
    criticism: str = Field(
        ...,
        description="Detailed critique explaining why the pro argument is weak, flawed, or incomplete.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence, examples, or logical reasoning that backs this criticism.",
    )


class Risk(BaseModel):
    """An independent risk or concern related to the topic.

    Attributes:
        title: Short, descriptive headline for the risk.
        description: Detailed explanation of the risk and its potential impact.
        severity: Risk severity level — LOW, MEDIUM, or HIGH.
    """

    title: str = Field(
        ...,
        description="Short descriptive headline for the risk.",
    )
    description: str = Field(
        ...,
        description="Detailed explanation of the risk and its potential impact.",
    )
    severity: str = Field(
        ...,
        description="Risk severity level: 'LOW', 'MEDIUM', or 'HIGH'.",
        pattern=r"^(LOW|MEDIUM|HIGH)$",
    )


class ContraAgentOutput(BaseModel):
    """Structured output produced by the Contra Agent.

    Attributes:
        stance: The position taken — always 'CONTRA' for this agent.
        counter_arguments: Critiques targeting specific pro arguments (minimum 3).
        risks: Independent risks and concerns about the topic (minimum 2).
        summary: Brief concluding statement summarising the contra position.
    """

    stance: str = Field(
        default="CONTRA",
        description="The position taken by this agent. Always 'CONTRA'.",
    )
    counter_arguments: list[CounterArgument] = Field(
        ...,
        min_length=3,
        description="Critiques targeting specific pro arguments (minimum 3).",
    )
    risks: list[Risk] = Field(
        ...,
        min_length=2,
        description="Independent risks and concerns about the topic (minimum 2).",
    )
    summary: str = Field(
        ...,
        description="Brief concluding statement summarising the contra position.",
    )


# ──────────────────────────────────────────────
# Debate Loop Models  (Memory & Transcript)
# ──────────────────────────────────────────────
class DebateEntry(BaseModel):
    """A single turn in the debate — one agent's contribution in one round.

    Attributes:
        agent: Which agent produced this entry ('PRO' or 'CONTRA').
        round_number: The debate round (1-indexed).
        content: The agent's full structured output, serialized as a dict.
    """

    agent: str = Field(
        ...,
        description="Agent identifier: 'PRO' or 'CONTRA'.",
        pattern=r"^(PRO|CONTRA)$",
    )
    round_number: int = Field(
        ...,
        ge=1,
        description="Debate round number (1-indexed).",
    )
    content: dict = Field(
        ...,
        description="The agent's full structured output as a dictionary.",
    )


class DebateRound(BaseModel):
    """A complete debate round containing both the pro and contra entries.

    Attributes:
        round_number: The round number (1-indexed).
        pro_entry: The Pro Agent's contribution.
        contra_entry: The Contra Agent's contribution.
    """

    round_number: int = Field(..., ge=1, description="Round number (1-indexed).")
    pro_entry: DebateEntry = Field(..., description="Pro Agent's entry for this round.")
    contra_entry: DebateEntry = Field(..., description="Contra Agent's entry for this round.")


class DebateTranscript(BaseModel):
    """The complete debate transcript, ready for export to Evidence Verifier and Judge Agent.

    Attributes:
        topic: The topic under debate.
        total_rounds: How many rounds the debate lasted.
        rounds: Ordered list of all debate rounds.
        full_history: Flat chronological list of every entry.
    """

    topic: str = Field(..., description="The topic under debate.")
    total_rounds: int = Field(..., ge=1, description="Total number of rounds completed.")
    rounds: list[DebateRound] = Field(..., description="Ordered list of debate rounds.")
    full_history: list[DebateEntry] = Field(
        ...,
        description="Flat chronological list of all debate entries.",
    )
