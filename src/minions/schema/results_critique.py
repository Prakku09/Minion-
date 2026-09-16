"""Pydantic schemas for Results & Experimental Evaluation Critic.

Grounding Contract:
- Every critique must cite one or more verified document anchors.
- quoted_evidence must be grounded against the exact document text,
  figure/table description, or other indexed evidence.
- critique_text contains analytical evaluation based on the verified evidence.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from minions.schema.critique import (
    ConsensusType,
    CritiqueConfidence,
)


class ResultsDimension(str, Enum):
    """Evaluation dimensions for experimental/results analysis."""

    EXPERIMENTAL_REPRODUCIBILITY = "experimental_reproducibility"
    EVALUATION_RIGOR = "evaluation_rigor"
    EXPERIMENTAL_DESIGN = "experimental_design"
    RESULTS_INTERPRETATION = "results_interpretation"


class ResultsCritiquePoint(BaseModel):
    """A single evidence-grounded results critique or strength."""

    model_config = ConfigDict(extra="forbid")

    anchor_ids: List[str] = Field(
        ...,
        description=(
            "Exact anchor ID(s) from the parsed paper that support "
            "this critique or strength."
        ),
    )

    quoted_evidence: str = Field(
        ...,
        description=(
            "Verbatim evidence from the cited anchor. "
            "Must be verified against the document."
        ),
    )

    critique_dimension: ResultsDimension = Field(
        ...,
        description=(
            "Results evaluation dimension: experimental reproducibility, "
            "evaluation rigor, experimental design, or results interpretation."
        ),
    )

    critique_text: str = Field(
        ...,
        description=(
            "Analytical evaluation reacting to the verified evidence."
        ),
    )

    confidence: CritiqueConfidence = Field(
        ...,
        description="Confidence based on clarity of the supporting evidence.",
    )

    consensus: Optional[ConsensusType] = Field(
        default=None,
        description=(
            "Consensus classification across independent critic runs."
        ),
    )


class DroppedResultsCritiquePoint(BaseModel):
    """A candidate results point rejected during grounding verification."""

    model_config = ConfigDict(extra="forbid")

    point: ResultsCritiquePoint = Field(
        ...,
        description="Candidate point rejected during verification.",
    )

    drop_reason: str = Field(
        ...,
        description="Exact reason the candidate point failed verification.",
    )


class ResultsCritiqueReport(BaseModel):
    """Complete structured report produced by the Results Critic."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="1.0.0",
        description="Schema version of the results critique output.",
    )

    paper_title: str = Field(
        ...,
        description="Title of the evaluated paper.",
    )

    target_sections: List[str] = Field(
        ...,
        description=(
            "Section IDs evaluated by the Results Critic."
        ),
    )

    critiques: List[ResultsCritiquePoint] = Field(
        default_factory=list,
        description=(
            "Evidence-grounded weaknesses, gaps, ambiguities, "
            "or experimental risks."
        ),
    )

    strengths: List[ResultsCritiquePoint] = Field(
        default_factory=list,
        description=(
            "Evidence-grounded experimental strengths "
            "and well-supported results."
        ),
    )

    dropped_points: List[DroppedResultsCritiquePoint] = Field(
        default_factory=list,
        description=(
            "Candidate points rejected because grounding verification failed."
        ),
    )

    hallucination_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of candidate points rejected during verification."
        ),
    )

    summary: Optional[str] = Field(
        default=None,
        description="High-level synthesis of experimental/results findings.",
    )

    runs_evaluated_count: Optional[int] = Field(
        default=None,
        description="Number of independent critic runs used for consensus.",
    )