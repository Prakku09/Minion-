"""Pydantic schemas for Critic agents."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CritiqueDimension(str, Enum):
    REPRODUCIBILITY = "reproducibility"
    ASSUMPTIONS = "assumptions"
    LIMITATIONS = "limitations"
    APPROPRIATENESS = "appropriateness"


class CritiqueConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CritiquePoint(BaseModel):
    """A single evidence-grounded critique point."""

    model_config = ConfigDict(extra="forbid")

    anchor_ids: List[str] = Field(
        ...,
        description="Exact anchor ID(s) (e.g. sec_methodology_3_4_implementation_b01, eq_1) this critique is reacting to.",
    )
    quoted_evidence: str = Field(
        ...,
        description="Verbatim quoted text or asset description from the cited anchor block.",
    )
    critique_dimension: CritiqueDimension = Field(
        ...,
        description="The evaluation dimension: reproducibility, assumptions, limitations, or appropriateness.",
    )
    critique_text: str = Field(
        ...,
        description="The structured evaluation / critique reacting to the cited evidence.",
    )
    confidence: CritiqueConfidence = Field(
        ...,
        description="Confidence level of the critique based on clarity of document evidence (high, medium, low).",
    )


class DroppedCritiquePoint(BaseModel):
    """A candidate point dropped during post-hoc grounding verification."""

    model_config = ConfigDict(extra="forbid")

    point: CritiquePoint = Field(..., description="The candidate point that was rejected.")
    drop_reason: str = Field(..., description="The exact reason verification failed.")


class MethodologyCritiqueReport(BaseModel):
    """Complete structured critique report for Methodology sections."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="1.0.0", description="Schema version of the critique output."
    )
    paper_title: str = Field(..., description="Title of the evaluated paper.")
    target_sections: List[str] = Field(
        ...,
        description="List of section IDs evaluated (e.g. ['sec_methodology_3_1_residual_learning', ...]).",
    )
    critiques: List[CritiquePoint] = Field(
        default_factory=list,
        description="List of evidence-grounded critique objects identifying genuine gaps, ambiguities, unjustified choices, or risks.",
    )
    strengths: List[CritiquePoint] = Field(
        default_factory=list,
        description="List of evidence-grounded strength and confirmatory observations where methodology is well-specified or rigorous.",
    )
    dropped_points: List[DroppedCritiquePoint] = Field(
        default_factory=list,
        description="List of candidate points dropped due to failed post-hoc grounding verification.",
    )
    hallucination_rate: float = Field(
        default=0.0,
        description="Fraction of candidate points dropped by verification (0.0 to 1.0).",
    )
    summary: Optional[str] = Field(
        default=None, description="High-level synthesis of methodological findings."
    )
