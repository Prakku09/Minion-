"""Schema definitions for Minions."""

from minions.schema.paper import (
    CanonicalSectionType,
    ContentBlock,
    ContentBlockType,
    DocumentMetadata,
    Equation,
    Figure,
    PaperStructure,
    ParserDiagnostics,
    Reference,
    Section,
    SectionParseConfidence,
    Table,
)
from minions.schema.critique import (
    ConsensusType,
    CritiqueConfidence,
    CritiqueDimension,
    CritiquePoint,
    DimensionScore,
    DroppedCritiquePoint,
    MethodologyCritiqueReport,
    MethodologyScores,
)

__all__ = [
    "CanonicalSectionType",
    "ContentBlock",
    "ContentBlockType",
    "DocumentMetadata",
    "Equation",
    "Figure",
    "PaperStructure",
    "ParserDiagnostics",
    "Reference",
    "Section",
    "SectionParseConfidence",
    "Table",
    "ConsensusType",
    "CritiqueConfidence",
    "CritiqueDimension",
    "CritiquePoint",
    "DimensionScore",
    "DroppedCritiquePoint",
    "MethodologyCritiqueReport",
    "MethodologyScores",
]
