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
    CritiqueConfidence,
    CritiqueDimension,
    CritiquePoint,
    DroppedCritiquePoint,
    MethodologyCritiqueReport,
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
    "CritiqueConfidence",
    "CritiqueDimension",
    "CritiquePoint",
    "DroppedCritiquePoint",
    "MethodologyCritiqueReport",
]
