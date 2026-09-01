"""Pydantic schemas for structured representation of scientific papers.

Schema Version: 1.0.0
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class CanonicalSectionType(str, Enum):
    """Canonical section categories for academic papers."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"


class ContentBlockType(str, Enum):
    """Types of fine-grained content blocks within a section."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    EQUATION = "equation"
    TABLE_REF = "table_ref"
    FIGURE_REF = "figure_ref"


class ContentBlock(BaseModel):
    """A fine-grained, verifiable text/content unit with a deterministic anchor ID."""
    id: str = Field(..., description="Unique anchor ID, e.g., 'sec_meth_b01'")
    type: ContentBlockType = Field(ContentBlockType.PARAGRAPH, description="Type of content block")
    text: str = Field(..., description="Exact verbatim text extracted from PDF")
    page_number: int = Field(..., ge=1, description="1-indexed page number where block begins")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None, description="[x0, y0, x1, y1] coordinates on the page in PDF points"
    )

    model_config = ConfigDict(extra="forbid")


class Figure(BaseModel):
    """Extracted figure asset with metadata, crop image path, and caption."""
    id: str = Field(..., description="Unique figure ID, e.g. 'fig_1'")
    label: str = Field(..., description="Figure label from document, e.g. 'Figure 1'")
    caption: str = Field(..., description="Full figure caption text")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None, description="[x0, y0, x1, y1] bounding box in PDF points"
    )
    image_path: str = Field(..., description="Relative path to saved cropped PNG asset")
    visual_summary: Optional[str] = Field(None, description="Multimodal visual summary/description of chart or diagram")
    associated_section_id: Optional[str] = Field(None, description="ID of section where figure is located or discussed")

    model_config = ConfigDict(extra="forbid")


class Table(BaseModel):
    """Extracted table with structured markdown representation and optional crop image."""
    id: str = Field(..., description="Unique table ID, e.g. 'tab_1'")
    label: str = Field(..., description="Table label from document, e.g. 'Table 1'")
    caption: str = Field(..., description="Full table caption text")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None, description="[x0, y0, x1, y1] bounding box in PDF points"
    )
    markdown_content: str = Field(..., description="Clean Markdown table grid")
    image_path: Optional[str] = Field(None, description="Relative path to saved cropped PNG asset if captured")
    associated_section_id: Optional[str] = Field(None, description="ID of section containing table")

    model_config = ConfigDict(extra="forbid")


class Equation(BaseModel):
    """Extracted mathematical formula with LaTeX representation or visual crop fallback."""
    id: str = Field(..., description="Unique equation ID, e.g. 'eq_1'")
    label: Optional[str] = Field(None, description="Equation number/label if present, e.g. '(1)' or '(2.3)'")
    latex: Optional[str] = Field(None, description="Reconstructed LaTeX formula if confidence >= threshold")
    raw_text: str = Field(..., description="Extracted raw text or Unicode representation")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None, description="[x0, y0, x1, y1] bounding box in PDF points"
    )
    is_image_fallback: bool = Field(
        False, description="True if formula extraction fell back to image cropping due to low confidence or complex glyphs"
    )
    latex_confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Confidence score of LaTeX/text reconstruction (0.0 to 1.0)"
    )
    image_path: Optional[str] = Field(None, description="Path to cropped equation image when visual fallback is active")
    associated_section_id: Optional[str] = Field(None, description="ID of section where equation appears")

    model_config = ConfigDict(extra="forbid")


class Reference(BaseModel):
    """Bibliography reference entry with optional structured fields."""
    ref_id: str = Field(..., description="Unique citation key or marker, e.g. '[1]', 'Vaswani2017'")
    raw_text: str = Field(..., description="Full verbatim reference text as printed in bibliography")
    title: Optional[str] = Field(None, description="Parsed paper/book title")
    authors: Optional[List[str]] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Journal, conference, or publisher")
    doi_or_url: Optional[str] = Field(None, description="DOI link or URL if available")

    model_config = ConfigDict(extra="forbid")


class Section(BaseModel):
    """Hierarchical section node in the structured paper map."""
    id: str = Field(..., description="Unique section ID, e.g. 'sec_methodology' or 'sec_meth_3_1'")
    canonical_type: CanonicalSectionType = Field(..., description="Canonical standard section category")
    heading_title: str = Field(..., description="Raw heading title verbatim from paper")
    level: int = Field(1, ge=1, description="Heading level: 1=top-level section, 2=subsection, etc.")
    subsections: List["Section"] = Field(default_factory=list, description="Nested child subsections")
    content_blocks: List[ContentBlock] = Field(
        default_factory=list, description="Ordered verbatim text blocks with anchor IDs"
    )
    raw_text: str = Field("", description="Aggregated clean plain text of section content")
    figure_ids: List[str] = Field(default_factory=list, description="IDs of figures referenced or situated here")
    table_ids: List[str] = Field(default_factory=list, description="IDs of tables referenced or situated here")
    equation_ids: List[str] = Field(default_factory=list, description="IDs of equations referenced or situated here")

    model_config = ConfigDict(extra="forbid")


class SectionParseConfidence(BaseModel):
    """Diagnostics and confidence score for a specific parsed section."""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Parse confidence score (0.0 to 1.0)")
    blocks_count: int = Field(..., ge=0, description="Total content blocks in section")
    issues: List[str] = Field(default_factory=list, description="Any detected formatting or extraction anomalies")

    model_config = ConfigDict(extra="forbid")


class ParserDiagnostics(BaseModel):
    """Document-wide and per-section extraction diagnostics."""
    total_pages_parsed: int = Field(..., ge=1, description="Number of pages processed")
    sections_found: List[str] = Field(..., description="List of canonical sections identified")
    unassigned_blocks_count: int = Field(0, ge=0, description="Blocks not assigned to any major section")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregate parser confidence (0.0 to 1.0)")
    section_confidences: Dict[str, SectionParseConfidence] = Field(
        default_factory=dict, description="Per-section confidence breakdown keyed by section ID"
    )
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings encountered during parsing")

    model_config = ConfigDict(extra="forbid")


class DocumentMetadata(BaseModel):
    """High-level paper metadata."""
    title: str = Field(..., description="Title of research paper")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract_preview: Optional[str] = Field(None, description="Initial preview text of abstract")
    venue: Optional[str] = Field(None, description="Publication venue / conference / journal")
    year: Optional[int] = Field(None, description="Publication year")
    page_count: int = Field(..., ge=1, description="Total page count")
    source_file: str = Field(..., description="Original filename or path of input PDF")

    model_config = ConfigDict(extra="forbid")


class PaperStructure(BaseModel):
    """Root structured document model produced by Structure Parser (Schema v1.0.0)."""
    schema_version: Literal["1.0.0"] = Field("1.0.0", description="Semantic schema version")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    sections: List[Section] = Field(..., description="Structured hierarchical sections")
    figures: Dict[str, Figure] = Field(default_factory=dict, description="Map of figure ID -> Figure details")
    tables: Dict[str, Table] = Field(default_factory=dict, description="Map of table ID -> Table details")
    equations: Dict[str, Equation] = Field(default_factory=dict, description="Map of equation ID -> Equation details")
    references: List[Reference] = Field(default_factory=list, description="Ordered list of bibliographic references")
    parser_diagnostics: ParserDiagnostics = Field(..., description="Parser performance & confidence diagnostics")

    model_config = ConfigDict(extra="forbid")
