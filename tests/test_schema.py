"""Unit tests for PaperStructure schema validation and serialization."""

import pytest
from pydantic import ValidationError
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


def test_schema_valid_construction():
    """Test valid instantiation of PaperStructure."""
    doc = PaperStructure(
        schema_version="1.0.0",
        metadata=DocumentMetadata(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            abstract_preview="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            venue="NeurIPS",
            year=2017,
            page_count=11,
            source_file="transformer.pdf",
        ),
        sections=[
            Section(
                id="sec_abstract",
                canonical_type=CanonicalSectionType.ABSTRACT,
                heading_title="Abstract",
                level=1,
                content_blocks=[
                    ContentBlock(
                        id="sec_abs_b01",
                        type=ContentBlockType.PARAGRAPH,
                        text="The dominant sequence transduction models are based on recurrent or convolutional neural networks...",
                        page_number=1,
                        bbox=(50.0, 100.0, 300.0, 150.0),
                    )
                ],
                raw_text="The dominant sequence transduction models are based on recurrent or convolutional neural networks...",
                figure_ids=[],
                table_ids=[],
                equation_ids=[],
            ),
            Section(
                id="sec_methodology",
                canonical_type=CanonicalSectionType.METHODOLOGY,
                heading_title="3 Model Architecture",
                level=1,
                content_blocks=[
                    ContentBlock(
                        id="sec_meth_b01",
                        type=ContentBlockType.PARAGRAPH,
                        text="Most competitive neural sequence transduction models have an encoder-decoder structure.",
                        page_number=2,
                        bbox=(50.0, 200.0, 300.0, 250.0),
                    )
                ],
                raw_text="Most competitive neural sequence transduction models have an encoder-decoder structure.",
                figure_ids=["fig_1"],
                table_ids=["tab_1"],
                equation_ids=["eq_1"],
            ),
        ],
        figures={
            "fig_1": Figure(
                id="fig_1",
                label="Figure 1",
                caption="Figure 1: The Transformer - model architecture.",
                page_number=3,
                bbox=(50.0, 50.0, 500.0, 400.0),
                image_path="assets/figures/fig_1.png",
                visual_summary="Diagram showing multi-head attention encoder-decoder architecture.",
                associated_section_id="sec_methodology",
            )
        },
        tables={
            "tab_1": Table(
                id="tab_1",
                label="Table 1",
                caption="Table 1: Maximum path lengths and per-layer complexity.",
                page_number=6,
                bbox=(50.0, 100.0, 500.0, 250.0),
                markdown_content="| Layer Type | Complexity per Layer | Max Path Length |\n|---|---|---|\n| Self-Attention | O(1) | O(1) |",
                associated_section_id="sec_methodology",
            )
        },
        equations={
            "eq_1": Equation(
                id="eq_1",
                label="(1)",
                latex=r"Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V",
                raw_text="Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
                page_number=4,
                bbox=(100.0, 300.0, 400.0, 330.0),
                is_image_fallback=False,
                latex_confidence=0.98,
                associated_section_id="sec_methodology",
            ),
            "eq_2": Equation(
                id="eq_2",
                label="(2)",
                latex=None,
                raw_text="complex 2D custom glyph formula",
                page_number=4,
                bbox=(100.0, 350.0, 400.0, 390.0),
                is_image_fallback=True,
                latex_confidence=0.45,
                image_path="assets/equations/eq_2.png",
                associated_section_id="sec_methodology",
            ),
        },
        references=[
            Reference(
                ref_id="[1]",
                raw_text="Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv:1607.06450, 2016.",
                title="Layer normalization",
                authors=["Jimmy Lei Ba", "Jamie Ryan Kiros", "Geoffrey E Hinton"],
                year=2016,
                venue="arXiv:1607.06450",
            )
        ],
        parser_diagnostics=ParserDiagnostics(
            total_pages_parsed=11,
            sections_found=["abstract", "methodology"],
            unassigned_blocks_count=0,
            overall_confidence=0.95,
            section_confidences={
                "sec_abstract": SectionParseConfidence(confidence=0.98, blocks_count=1, issues=[]),
                "sec_methodology": SectionParseConfidence(confidence=0.92, blocks_count=1, issues=[]),
            },
            warnings=[],
        ),
    )

    # Test serialization round-trip
    json_data = doc.model_dump_json()
    assert "schema_version" in json_data
    assert "1.0.0" in json_data
    loaded_doc = PaperStructure.model_validate_json(json_data)
    assert loaded_doc.metadata.title == "Attention Is All You Need"
    assert loaded_doc.equations["eq_2"].is_image_fallback is True
    assert loaded_doc.parser_diagnostics.section_confidences["sec_methodology"].confidence == 0.92
    assert loaded_doc.references[0].year == 2016


def test_schema_extra_fields_forbidden():
    """Verify extra fields are strictly forbidden by ConfigDict(extra='forbid')."""
    with pytest.raises(ValidationError):
        ContentBlock(
            id="sec_b01",
            type=ContentBlockType.PARAGRAPH,
            text="hello",
            page_number=1,
            invalid_extra_field="should_fail",
        )
