"""Validation test suite executing Structure Parser on real academic research papers."""

import os
import pytest
from minions.parser.pipeline import StructureParserPipeline
from minions.schema.paper import CanonicalSectionType, PaperStructure


@pytest.mark.parametrize(
    "paper_filename,expected_title_keywords,expected_sections",
    [
        (
            "attention.pdf",
            ["Attention", "Need"],
            [
                CanonicalSectionType.ABSTRACT,
                CanonicalSectionType.INTRODUCTION,
                CanonicalSectionType.METHODOLOGY,
                CanonicalSectionType.RESULTS,
                CanonicalSectionType.CONCLUSION,
                CanonicalSectionType.REFERENCES,
            ],
        ),
        (
            "lora.pdf",
            ["LoRA", "Low-Rank", "Adaptation"],
            [
                CanonicalSectionType.ABSTRACT,
                CanonicalSectionType.INTRODUCTION,
                CanonicalSectionType.METHODOLOGY,
                CanonicalSectionType.RESULTS,
                CanonicalSectionType.REFERENCES,
            ],
        ),
        (
            "resnet.pdf",
            ["Deep", "Residual", "Learning", "Image"],
            [
                CanonicalSectionType.ABSTRACT,
                CanonicalSectionType.INTRODUCTION,
                CanonicalSectionType.METHODOLOGY,
                CanonicalSectionType.RESULTS,
                CanonicalSectionType.REFERENCES,
            ],
        ),
        (
            "single_column_bio.pdf",
            ["Spatial", "Stochastic", "Dynamics", "Gene"],
            [
                CanonicalSectionType.ABSTRACT,
                CanonicalSectionType.INTRODUCTION,
                CanonicalSectionType.METHODOLOGY,
                CanonicalSectionType.RESULTS,
                CanonicalSectionType.DISCUSSION,
                CanonicalSectionType.CONCLUSION,
                CanonicalSectionType.REFERENCES,
            ],
        ),
    ],
)
def test_parse_real_papers(tmp_path, paper_filename, expected_title_keywords, expected_sections):
    """Test full Structure Parser pipeline on real arXiv research papers."""
    pdf_path = os.path.join("tests", "data", paper_filename)
    if not os.path.exists(pdf_path):
        pytest.skip(f"Paper file not found: {pdf_path}")

    out_dir = str(tmp_path / f"run_{paper_filename}")
    pipeline = StructureParserPipeline(run_dir_root=out_dir, dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    # 1. Assert Root Schema Version
    assert struct.schema_version == "1.0.0"

    # 2. Assert Document Metadata
    assert struct.metadata.page_count >= 3
    assert any(kw.lower() in struct.metadata.title.lower() for kw in expected_title_keywords)

    # 3. Assert Section Segmentation
    found_types = {s.canonical_type for s in struct.sections}
    for exp_sec in expected_sections:
        assert exp_sec in found_types, f"Expected canonical section '{exp_sec.value}' in {paper_filename}, found: {found_types}"

    # 4. Assert Content Blocks and Grounding Anchors
    total_blocks = sum(len(s.content_blocks) for s in struct.sections)
    assert total_blocks >= 10, f"Expected at least 10 content blocks in {paper_filename}, got {total_blocks}"

    # Verify anchor ID format
    for s in struct.sections:
        for cb in s.content_blocks:
            assert cb.id.startswith(f"{s.id}_b")
            assert cb.page_number >= 1
            assert cb.text.strip()

    # 5. Assert Figures & Visual Assets
    assert len(struct.figures) >= 1, f"Expected at least 1 figure in {paper_filename}"
    for fig_id, fig in struct.figures.items():
        assert fig.caption
        assert fig.image_path
        full_fig_path = os.path.join(out_dir, fig.image_path)
        assert os.path.exists(full_fig_path), f"Cropped figure file {full_fig_path} does not exist"
        assert os.path.getsize(full_fig_path) > 500, f"Cropped figure file {full_fig_path} is too small / empty"

    # 6. Assert Tables
    assert len(struct.tables) >= 1, f"Expected at least 1 table in {paper_filename}"
    for tab_id, tab in struct.tables.items():
        assert tab.caption
        assert tab.markdown_content

    # 7. Assert Equations
    assert len(struct.equations) >= 1, f"Expected at least 1 equation in {paper_filename}"
    for eq_id, eq in struct.equations.items():
        assert eq.raw_text
        assert 0.0 <= eq.latex_confidence <= 1.0
        if eq.is_image_fallback:
            assert eq.image_path
            full_eq_path = os.path.join(out_dir, eq.image_path)
            assert os.path.exists(full_eq_path)

    # 8. Assert References
    assert len(struct.references) >= 5, f"Expected at least 5 references in {paper_filename}, got {len(struct.references)}"
    assert struct.references[0].raw_text

    # 9. Assert Per-Section Diagnostics
    assert struct.parser_diagnostics.overall_confidence >= 0.70
    assert len(struct.parser_diagnostics.section_confidences) >= len(struct.sections)
    for sec in struct.sections:
        assert sec.id in struct.parser_diagnostics.section_confidences
        sec_conf = struct.parser_diagnostics.section_confidences[sec.id]
        assert 0.0 <= sec_conf.confidence <= 1.0

    # 10. Assert JSON Output Artifact
    json_path = os.path.join(out_dir, "01_structure_parser.json")
    assert os.path.exists(json_path)
    assert os.path.getsize(json_path) > 1000
