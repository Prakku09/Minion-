"""Grounding index round-trip test: samples anchor IDs and verifies text/bbox against raw PDF."""

import fitz  # PyMuPDF
import pytest
from minions.parser.pipeline import StructureParserPipeline


@pytest.fixture
def sample_academic_pdf(tmp_path) -> str:
    """Create a structured multi-page PDF with methodology, equations, and tables."""
    pdf_path = str(tmp_path / "grounding_test_paper.pdf")
    doc = fitz.open()

    # Page 1: Title, Abstract, Introduction
    p1 = doc.new_page(width=612, height=792)
    p1.insert_textbox(
        fitz.Rect(50, 40, 562, 70),
        "Graph Neural Diffusion for Molecular Property Prediction",
        fontsize=16,
        fontname="helv",
    )
    p1.insert_textbox(
        fitz.Rect(50, 90, 562, 140),
        "Abstract\nWe present a graph diffusion framework that captures long-range molecular interactions.",
        fontsize=10,
        fontname="helv",
    )
    p1.insert_textbox(
        fitz.Rect(50, 160, 562, 230),
        "1 Introduction\nMolecular graph learning requires modeling both local bonds and non-local interactions.",
        fontsize=10,
        fontname="helv",
    )

    # Page 2: Methodology & Equation
    p2 = doc.new_page(width=612, height=792)
    p2.insert_textbox(
        fitz.Rect(50, 50, 562, 90),
        "3 Methodology\nLet G = (V, E) denote a molecular graph with node feature matrix X in R^{N x D}.",
        fontsize=10,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(100, 120, 500, 150),
        "h_i^{(t+1)} = σ(∑_{j ∈ N(i)} W h_j^{(t)} + b) (1)",
        fontsize=10,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(50, 180, 562, 250),
        "We optimize the diffusion parameter using Adam with a learning rate of 0.001.",
        fontsize=10,
        fontname="helv",
    )

    # Page 3: References
    p3 = doc.new_page(width=612, height=792)
    p3.insert_textbox(
        fitz.Rect(50, 50, 562, 80),
        "References",
        fontsize=12,
        fontname="helv",
    )
    p3.insert_textbox(
        fitz.Rect(50, 90, 562, 140),
        "[1] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. ICLR 2017.",
        fontsize=9,
        fontname="helv",
    )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_grounding_roundtrip(sample_academic_pdf: str, tmp_path):
    """Confirm sampled anchor IDs accurately match their actual page text and bounding box."""
    run_dir = str(tmp_path / "run_grounding")
    pipeline = StructureParserPipeline(run_dir_root=run_dir)
    parsed_doc = pipeline.parse_pdf(sample_academic_pdf, output_dir=run_dir)

    # Open raw PDF for independent ground-truth verification
    raw_doc = fitz.open(sample_academic_pdf)

    # 1. Verify ContentBlock anchors
    sampled_blocks = []
    for sec in parsed_doc.sections:
        for cb in sec.content_blocks:
            sampled_blocks.append(cb)

    assert len(sampled_blocks) >= 3, "Should have extracted multiple content blocks"

    for cb in sampled_blocks:
        page_idx = cb.page_number - 1
        page = raw_doc[page_idx]
        assert cb.bbox is not None, f"Anchor {cb.id} must have a non-null bounding box"

        # Extract text from raw PDF at the exact bounding box
        rect = fitz.Rect(cb.bbox)
        # Expand slightly to account for line baseline anti-aliasing
        expanded_rect = fitz.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2)
        raw_box_text = page.get_textbox(expanded_rect).strip()

        # Check that key anchor words exist in the raw box text
        anchor_words = cb.text.split()
        for word in anchor_words[:3]:
            assert word in raw_box_text, f"Word '{word}' from anchor {cb.id} not found in raw PDF bbox ({raw_box_text})"

    # 2. Verify Equation anchors
    assert len(parsed_doc.equations) >= 1
    for eq_id, eq in parsed_doc.equations.items():
        page = raw_doc[eq.page_number - 1]
        assert eq.bbox is not None
        raw_eq_text = page.get_textbox(fitz.Rect(eq.bbox)).strip()
        assert "h_i" in raw_eq_text or "∑" in raw_eq_text or "W" in raw_eq_text

    raw_doc.close()
