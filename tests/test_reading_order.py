"""Tests for multi-column layout detection and reading-order validation."""

import os
import fitz  # PyMuPDF
import pytest
from minions.parser.layout import LayoutEngine


@pytest.fixture
def multi_column_pdf(tmp_path) -> str:
    """Generate a synthetic 2-column academic paper PDF fixture."""
    pdf_path = str(tmp_path / "two_column_test.pdf")
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard Letter size

    # Full-width Title
    page.insert_textbox(
        fitz.Rect(50, 40, 562, 70),
        "A Novel Architecture for Neural Reasoning",
        fontsize=16,
        fontname="helv",
    )

    # Full-width Abstract
    page.insert_textbox(
        fitz.Rect(80, 80, 532, 130),
        "Abstract: In this paper we present a dual branch network for scientific paper analysis.",
        fontsize=10,
        fontname="helv",
    )

    # Left Column (x0=50, x1=280)
    page.insert_textbox(
        fitz.Rect(50, 150, 280, 220),
        "1 Introduction\nLeft column paragraph one discussing background context.",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(50, 240, 280, 310),
        "Left column paragraph two discussing foundational methodology and motivation.",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(50, 330, 280, 400),
        "Left column paragraph three concluding the left side discussion.",
        fontsize=10,
        fontname="helv",
    )

    # Right Column (x0=332, x1=562)
    page.insert_textbox(
        fitz.Rect(332, 150, 562, 220),
        "2 Related Work\nRight column paragraph one discussing prior transformers and CNNs.",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(332, 240, 562, 310),
        "Right column paragraph two comparing benchmark datasets and evaluation metrics.",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(332, 330, 562, 400),
        "Right column paragraph three summarizing the differences in our approach.",
        fontsize=10,
        fontname="helv",
    )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_multi_column_reading_order(multi_column_pdf: str):
    """Assert that left column paragraphs are read strictly before right column paragraphs."""
    doc = fitz.open(multi_column_pdf)
    page = doc[0]
    engine = LayoutEngine()
    blocks = engine.extract_page_blocks(page, page_number=1)

    block_texts = [b.text for b in blocks]
    doc.close()

    # 1. Spanning title and abstract must be read first
    assert any("A Novel Architecture" in t for t in block_texts[:2])
    assert any("Abstract" in t for t in block_texts[:3])

    # 2. Left column blocks must precede Right column blocks
    intro_idx = next(i for i, t in enumerate(block_texts) if "1 Introduction" in t or "Left column paragraph one" in t)
    left_p3_idx = next(i for i, t in enumerate(block_texts) if "Left column paragraph three" in t)
    related_idx = next(i for i, t in enumerate(block_texts) if "2 Related Work" in t or "Right column paragraph one" in t)
    right_p3_idx = next(i for i, t in enumerate(block_texts) if "Right column paragraph three" in t)

    assert intro_idx < left_p3_idx, "Left column must be read in top-to-bottom sequence"
    assert left_p3_idx < related_idx, "All left-column paragraphs must precede right-column paragraphs (no interleaving)"
    assert related_idx < right_p3_idx, "Right column must be read in top-to-bottom sequence after left column"
