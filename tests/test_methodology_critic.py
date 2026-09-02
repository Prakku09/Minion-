"""Unit and integration tests for MethodologyCritic agent."""

import os
import pytest
from minions.critics.methodology import MethodologyCritic
from minions.parser.pipeline import StructureParserPipeline
from minions.schema.critique import (
    CritiqueConfidence,
    CritiqueDimension,
    CritiquePoint,
    MethodologyCritiqueReport,
)


def test_critique_schema_validation():
    """Verify CritiquePoint and MethodologyCritiqueReport serialization and validation."""
    cp = CritiquePoint(
        anchor_ids=["sec_methodology_b01"],
        quoted_evidence="We adopt batch normalization right after each convolution",
        critique_dimension=CritiqueDimension.APPROPRIATENESS,
        critique_text="Appropriate normalization placement ensures stable training dynamics.",
        confidence=CritiqueConfidence.HIGH,
    )
    report = MethodologyCritiqueReport(
        schema_version="1.0.0",
        paper_title="Deep Residual Learning",
        target_sections=["sec_methodology"],
        critiques=[cp],
        summary="Test summary",
    )

    data = report.model_dump()
    assert data["schema_version"] == "1.0.0"
    assert len(data["critiques"]) == 1
    assert data["critiques"][0]["critique_dimension"] == "appropriateness"
    assert data["critiques"][0]["confidence"] == "high"


def test_grounding_rejection_of_hallucinated_anchors():
    """Verify that ungrounded or mismatched critiques are strictly rejected."""
    critic = MethodologyCritic()
    anchor_map = {
        "sec_methodology_b01": "We use SGD with mini-batch size of 256 and learning rate of 0.1."
    }

    # 1. Valid Grounded Critique
    valid_cp = CritiquePoint(
        anchor_ids=["sec_methodology_b01"],
        quoted_evidence="SGD with mini-batch size of 256",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Specific optimizer settings are clearly provided.",
        confidence=CritiqueConfidence.HIGH,
    )
    is_valid, _ = critic._verify_critique_grounding(valid_cp, anchor_map)
    assert is_valid is True

    # 2. Invalid Anchor ID
    invalid_aid_cp = CritiquePoint(
        anchor_ids=["sec_nonexistent_b99"],
        quoted_evidence="SGD with mini-batch size of 256",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Specific optimizer settings are clearly provided.",
        confidence=CritiqueConfidence.HIGH,
    )
    is_valid, reason = critic._verify_critique_grounding(invalid_aid_cp, anchor_map)
    assert is_valid is False
    assert "does not exist" in reason

    # 3. Quoted text not in anchor
    mismatched_text_cp = CritiquePoint(
        anchor_ids=["sec_methodology_b01"],
        quoted_evidence="Adam optimizer with beta1=0.9 and beta2=0.999",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Adam settings mentioned.",
        confidence=CritiqueConfidence.HIGH,
    )
    is_valid, reason = critic._verify_critique_grounding(mismatched_text_cp, anchor_map)
    assert is_valid is False
    assert "not found" in reason


def test_methodology_critic_real_resnet(tmp_path):
    """Integration test verifying MethodologyCritic on resnet.pdf."""
    pdf_path = os.path.join("tests", "data", "resnet.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("resnet.pdf not found")

    out_dir = str(tmp_path / "resnet_critique_run")
    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    critic = MethodologyCritic()
    report = critic.critique_paper(struct, output_dir=out_dir)

    # Assertions
    assert report.schema_version == "1.0.0"
    assert "Residual" in report.paper_title
    assert len(report.target_sections) >= 4  # 3.1, 3.2, 3.3, 3.4
    assert len(report.critiques) >= 8

    # Verify dimensions covered
    dimensions = {c.critique_dimension for c in report.critiques}
    assert CritiqueDimension.REPRODUCIBILITY in dimensions
    assert CritiqueDimension.ASSUMPTIONS in dimensions
    assert CritiqueDimension.LIMITATIONS in dimensions
    assert CritiqueDimension.APPROPRIATENESS in dimensions

    # Verify every critique anchor matches actual text in struct
    block_map = {}
    for s in struct.sections:
        for cb in s.content_blocks:
            block_map[cb.id] = cb.text

    for cp in report.critiques:
        for aid in cp.anchor_ids:
            assert aid in block_map
            # Verify quote is a case-insensitive normalized substring
            norm_quote = " ".join(cp.quoted_evidence.lower().split())
            norm_actual = " ".join(block_map[aid].lower().split())
            # Check prefix words
            assert any(word in norm_actual for word in norm_quote.split()[:3])

    # Assert report saved to disk
    json_path = os.path.join(out_dir, "02_methodology_critic.json")
    assert os.path.exists(json_path)
    assert os.path.getsize(json_path) > 500
