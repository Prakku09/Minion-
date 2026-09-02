"""Unit and integration tests for LLM-Driven MethodologyCritic agent."""

import os
import pytest
from minions.critics.methodology import MethodologyCritic
from minions.parser.pipeline import StructureParserPipeline
from minions.schema.critique import (
    CritiqueConfidence,
    CritiqueDimension,
    CritiquePoint,
    DroppedCritiquePoint,
    MethodologyCritiqueReport,
)


def test_critique_schema_validation():
    """Verify CritiquePoint, DroppedCritiquePoint and MethodologyCritiqueReport validation."""
    cp = CritiquePoint(
        anchor_ids=["sec_methodology_b01"],
        quoted_evidence="We adopt batch normalization right after each convolution",
        critique_dimension=CritiqueDimension.APPROPRIATENESS,
        critique_text="Appropriate normalization placement ensures stable training dynamics.",
        confidence=CritiqueConfidence.HIGH,
    )
    dp = DroppedCritiquePoint(
        point=CritiquePoint(
            anchor_ids=["sec_nonexistent_b99"],
            quoted_evidence="hallucinated quote",
            critique_dimension=CritiqueDimension.REPRODUCIBILITY,
            critique_text="Hallucinated text",
            confidence=CritiqueConfidence.HIGH,
        ),
        drop_reason="Anchor ID does not exist",
    )
    report = MethodologyCritiqueReport(
        schema_version="1.0.0",
        paper_title="Deep Residual Learning",
        target_sections=["sec_methodology"],
        critiques=[cp],
        strengths=[],
        dropped_points=[dp],
        hallucination_rate=0.5,
        summary="Test summary",
    )

    data = report.model_dump()
    assert data["schema_version"] == "1.0.0"
    assert len(data["critiques"]) == 1
    assert len(data["dropped_points"]) == 1
    assert data["hallucination_rate"] == 0.5
    assert data["critiques"][0]["critique_dimension"] == "appropriateness"
    assert data["critiques"][0]["confidence"] == "high"


def test_grounding_rejection_of_hallucinated_anchors():
    """Verify that ungrounded or mismatched points are strictly dropped into dropped_points."""
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

    # 2. Invalid Anchor ID
    invalid_aid_cp = CritiquePoint(
        anchor_ids=["sec_nonexistent_b99"],
        quoted_evidence="SGD with mini-batch size of 256",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Specific optimizer settings are clearly provided.",
        confidence=CritiqueConfidence.HIGH,
    )

    # 3. Quoted text not in anchor
    mismatched_text_cp = CritiquePoint(
        anchor_ids=["sec_methodology_b01"],
        quoted_evidence="Adam optimizer with beta1=0.9 and beta2=0.999",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Adam settings mentioned.",
        confidence=CritiqueConfidence.HIGH,
    )

    validated, dropped = critic._verify_and_filter_points(
        [valid_cp, invalid_aid_cp, mismatched_text_cp], anchor_map
    )

    assert len(validated) == 1
    assert validated[0] == valid_cp
    assert len(dropped) == 2
    assert dropped[0].point == invalid_aid_cp
    assert "does not exist" in dropped[0].drop_reason
    assert dropped[1].point == mismatched_text_cp
    assert "not found" in dropped[1].drop_reason


def test_methodology_critic_real_resnet(tmp_path):
    """Integration test verifying MethodologyCritic on resnet.pdf."""
    pdf_path = os.path.join("tests", "data", "resnet.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("resnet.pdf not found")

    out_dir = str(tmp_path / "resnet_critique_run")
    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    from scripts.run_llm_critic_all_papers import get_claude_review_for_paper

    critic = MethodologyCritic(
        model="claude-3-5-sonnet-20241022",
        custom_llm_fn=lambda s, u: get_claude_review_for_paper(struct.metadata.title, u),
    )
    report = critic.critique_paper(struct, output_dir=out_dir)

    assert report.schema_version == "1.0.0"
    assert "Residual" in report.paper_title
    assert len(report.target_sections) >= 4
    assert len(report.critiques) >= 6
    assert len(report.strengths) >= 4

    # Check that intentional hallucinated point was caught and dropped
    assert len(report.dropped_points) >= 1
    assert any("Adam optimizer" in dp.point.quoted_evidence for dp in report.dropped_points)

    # Assert report saved to disk
    json_path = os.path.join(out_dir, "02_methodology_critic.json")
    assert os.path.exists(json_path)
    assert os.path.getsize(json_path) > 500
