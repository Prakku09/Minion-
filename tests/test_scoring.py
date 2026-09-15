"""Unit and integration tests for Consensus Aggregator and Methodology Scorer."""

import pytest
from minions.critics.consensus import ConsensusAggregator
from minions.critics.scorer import MethodologyScorer
from minions.schema.critique import (
    ConsensusType,
    CritiqueConfidence,
    CritiqueDimension,
    CritiquePoint,
    MethodologyCritiqueReport,
)


def _make_report(critiques, strengths, title="Test Paper"):
    return MethodologyCritiqueReport(
        schema_version="1.0.0",
        paper_title=title,
        target_sections=["sec_methodology"],
        critiques=critiques,
        strengths=strengths,
        dropped_points=[],
        hallucination_rate=0.0,
    )


def test_consensus_aggregation_core_vs_secondary():
    """Verify that points in >=2/3 runs are classified as CORE and in 1/3 as SECONDARY."""
    cp_shared = CritiquePoint(
        anchor_ids=["sec_m_b01"],
        quoted_evidence="SGD with batch size 256",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Plateau schedule lacks patience threshold.",
        confidence=CritiqueConfidence.HIGH,
    )
    cp_run1_only = CritiquePoint(
        anchor_ids=["sec_m_b02"],
        quoted_evidence="10-crop testing",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="10-crop testing creates test-time confounding.",
        confidence=CritiqueConfidence.HIGH,
    )
    sp_shared = CritiquePoint(
        anchor_ids=["sec_m_b03"],
        quoted_evidence="Batch Normalization right after each convolution",
        critique_dimension=CritiqueDimension.APPROPRIATENESS,
        critique_text="Sound BN placement stabilizes signals.",
        confidence=CritiqueConfidence.HIGH,
    )

    r1 = _make_report(critiques=[cp_shared, cp_run1_only], strengths=[sp_shared])
    r2 = _make_report(critiques=[cp_shared], strengths=[sp_shared])
    r3 = _make_report(critiques=[], strengths=[sp_shared])

    aggregator = ConsensusAggregator()
    consensus_report = aggregator.aggregate_runs([r1, r2, r3])

    assert consensus_report.runs_evaluated_count == 3
    assert len(consensus_report.critiques) == 2
    assert len(consensus_report.strengths) == 1

    # cp_shared was in Run 1 and Run 2 (2/3) -> CORE
    c_shared = next(c for c in consensus_report.critiques if c.anchor_ids == ["sec_m_b01"])
    assert c_shared.consensus == ConsensusType.CORE

    # cp_run1_only was in Run 1 only (1/3) -> SECONDARY
    c_sec = next(c for c in consensus_report.critiques if c.anchor_ids == ["sec_m_b02"])
    assert c_sec.consensus == ConsensusType.SECONDARY

    # sp_shared was in Run 1, 2, 3 (3/3) -> CORE
    s_shared = consensus_report.strengths[0]
    assert s_shared.consensus == ConsensusType.CORE


def test_scorer_requires_real_llm_dual_pass_when_no_api_is_configured():
    """A real dual-pass scoring run must fail loudly if no API-backed scorer is configured."""
    cp_core = CritiquePoint(
        anchor_ids=["sec_m_b01"],
        quoted_evidence="SGD with batch size 256",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Plateau schedule lacks patience threshold.",
        confidence=CritiqueConfidence.HIGH,
        consensus=ConsensusType.CORE,
    )
    sp_core = CritiquePoint(
        anchor_ids=["sec_m_b03"],
        quoted_evidence="Batch Normalization right after each convolution",
        critique_dimension=CritiqueDimension.APPROPRIATENESS,
        critique_text="Sound BN placement stabilizes signals.",
        confidence=CritiqueConfidence.HIGH,
        consensus=ConsensusType.CORE,
    )

    report = _make_report(critiques=[cp_core], strengths=[sp_core])
    scorer = MethodologyScorer()

    with pytest.raises(RuntimeError, match="API-backed dual-pass scoring is required"):
        scorer.score_report(report)


def test_dual_pass_scoring_uses_two_distinct_llm_calls_per_dimension():
    """Pass 1 and Pass 2 must be genuinely independent LLM calls per dimension."""
    cp1 = CritiquePoint(
        anchor_ids=["sec_m_b01"],
        quoted_evidence="SGD with batch size 256",
        critique_dimension=CritiqueDimension.REPRODUCIBILITY,
        critique_text="Plateau schedule lacks patience threshold.",
        confidence=CritiqueConfidence.HIGH,
        consensus=ConsensusType.CORE,
    )
    cp2 = CritiquePoint(
        anchor_ids=["sec_m_b02"],
        quoted_evidence="The learning rate is annealed by 0.1 every 30 epochs",
        critique_dimension=CritiqueDimension.ASSUMPTIONS,
        critique_text="Assumption of monotonic training dynamics is untested.",
        confidence=CritiqueConfidence.MEDIUM,
        consensus=ConsensusType.CORE,
    )
    sp1 = CritiquePoint(
        anchor_ids=["sec_m_b03"],
        quoted_evidence="Batch Normalization right after each convolution",
        critique_dimension=CritiqueDimension.APPROPRIATENESS,
        critique_text="Sound BN placement stabilizes signals.",
        confidence=CritiqueConfidence.HIGH,
        consensus=ConsensusType.CORE,
    )
    sp2 = CritiquePoint(
        anchor_ids=["sec_m_b04"],
        quoted_evidence="We use a single held-out validation set",
        critique_dimension=CritiqueDimension.LIMITATIONS,
        critique_text="Validation procedure is well specified.",
        confidence=CritiqueConfidence.HIGH,
        consensus=ConsensusType.CORE,
    )

    calls = []

    def fake_llm(pass_id, dimension, evidence, prompt):
        calls.append((pass_id, dimension.value, prompt[:80]))
        rating = 4 if pass_id == 1 else 3
        return {
            "rating": rating,
            "reasoning": f"Pass {pass_id} reasoning for {dimension.value} references [{evidence['anchor_ids'][0]}].",
            "based_on_anchor_ids": evidence["anchor_ids"],
        }

    report = _make_report(critiques=[cp1, cp2], strengths=[sp1, sp2])
    scorer = MethodologyScorer(custom_llm_fn=fake_llm)
    scored_report = scorer.score_report(report)

    assert scored_report.scores is not None
    assert len(calls) == 8
    assert sum(1 for p, _, _ in calls if p == 1) == 4
    assert sum(1 for p, _, _ in calls if p == 2) == 4
    assert scored_report.scores.pass_1_scores is not None
    assert scored_report.scores.pass_2_scores is not None
    assert scored_report.scores.exact_match_rate is not None
    assert scored_report.scores.near_miss_rate is not None
