import json
import os
import sys
sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from minions.critics.consensus import ConsensusAggregator
from minions.critics.methodology import MethodologyCritic
from minions.critics.scorer import MethodologyScorer
from minions.parser.pipeline import StructureParserPipeline
from scripts.test_resnet_stability import get_resnet_review_run


def run_phase3_resnet_scoring():
    pdf_path = "tests/data/resnet.pdf"
    out_dir = ".minions/runs/resnet_phase3_scoring"
    os.makedirs(out_dir, exist_ok=True)

    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    from scripts.test_resnet_stability import get_resnet_review_run

    # 1. Run N=3 Independent Critic Runs
    print("=" * 80)
    print("PHASE 3: CONSENSUS SCORING PIPELINE (RESNET.PDF)")
    print("=" * 80)
    print("Executing N=3 independent critic passes...")

    raw_reports = []
    for run_id in [1, 2, 3]:
        critic = MethodologyCritic(
            model="claude-3-5-sonnet-20241022",
            custom_llm_fn=lambda s, u, r=run_id: get_resnet_review_run(r),
        )
        report = critic.critique_paper(struct)
        raw_reports.append(report)
        print(f"  Run {run_id}: {len(report.critiques)} verified critiques, {len(report.strengths)} verified strengths.")

    # 2. Step 1: Consensus Classification (Core >=2/3, Secondary 1/3)
    aggregator = ConsensusAggregator(core_threshold_ratio=0.60)
    consensus_report = aggregator.aggregate_runs(raw_reports)

    print("\n--- STEP 1: CONSENSUS CLASSIFICATION ---")
    core_critiques = [c for c in consensus_report.critiques if c.consensus.value == "core"]
    sec_critiques = [c for c in consensus_report.critiques if c.consensus.value == "secondary"]
    core_strengths = [s for s in consensus_report.strengths if s.consensus.value == "core"]
    sec_strengths = [s for s in consensus_report.strengths if s.consensus.value == "secondary"]

    print(f"Total Critiques: {len(consensus_report.critiques)} ({len(core_critiques)} CORE, {len(sec_critiques)} SECONDARY)")
    print(f"Total Strengths: {len(consensus_report.strengths)} ({len(core_strengths)} CORE, {len(sec_strengths)} SECONDARY)\n")

    # 3. Step 2 & 3: Dual-Pass Rubric Scoring
    scorer = MethodologyScorer()
    scored_report = scorer.score_report(consensus_report, output_dir=out_dir)
    scores = scored_report.scores

    # 4. Print Full Results
    print("=" * 80)
    print("CONSENSUS-CLASSIFIED CRITIQUES ARRAY")
    print("=" * 80)
    for idx, c in enumerate(scored_report.critiques, 1):
        print(f"[{idx:02d}] [{c.consensus.value.upper()}] [{c.critique_dimension.value.upper()}] (Confidence: {c.confidence.value})")
        print(f"     Anchors: {c.anchor_ids}")
        print(f"     Quote  : \"{c.quoted_evidence}\"")
        print(f"     Text   : {c.critique_text}\n")

    print("=" * 80)
    print("CONSENSUS-CLASSIFIED STRENGTHS ARRAY")
    print("=" * 80)
    for idx, s in enumerate(scored_report.strengths, 1):
        print(f"[{idx:02d}] [{s.consensus.value.upper()}] [{s.critique_dimension.value.upper()}] (Confidence: {s.confidence.value})")
        print(f"     Anchors: {s.anchor_ids}")
        print(f"     Quote  : \"{s.quoted_evidence}\"")
        print(f"     Text   : {s.critique_text}\n")

    print("=" * 80)
    print("METHODOLOGY RUBRIC SCORES (1-5 SCALE)")
    print("=" * 80)
    dims = [
        ("Reproducibility", scores.reproducibility),
        ("Stated Assumptions", scores.assumptions),
        ("Acknowledged Limitations", scores.limitations),
        ("Method Appropriateness", scores.appropriateness),
    ]

    for dim_name, ds in dims:
        print(f"▶ {dim_name.upper()}: {ds.rating} / 5")
        print(f"  Anchors  : {ds.based_on_anchor_ids}")
        print(f"  Reasoning: {ds.reasoning}\n")

    print(f"OVERALL METHODOLOGY RATING: {scores.overall_rating} / 5.0")
    print(f"DUAL-PASS SCORING AGREEMENT: {scores.scoring_pass_agreement * 100:.1f}%")
    print("=" * 80)

    # Save to JSON
    json_path = os.path.join(out_dir, "03_methodology_scored.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(scored_report.model_dump_json(indent=2))
    print(f"Saved complete scored report to: {json_path}\n")

    return scored_report


if __name__ == "__main__":
    run_phase3_resnet_scoring()
