"""Test script executing MethodologyCritic against resnet.pdf and performing grounding self-checks."""

import json
import os
import re
import sys
from minions.critics.methodology import MethodologyCritic
from minions.parser.pipeline import StructureParserPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_methodology_critique_test():
    print("=" * 80)
    print("PHASE 2: METHODOLOGY CRITIC EVALUATION ON RESNET.PDF")
    print("=" * 80)

    # 1. Parse Paper Structure
    pdf_path = "tests/data/resnet.pdf"
    out_dir = ".minions/runs/resnet_critique_run"
    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    # 2. Run Methodology Critic Agent
    critic = MethodologyCritic()
    report = critic.critique_paper(struct, output_dir=out_dir)

    print(f"Paper Title: {report.paper_title}")
    print(f"Target Methodology Sections Evaluated: {report.target_sections}")
    print(f"Total Critique Points Generated: {len(report.critiques)}\n")

    # 3. Print Full List of Generated Critiques
    print("=" * 80)
    print("1. FULL LIST OF GENERATED METHODOLOGY CRITIQUES")
    print("=" * 80)

    for idx, cp in enumerate(report.critiques):
        print(f"\n[Critique #{idx+1:02d}]")
        print(f"  Dimension       : {cp.critique_dimension.value.upper()}")
        print(f"  Confidence      : {cp.confidence.value.upper()}")
        print(f"  Cited Anchor IDs: {cp.anchor_ids}")
        print(f"  Quoted Evidence : \"{cp.quoted_evidence}\"")
        print(f"  Critique Text   : {cp.critique_text}")

    # 4. Self-Check: Verify quoted_evidence matches anchor text in struct
    print("\n" + "=" * 80)
    print("2. GROUNDING SELF-CHECK: CONFIRMING QUOTED EVIDENCE AT CITED ANCHORS")
    print("=" * 80)

    # Build map of all blocks
    block_map = {}
    for s in struct.sections:
        for cb in s.content_blocks:
            block_map[cb.id] = cb.text.strip()

    verified_count = 0
    for idx, cp in enumerate(report.critiques):
        anchor_id = cp.anchor_ids[0]
        actual_block_text = block_map.get(anchor_id, "")
        
        # Clean whitespaces
        norm_actual = re.sub(r"\s+", " ", actual_block_text)
        norm_quote = re.sub(r"\s+", " ", cp.quoted_evidence)

        # Check if quote is contained in anchor block text
        is_exact_match = norm_quote.lower() in norm_actual.lower()

        print(f"Critique #{idx+1:02d} -> Anchor: {anchor_id}")
        if is_exact_match:
            verified_count += 1
            print("  Status: EXACT VERBATIM SUBSTRING MATCH (VERIFIED)")
        else:
            print("  Status: MISMATCH")
            print(f"    Expected inside: \"{norm_actual[:100]}...\"")
            print(f"    Quoted text    : \"{norm_quote}\"")

    print("-" * 80)
    print(f"Grounding Self-Check Result: {verified_count}/{len(report.critiques)} Critiques 100% Verbatim Grounded in Cited Anchor Blocks.")

    # 5. Review Flagged / Medium / Low Confidence Critiques
    print("\n" + "=" * 80)
    print("3. FLAGGED / NON-HIGH CONFIDENCE CRITIQUES REVIEW")
    print("=" * 80)
    flagged = [cp for cp in report.critiques if cp.confidence.value != "high"]
    if flagged:
        print(f"Found {len(flagged)} critique(s) marked with Medium/Low confidence for explicit user inspection:")
        for idx, cp in enumerate(flagged):
            print(f"\n  [Flagged #{idx+1}] Dimension: {cp.critique_dimension.value} (Confidence: {cp.confidence.value})")
            print(f"  Anchor : {cp.anchor_ids}")
            print(f"  Reason : {cp.critique_text}")
    else:
        print("All critiques have HIGH confidence.")

    print("\n" + "=" * 80)
    print(f"JSON Report written to: {os.path.join(out_dir, '02_methodology_critic.json')}")
    print("=" * 80)


if __name__ == "__main__":
    run_methodology_critique_test()
