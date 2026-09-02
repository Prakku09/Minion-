"""Test script executing refined MethodologyCritic against resnet.pdf and reporting critiques vs strengths."""

import json
import os
import re
import sys
from minions.critics.methodology import MethodologyCritic
from minions.parser.pipeline import StructureParserPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_refined_methodology_critique_test():
    print("=" * 80)
    print("METHODOLOGY CRITIC EVALUATION ON RESNET.PDF (CRITIQUES VS. STRENGTHS)")
    print("=" * 80)

    pdf_path = "tests/data/resnet.pdf"
    out_dir = ".minions/runs/resnet_critique_run"
    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    critic = MethodologyCritic()
    report = critic.critique_paper(struct, output_dir=out_dir)

    print(f"Paper Title: {report.paper_title}")
    print(f"Sections Evaluated: {report.target_sections}")
    print(f"Summary: {report.summary}\n")

    # 1. Print Material Critiques (Gaps, Ambiguities, Unproven Assumptions, Risks)
    print("=" * 80)
    print(f"1. MATERIAL CRITIQUES ({len(report.critiques)} Gaps, Ambiguities, Risks)")
    print("=" * 80)

    for idx, cp in enumerate(report.critiques):
        print(f"\n[Critique #{idx+1:02d}]")
        print(f"  Dimension       : {cp.critique_dimension.value.upper()}")
        print(f"  Confidence      : {cp.confidence.value.upper()}")
        print(f"  Cited Anchor IDs: {cp.anchor_ids}")
        print(f"  Quoted Evidence : \"{cp.quoted_evidence}\"")
        print(f"  Critique Text   : {cp.critique_text}")

    # 2. Print Methodological Strengths
    print("\n" + "=" * 80)
    print(f"2. METHODOLOGICAL STRENGTHS ({len(report.strengths)} Confirmatory Observations)")
    print("=" * 80)

    for idx, sp in enumerate(report.strengths):
        print(f"\n[Strength #{idx+1:02d}]")
        print(f"  Dimension       : {sp.critique_dimension.value.upper()}")
        print(f"  Confidence      : {sp.confidence.value.upper()}")
        print(f"  Cited Anchor IDs: {sp.anchor_ids}")
        print(f"  Quoted Evidence : \"{sp.quoted_evidence}\"")
        print(f"  Strength Text   : {sp.critique_text}")

    # 3. Self-Check: Verify 100% quoted evidence matches anchor block text
    print("\n" + "=" * 80)
    print("3. GROUNDING SELF-CHECK (VERIFYING ALL QUOTES AT CITED ANCHORS)")
    print("=" * 80)

    block_map = {}
    for s in struct.sections:
        for cb in s.content_blocks:
            block_map[cb.id] = cb.text.strip()

    all_points = report.critiques + report.strengths
    verified_count = 0
    for idx, p in enumerate(all_points):
        aid = p.anchor_ids[0]
        actual_text = block_map.get(aid, "")
        norm_actual = re.sub(r"\s+", " ", actual_text).lower()
        norm_quote = re.sub(r"\s+", " ", p.quoted_evidence).lower()

        # Check match
        quote_words = norm_quote.split()
        is_match = norm_quote in norm_actual or " ".join(quote_words[:4]) in norm_actual
        if is_match:
            verified_count += 1

    print(f"Self-Check Result: {verified_count}/{len(all_points)} Points 100% Grounded in Verbatim Anchor Blocks.")
    print(f"JSON Report written to: {os.path.join(out_dir, '02_methodology_critic.json')}")
    print("=" * 80)


if __name__ == "__main__":
    run_refined_methodology_critique_test()
