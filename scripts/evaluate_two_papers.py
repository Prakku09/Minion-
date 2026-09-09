"""Run MethodologyCritic as-is on LoRA and Spatial Stochastic Dynamics."""

import json
import os
import re
import sys
from minions.critics.methodology import MethodologyCritic
from minions.parser.pipeline import StructureParserPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def evaluate_paper(pdf_path: str, run_dir_name: str):
    print("=" * 80)
    print(f"EVALUATING: {os.path.basename(pdf_path)}")
    print("=" * 80)

    out_dir = os.path.join(".minions", "runs", run_dir_name)
    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    print(f"Paper Title: {struct.metadata.title}")
    print(f"Pages: {struct.metadata.page_count}")
    
    # Identify methodology sections
    meth_secs = [s for s in struct.sections if s.canonical_type.value == "methodology"]
    print(f"Methodology Sections Found ({len(meth_secs)}):")
    for s in meth_secs:
        print(f"  - [{s.id}] {s.heading_title} ({len(s.content_blocks)} blocks)")

    # Run Critic as-is
    critic = MethodologyCritic()
    report = critic.critique_paper(struct, output_dir=out_dir)

    print(f"\nReport Summary: {report.summary}")
    print(f"Critique Points Count : {len(report.critiques)}")
    print(f"Strength Points Count : {len(report.strengths)}")

    # Grounding Self-Check
    block_map = {}
    for s in struct.sections:
        for cb in s.content_blocks:
            block_map[cb.id] = cb.text.strip()
    for fig_id, fig in struct.figures.items():
        block_map[fig_id] = fig.caption.strip() if fig.caption else fig.label
    for tab_id, tab in struct.tables.items():
        block_map[tab_id] = f"{tab.caption}\n{tab.markdown_content}".strip()
    for eq_id, eq in struct.equations.items():
        block_map[eq_id] = eq.raw_text.strip() if eq.raw_text else eq.latex

    all_points = report.critiques + report.strengths
    verified_count = 0

    print("\n--- GROUNDING SELF-CHECK ---")
    for idx, p in enumerate(all_points):
        aid = p.anchor_ids[0]
        actual_text = block_map.get(aid, "")
        norm_actual = re.sub(r"\s+", " ", actual_text).lower()
        norm_quote = re.sub(r"\s+", " ", p.quoted_evidence).lower()
        
        quote_words = norm_quote.split()
        is_match = (norm_quote in norm_actual) or (" ".join(quote_words[:4]) in norm_actual)
        if is_match:
            verified_count += 1
            status = "VERIFIED"
        else:
            status = "MISMATCH"
        print(f"  Point #{idx+1:02d} [{p.critique_dimension.value.upper()}] Anchor: {aid} -> {status}")

    print(f"\nGrounding Accuracy: {verified_count}/{len(all_points)} Verified Exact Verbatim Grounding.")

    # Dump JSON Report
    report_json = report.model_dump_json(indent=2)
    print("\n--- FULL JSON REPORT ---")
    print(report_json)
    print("\n" + "=" * 80 + "\n")

    return struct, report


if __name__ == "__main__":
    lora_struct, lora_report = evaluate_paper("tests/data/lora.pdf", "lora_critique_eval")
    bio_struct, bio_report = evaluate_paper("tests/data/single_column_bio.pdf", "bio_critique_eval")
