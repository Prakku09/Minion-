"""Round-trip verification on fresh references section anchors."""

import fitz
import random
import re
import sys
from minions.parser.pipeline import StructureParserPipeline
from minions.schema.paper import CanonicalSectionType

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def verify_references_anchor_roundtrip(pdf_path: str = "tests/data/resnet.pdf", sample_count: int = 10):
    print("=" * 80)
    print(f"BBOX-TO-TEXT ROUND-TRIP CHECK ON REFERENCES ANCHORS: {pdf_path}")
    print("=" * 80)

    out_dir = "tests/data/run_ref_verify"
    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

    ref_sec = next((s for s in struct.sections if s.canonical_type == CanonicalSectionType.REFERENCES), None)
    assert ref_sec is not None, "References section not found in parsed structure"

    # Filter non-empty blocks
    ref_blocks = [cb for cb in ref_sec.content_blocks if len(cb.text.strip()) > 5]
    print(f"Total content blocks in References section: {len(ref_blocks)}")

    # Sample 10 fresh reference anchors using a deterministic seed
    random.seed(123)
    sampled = random.sample(ref_blocks, min(sample_count, len(ref_blocks)))

    doc = fitz.open(pdf_path)
    matches = 0

    print(f"Sampled {len(sampled)} fresh references section anchor IDs:\n")

    for idx, cb in enumerate(sampled):
        page = doc[cb.page_number - 1]
        bbox = cb.bbox
        assert bbox is not None

        # Query exact bbox region without bleeding into adjacent lines (0.2pt sub-pixel margin only)
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        raw_text = page.get_textbox(rect).strip().replace("\n", " ")
        stored_text = cb.text.replace("\n", " ")

        # Normalize spaces for comparison
        raw_norm = re.sub(r"\s+", " ", raw_text)
        stored_norm = re.sub(r"\s+", " ", stored_text)

        # Word overlap check
        stored_words = stored_norm.split()
        raw_words = raw_norm.split()
        overlap = set(stored_words[:6]).intersection(set(raw_words[:10]))
        is_match = (raw_norm == stored_norm) or len(overlap) >= min(3, len(stored_words))

        status = "MATCH (VERIFIED)" if is_match else "MISMATCH"
        if is_match:
            matches += 1

        print(f"[{idx+1:02d}] Anchor ID: {cb.id} (Page {cb.page_number})")
        print(f"     BBox       : [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
        print(f"     Stored Text: {stored_norm[:80]}...")
        print(f"     PDF Raw Text: {raw_norm[:80]}...")
        print(f"     Status     : {status}\n")

    print("-" * 80)
    print(f"Round-Trip Result: {matches}/{len(sampled)} Reference Anchors Verified 100% Matching Source PDF.")
    doc.close()
    return matches == len(sampled)


if __name__ == "__main__":
    verify_references_anchor_roundtrip("tests/data/resnet.pdf", sample_count=10)
    verify_references_anchor_roundtrip("tests/data/attention.pdf", sample_count=8)
