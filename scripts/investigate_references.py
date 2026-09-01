"""Investigative script for references anchors and structured fields."""

import fitz
import json
import re
from minions.parser.pipeline import StructureParserPipeline
from minions.schema.paper import CanonicalSectionType

def investigate_resnet_references():
    print("=" * 80)
    print("INVESTIGATION: REFERENCES SECTION IN RESNET.PDF")
    print("=" * 80)

    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf("tests/data/resnet.pdf", output_dir=".minions/runs/debug_ref_inv")

    ref_sec = next((s for s in struct.sections if s.canonical_type == CanonicalSectionType.REFERENCES), None)
    assert ref_sec is not None, "References section not found"

    print(f"Total Content Blocks in Reference Section: {len(ref_sec.content_blocks)}")
    print(f"Total Structured Reference Entries: {len(struct.references)}\n")

    doc = fitz.open("tests/data/resnet.pdf")

    print("--- SPOT CHECK: 10 CONSECUTIVE REFERENCE SECTION CONTENT BLOCKS ---")
    for i, cb in enumerate(ref_sec.content_blocks[20:30]):
        page = doc[cb.page_number - 1]
        bbox = cb.bbox
        # Query exact bbox without padding
        rect_exact = fitz.Rect(bbox)
        # Query with 0.5pt padding
        rect_pad = fitz.Rect(bbox[0]-0.5, bbox[1]-0.5, bbox[2]+0.5, bbox[3]+0.5)

        raw_exact = page.get_textbox(rect_exact).strip().replace("\n", " ")
        raw_pad = page.get_textbox(rect_pad).strip().replace("\n", " ")
        stored = cb.text.replace("\n", " ")

        print(f"[{i+21:02d}] Anchor: {cb.id} (Page {cb.page_number})")
        print(f"     BBox       : [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
        print(f"     Stored Text: {stored}")
        print(f"     Raw Exact  : {raw_exact}")
        print(f"     Raw Pad    : {raw_pad}")
        match = (raw_exact == stored) or (stored in raw_pad) or (raw_exact in stored)
        print(f"     Exact Match: {match}\n")

    print("--- SPOT CHECK: 10 STRUCTURED REFERENCE OBJECTS (Authors, Year, Title) ---")
    for i, ref in enumerate(struct.references[20:30]):
        print(f"Ref {ref.ref_id}:")
        print(f"  Raw Text: {ref.raw_text}")
        print(f"  Authors : {ref.authors}")
        print(f"  Year    : {ref.year}")
        print(f"  Title   : {ref.title}")
        print(f"  DOI/URL : {ref.doi_or_url}\n")

    doc.close()

if __name__ == "__main__":
    investigate_resnet_references()
