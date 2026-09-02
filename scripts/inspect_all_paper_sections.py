"""Inspect section trees across all 4 test papers."""

import sys
from minions.parser.pipeline import StructureParserPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pipeline = StructureParserPipeline(dpi=150)
papers = [
    ("resnet.pdf", "tests/data/resnet.pdf"),
    ("attention.pdf", "tests/data/attention.pdf"),
    ("lora.pdf", "tests/data/lora.pdf"),
    ("single_column_bio.pdf", "tests/data/single_column_bio.pdf"),
]

for name, path in papers:
    print("=" * 80)
    print(f"PAPER: {name}")
    print("=" * 80)
    struct = pipeline.parse_pdf(path, output_dir=f".minions/runs/inspect_{name}")
    meth_count = 0
    for s in struct.sections:
        is_meth = s.canonical_type.value == "methodology"
        if is_meth:
            meth_count += 1
        mark = "--> [METHODOLOGY]" if is_meth else f"    [{s.canonical_type.value:12s}]"
        print(f"{mark} L{s.level} | {s.id:35s} | {s.heading_title} ({len(s.content_blocks)} blocks)")
    print(f"Total Methodology Sections in {name}: {meth_count}\n")
