"""Inspect all methodology content blocks and entities in ResNet."""

import sys
from minions.parser.pipeline import StructureParserPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pipeline = StructureParserPipeline(dpi=150)
struct = pipeline.parse_pdf("tests/data/resnet.pdf", output_dir=".minions/runs/resnet_meth_inspect")

meth_secs = [s for s in struct.sections if s.canonical_type.value == "methodology"]

for s in meth_secs:
    print("=" * 80)
    print(f"SECTION: [{s.id}] {s.heading_title} (Level {s.level})")
    print(f"Entities bound: Figures={s.figure_ids}, Tables={s.table_ids}, Equations={s.equation_ids}")
    print("=" * 80)
    for cb in s.content_blocks:
        text = cb.text.strip()
        if len(text) > 20:
            print(f"[{cb.id}] (Page {cb.page_number}, bbox={cb.bbox}):")
            print(f"  {text}\n")
