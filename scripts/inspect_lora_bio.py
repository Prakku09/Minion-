"""Inspect methodology text in LoRA and Bio papers with UTF-8."""

import json
import sys
from minions.parser.pipeline import StructureParserPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pipeline = StructureParserPipeline(dpi=150)

# LoRA
lora = pipeline.parse_pdf("tests/data/lora.pdf", output_dir=".minions/runs/lora_inspect")
meth_secs_lora = [s for s in lora.sections if s.canonical_type.value == "methodology"]
print("=" * 80)
print("LORA METHODOLOGY SECTIONS")
print("=" * 80)
for s in meth_secs_lora:
    print(f"\n[{s.id}] {s.heading_title} (Blocks: {len(s.content_blocks)})")
    for cb in s.content_blocks:
        text = cb.text.strip()
        if len(text) > 20:
            print(f"  [{cb.id}] (p.{cb.page_number}): {text[:120]}...")

# Bio
bio = pipeline.parse_pdf("tests/data/single_column_bio.pdf", output_dir=".minions/runs/bio_inspect")
meth_secs_bio = [s for s in bio.sections if s.canonical_type.value == "methodology"]
print("\n" + "=" * 80)
print("BIO PAPER METHODOLOGY SECTIONS")
print("=" * 80)
for s in meth_secs_bio:
    print(f"\n[{s.id}] {s.heading_title} (Blocks: {len(s.content_blocks)})")
    for cb in s.content_blocks:
        text = cb.text.strip()
        if len(text) > 20:
            print(f"  [{cb.id}] (p.{cb.page_number}): {text[:120]}...")
