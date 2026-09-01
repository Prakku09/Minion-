"""Command line interface for Minions Structure Parser."""

import argparse
import sys
from minions.parser.pipeline import StructureParserPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Minions Structure Parser: Extract grounded structural map from scientific PDF."
    )
    parser.add_argument("pdf_path", help="Path to research paper PDF")
    parser.add_argument(
        "--out", "-o", default=None, help="Output directory for structured JSON and assets"
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Rendering DPI for visual assets (default: 200)"
    )

    args = parser.parse_args()

    print(f"[Minions] Initializing Structure Parser for: {args.pdf_path}")
    pipeline = StructureParserPipeline(dpi=args.dpi)
    try:
        struct = pipeline.parse_pdf(args.pdf_path, output_dir=args.out)
        print(f"[Minions] Successfully parsed paper: {struct.metadata.title}")
        print(f"[Minions] Pages: {struct.metadata.page_count}")
        print(f"[Minions] Sections identified ({len(struct.sections)}): {[s.canonical_type.value for s in struct.sections]}")
        print(f"[Minions] Figures extracted: {len(struct.figures)}")
        print(f"[Minions] Tables extracted: {len(struct.tables)}")
        print(f"[Minions] Equations extracted: {len(struct.equations)}")
        print(f"[Minions] Overall Parse Confidence: {struct.parser_diagnostics.overall_confidence * 100:.1f}%")
    except Exception as e:
        print(f"[Minions Error] Failed to parse PDF: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
