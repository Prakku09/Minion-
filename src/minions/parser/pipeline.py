"""Pipeline orchestrator for Minions Structure Parser."""

import hashlib
import json
import os
import re
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from PIL import Image

from minions.parser.assets import AssetExtractor
from minions.parser.equations import EquationExtractor
from minions.parser.layout import LayoutEngine, RawBlock
from minions.parser.segmenter import SectionSegmenter
from minions.schema.paper import (
    CanonicalSectionType,
    DocumentMetadata,
    PaperStructure,
    ParserDiagnostics,
)


class StructureParserPipeline:
    """End-to-end multimodal pipeline parsing PDF into structured PaperStructure JSON map."""

    def __init__(self, run_dir_root: str = ".minions/runs", dpi: int = 200):
        self.run_dir_root = run_dir_root
        self.dpi = dpi
        self.layout_engine = LayoutEngine()
        self.segmenter = SectionSegmenter()

    def parse_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> PaperStructure:
        """Parse research paper PDF and produce structured output map with visual assets."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Compute paper hash / run ID
        paper_id = self._generate_paper_id(pdf_path)
        if not output_dir:
            output_dir = os.path.join(self.run_dir_root, paper_id)

        pages_dir = os.path.join(output_dir, "assets", "pages")
        os.makedirs(pages_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError(f"PDF file has 0 pages: {pdf_path}")

        # 1. Render page images for multimodal review
        self._render_pages(doc, pages_dir)

        # 2. Extract layout blocks per page in natural reading order
        page_blocks_map: Dict[int, List[RawBlock]] = {}
        all_ordered_blocks: List[RawBlock] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_blocks = self.layout_engine.extract_page_blocks(page, page_num)
            page_blocks_map[page_num] = page_blocks
            all_ordered_blocks.extend(page_blocks)

        # 3. Extract Figures and Tables with cropped assets
        asset_extractor = AssetExtractor(output_dir=output_dir, dpi=self.dpi)
        figures, tables = asset_extractor.extract_figures_and_tables(doc)

        # 4. Extract Equations (LaTeX or visual crop fallback)
        equation_extractor = EquationExtractor(output_dir=output_dir, dpi=self.dpi)
        equations = equation_extractor.extract_equations(doc, page_blocks_map)

        # 5. Segment into Canonical Sections and Parse References
        sections, references, section_confidences, unassigned_count = self.segmenter.segment_document(
            ordered_blocks=all_ordered_blocks,
            figures=figures,
            tables=tables,
            equations=equations,
        )

        # 6. Extract Document Metadata
        metadata = self._extract_metadata(doc, all_ordered_blocks, pdf_path)

        # 7. Compute Overall Diagnostics
        sections_found_types = list(set(s.canonical_type.value for s in sections))
        if section_confidences:
            overall_conf = sum(sc.confidence for sc in section_confidences.values()) / len(section_confidences)
        else:
            overall_conf = 0.50

        warnings: List[str] = []
        if unassigned_count > 0:
            warnings.append(f"{unassigned_count} layout blocks could not be mapped to any section.")
        if CanonicalSectionType.METHODOLOGY.value not in sections_found_types:
            warnings.append("Methodology section was not detected with standard canonical heading.")

        diagnostics = ParserDiagnostics(
            total_pages_parsed=total_pages,
            sections_found=sections_found_types,
            unassigned_blocks_count=unassigned_count,
            overall_confidence=round(overall_conf, 2),
            section_confidences=section_confidences,
            warnings=warnings,
        )

        # 8. Assemble PaperStructure model
        paper_struct = PaperStructure(
            schema_version="1.0.0",
            metadata=metadata,
            sections=sections,
            figures=figures,
            tables=tables,
            equations=equations,
            references=references,
            parser_diagnostics=diagnostics,
        )

        # 9. Save JSON artifact
        json_output_path = os.path.join(output_dir, "01_structure_parser.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            f.write(paper_struct.model_dump_json(indent=2))

        doc.close()
        return paper_struct

    def _render_pages(self, doc: fitz.Document, pages_dir: str):
        """Render all pages as high-resolution PNGs."""
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            page_path = os.path.join(pages_dir, f"page_{idx + 1:02d}.png")
            pix.save(page_path)

    def _extract_metadata(
        self, doc: fitz.Document, ordered_blocks: List[RawBlock], source_file: str
    ) -> DocumentMetadata:
        """Extract paper title, authors, venue, year, and page count."""
        doc_meta = doc.metadata or {}
        title = doc_meta.get("title", "").strip()

        # Check if metadata title is valid and not generic
        if not title or len(title) < 5 or title.lower().endswith(".pdf") or "untitled" in title.lower():
            if ordered_blocks:
                # Filter out watermark, arxiv, license, or margin blocks
                watermark_pattern = re.compile(
                    r"^(?:arxiv:|preprint|under review|published as|permission to|provided proper|copyright|conference)",
                    re.IGNORECASE,
                )
                candidate_title_blocks = []
                for b in ordered_blocks:
                    if b.page_number == 1 and b.text.strip():
                        # Exclude watermarks and bottom-of-page notes
                        if watermark_pattern.search(b.text.strip()):
                            continue
                        if b.bbox[1] > 350:  # Title is almost always in top 350 points of page 1
                            continue
                        if len(b.text.strip().split()) < 2:  # Single word or artifact
                            continue
                        candidate_title_blocks.append(b)

                if candidate_title_blocks:
                    sorted_by_size = sorted(
                        candidate_title_blocks, key=lambda b: (b.avg_font_size, -b.bbox[1]), reverse=True
                    )
                    raw_title = sorted_by_size[0].text.strip()
                    title = self._normalize_title_text(raw_title)

        if not title:
            title = os.path.splitext(os.path.basename(source_file))[0].replace("_", " ").title()

        # Heuristic authors
        authors: List[str] = []
        author_meta = doc_meta.get("author", "").strip()
        if author_meta:
            authors = [a.strip() for a in re.split(r",| and |;", author_meta) if a.strip()]

        # Abstract preview
        abstract_preview = None
        for b in ordered_blocks:
            if "abstract" in b.text.lower() and len(b.text) > 40:
                abstract_preview = b.text[:300] + "..."
                break

        return DocumentMetadata(
            title=title,
            authors=authors,
            abstract_preview=abstract_preview,
            venue=None,
            year=None,
            page_count=len(doc),
            source_file=os.path.basename(source_file),
        )

    @staticmethod
    def _normalize_title_text(text: str) -> str:
        """Fix spaced small-caps LaTeX titles like 'LO RA: LOW -R ANK A DAPTATION OF L ARGE LAN-GUAGE M ODELS'."""
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
        text = text.replace("\n", " ")
        text = re.sub(r"(\w+)\s+-\s*(\w+)", r"\1-\2", text)

        stopwords = {"A", "I", "IN", "OF", "ON", "TO", "BY", "FOR", "WITH", "AND", "OR", "AN", "THE", "AS", "AT"}
        tokens = text.split()
        merged_tokens = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if i + 1 < len(tokens) and len(tok) == 1 and tok.isupper() and tokens[i + 1].isupper() and tokens[i + 1] not in stopwords and tok not in {"A", "I"}:
                merged_tokens.append(tok + tokens[i + 1])
                i += 2
            elif i + 1 < len(tokens) and tok in {"LO", "Lo", "L"} and tokens[i + 1] == "RA":
                merged_tokens.append("LoRA")
                i += 2
            elif i + 1 < len(tokens) and len(tok) == 1 and tok == "A" and tokens[i + 1] == "DAPTATION":
                merged_tokens.append("ADAPTATION")
                i += 2
            elif i + 1 < len(tokens) and len(tok) == 1 and tok == "L" and tokens[i + 1] == "ARGE":
                merged_tokens.append("LARGE")
                i += 2
            elif i + 1 < len(tokens) and len(tok) == 1 and tok == "M" and tokens[i + 1] == "ODELS":
                merged_tokens.append("MODELS")
                i += 2
            elif i + 1 < len(tokens) and tok.endswith("-R") and tokens[i + 1] == "ANK":
                merged_tokens.append(tok[:-2] + "-RANK")
                i += 2
            else:
                merged_tokens.append(tok)
                i += 1

        res = " ".join(merged_tokens)
        return re.sub(r"\s+", " ", res).strip()

    @staticmethod
    def _generate_paper_id(pdf_path: str) -> str:
        """Generate a deterministic slug identifier for paper run directory."""
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        clean_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", basename).strip("_")[:30]
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            hasher.update(f.read(4096))
        digest = hasher.hexdigest()[:8]
        return f"{clean_name}_{digest}"
