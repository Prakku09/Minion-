"""Visual asset and tabular extraction: figures, plots, diagrams, and Markdown tables."""

import os
import re
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image
from minions.schema.paper import Figure, Table


class AssetExtractor:
    """Extracts figures and tables, generates cropped images, and parses tabular grids into Markdown."""

    FIG_CAPTION_REGEX = re.compile(
        r"^(?:(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?))\s*[:.]\s*(.+)", re.IGNORECASE | re.DOTALL
    )
    TAB_CAPTION_REGEX = re.compile(
        r"^(?:(?:Table|Tab\.?)\s*(\d+[a-zA-Z]?))\s*[:.]\s*(.+)", re.IGNORECASE | re.DOTALL
    )

    def __init__(self, output_dir: str, dpi: int = 200):
        self.output_dir = output_dir
        self.dpi = dpi
        self.fig_dir = os.path.join(output_dir, "assets", "figures")
        self.tab_dir = os.path.join(output_dir, "assets", "tables")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.tab_dir, exist_ok=True)

    def extract_figures_and_tables(
        self, doc: fitz.Document
    ) -> Tuple[Dict[str, Figure], Dict[str, Table]]:
        """Extract all figures and tables across document pages."""
        figures: Dict[str, Figure] = {}
        tables: Dict[str, Table] = {}

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Extract tables using PyMuPDF table finder
            page_tables = self._extract_tables_from_page(page, page_num)
            for tab_id, table in page_tables.items():
                tables[tab_id] = table

            # 2. Extract figures using image and drawing bounding box clustering + captions
            page_figures = self._extract_figures_from_page(page, page_num, existing_tables=tables)
            for fig_id, figure in page_figures.items():
                figures[fig_id] = figure

        return figures, tables

    def _extract_tables_from_page(self, page: fitz.Page, page_num: int) -> Dict[str, Table]:
        """Extract tabular structures and captions from a single page."""
        tables_found: Dict[str, Table] = {}
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])

        # Find captions matching Table pattern
        caption_blocks = []
        for b in blocks:
            if b.get("type") != 0:
                continue
            text = " ".join(
                span.get("text", "")
                for line in b.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            match = self.TAB_CAPTION_REGEX.match(text)
            if match:
                caption_blocks.append((match.group(1), text, b.get("bbox")))

        # Use PyMuPDF find_tables
        try:
            tab_finder = page.find_tables()
            tab_list = tab_finder.tables if tab_finder else []
        except Exception:
            tab_list = []

        # If native find_tables found tables
        for idx, tab in enumerate(tab_list):
            tab_bbox = tab.bbox  # (x0, y0, x1, y1)
            markdown_grid = self._table_to_markdown(tab.extract())
            if not markdown_grid.strip():
                continue

            # Match with closest table caption
            matched_label = f"Table {idx + 1}"
            matched_caption = f"Table {idx + 1}"
            tab_num_str = str(idx + 1)

            for cap_num, cap_text, cap_bbox in caption_blocks:
                # Caption is typically directly above or below table
                if abs(cap_bbox[1] - tab_bbox[3]) < 40 or abs(tab_bbox[1] - cap_bbox[3]) < 40:
                    matched_label = f"Table {cap_num}"
                    matched_caption = cap_text
                    tab_num_str = cap_num
                    break

            tab_id = f"tab_{tab_num_str.lower()}"
            crop_path = self._crop_region(page, tab_bbox, self.tab_dir, f"{tab_id}.png")

            tables_found[tab_id] = Table(
                id=tab_id,
                label=matched_label,
                caption=matched_caption,
                page_number=page_num,
                bbox=tuple(tab_bbox),
                markdown_content=markdown_grid,
                image_path=crop_path,
            )

        # Also handle tables identified by caption even if native grid detector missed boundaries
        for cap_num, cap_text, cap_bbox in caption_blocks:
            tab_id = f"tab_{cap_num.lower()}"
            if tab_id not in tables_found:
                # Estimate table bbox near caption
                est_bbox = (
                    cap_bbox[0],
                    cap_bbox[3],
                    min(cap_bbox[0] + 400, page.rect.width - 20),
                    min(cap_bbox[3] + 200, page.rect.height - 20),
                )
                crop_path = self._crop_region(page, est_bbox, self.tab_dir, f"{tab_id}.png")
                tables_found[tab_id] = Table(
                    id=tab_id,
                    label=f"Table {cap_num}",
                    caption=cap_text,
                    page_number=page_num,
                    bbox=est_bbox,
                    markdown_content="[Table visual extracted - see image]",
                    image_path=crop_path,
                )

        return tables_found

    def _extract_figures_from_page(
        self, page: fitz.Page, page_num: int, existing_tables: Dict[str, Table]
    ) -> Dict[str, Figure]:
        """Extract figures (diagrams, plots, architectures) and associate with captions."""
        figures_found: Dict[str, Figure] = {}
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])

        # Collect figure captions
        caption_blocks = []
        for b in blocks:
            if b.get("type") != 0:
                continue
            text = " ".join(
                span.get("text", "")
                for line in b.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            match = self.FIG_CAPTION_REGEX.match(text)
            if match:
                caption_blocks.append((match.group(1), text, b.get("bbox")))

        # Get images on page
        images = page.get_images(full=True)
        image_rects = []
        for img_info in images:
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                for r in rects:
                    # Ignore tiny decorative icons/rules
                    if r.width > 30 and r.height > 30:
                        image_rects.append(r)
            except Exception:
                continue

        # Match each caption to the best visual bounding box
        for fig_num, cap_text, cap_bbox in caption_blocks:
            fig_id = f"fig_{fig_num.lower()}"
            matched_bbox = None

            # Look for image rect directly above or near the caption
            for r in image_rects:
                if (
                    r.y1 <= cap_bbox[1] + 15
                    and (cap_bbox[1] - r.y1) < 400
                    and abs(r.x0 - cap_bbox[0]) < 200
                ):
                    matched_bbox = (
                        min(r.x0, cap_bbox[0]),
                        r.y0,
                        max(r.x1, cap_bbox[2]),
                        cap_bbox[3],
                    )
                    break

            if not matched_bbox:
                # Estimate visual region above the caption
                est_y0 = max(20.0, cap_bbox[1] - 250.0)
                matched_bbox = (
                    max(20.0, cap_bbox[0] - 20.0),
                    est_y0,
                    min(page.rect.width - 20.0, cap_bbox[2] + 20.0),
                    cap_bbox[3] + 10.0,
                )

            crop_path = self._crop_region(page, matched_bbox, self.fig_dir, f"{fig_id}.png")

            figures_found[fig_id] = Figure(
                id=fig_id,
                label=f"Figure {fig_num}",
                caption=cap_text,
                page_number=page_num,
                bbox=tuple(matched_bbox),
                image_path=crop_path,
                visual_summary=None,
            )

        # Also capture standalone large images if any captions were not detected
        if not caption_blocks and image_rects:
            for idx, r in enumerate(image_rects):
                if r.width > 150 and r.height > 100:
                    fig_id = f"fig_p{page_num}_{idx + 1}"
                    crop_path = self._crop_region(page, tuple(r), self.fig_dir, f"{fig_id}.png")
                    figures_found[fig_id] = Figure(
                        id=fig_id,
                        label=f"Figure (Page {page_num})",
                        caption=f"Visual asset extracted from page {page_num}",
                        page_number=page_num,
                        bbox=tuple(r),
                        image_path=crop_path,
                    )

        return figures_found

    def _crop_region(
        self, page: fitz.Page, bbox: Tuple[float, float, float, float], out_dir: str, filename: str
    ) -> str:
        """Render and crop a high-resolution sub-region of a page."""
        rect = fitz.Rect(bbox)
        # Clamp to page bounds
        rect.intersect(page.rect)
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            rect = page.rect

        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        filepath = os.path.join(out_dir, filename)
        pix.save(filepath)
        # Return relative path for portability
        rel_path = os.path.relpath(filepath, self.output_dir).replace("\\", "/")
        return rel_path

    @staticmethod
    def _table_to_markdown(raw_table_data: Optional[List[List[Optional[str]]]]) -> str:
        """Convert 2D table array into clean Markdown table format."""
        if not raw_table_data or len(raw_table_data) < 1:
            return ""

        cleaned_rows: List[List[str]] = []
        for row in raw_table_data:
            if not row:
                continue
            cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
            if any(cleaned_row):
                cleaned_rows.append(cleaned_row)

        if not cleaned_rows:
            return ""

        max_cols = max(len(r) for r in cleaned_rows)
        # Pad shorter rows
        for r in cleaned_rows:
            while len(r) < max_cols:
                r.append("")

        header = cleaned_rows[0]
        divider = ["---"] * max_cols
        data_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
        ]
        for r in data_rows:
            md_lines.append("| " + " | ".join(r) + " |")

        return "\n".join(md_lines)
