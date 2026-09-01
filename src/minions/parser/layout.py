"""Layout analysis, multi-column detection, and reading-order sorting for PDF documents."""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple
import fitz  # PyMuPDF


@dataclass
class RawSpan:
    """A span of text with font metrics and bounding box."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_name: str
    font_size: float
    flags: int  # font flags: bold, italic, serif, etc.
    color: int


@dataclass
class RawLine:
    """A line of text consisting of spans."""
    spans: List[RawSpan] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    text: str = ""


@dataclass
class RawBlock:
    """A layout block (paragraph, heading, caption, etc.) extracted from a page."""
    page_number: int
    block_index: int
    bbox: Tuple[float, float, float, float]
    lines: List[RawLine] = field(default_factory=list)
    text: str = ""
    block_type: str = "text"  # 'text', 'heading', 'caption', 'table', 'equation'
    avg_font_size: float = 10.0
    is_bold: bool = False
    column_index: int = 0  # 0=span/full-width, 1=left col, 2=right col, etc.


class LayoutEngine:
    """Extracts layout blocks from PDF pages and orders them in natural reading sequence."""

    def __init__(self, column_gap_threshold: float = 15.0):
        self.column_gap_threshold = column_gap_threshold

    def extract_page_blocks(self, page: fitz.Page, page_number: int) -> List[RawBlock]:
        """Extract all text blocks from a page, detect columns, and sort in reading order."""
        page_dict = page.get_text("dict", flags=fitz.TEXT_DEHYPHENATE)
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        raw_blocks: List[RawBlock] = []
        for b_idx, block_dict in enumerate(page_dict.get("blocks", [])):
            if block_dict.get("type") != 0:  # only text blocks
                continue

            lines: List[RawLine] = []
            block_text_pieces: List[str] = []
            font_sizes: List[float] = []
            bold_count = 0
            total_spans = 0

            for line_dict in block_dict.get("lines", []):
                spans: List[RawSpan] = []
                line_text_pieces: List[str] = []
                for span_dict in line_dict.get("spans", []):
                    text = span_dict.get("text", "")
                    if not text.strip():
                        continue
                    font_size = float(span_dict.get("size", 10.0))
                    flags = int(span_dict.get("flags", 0))
                    is_span_bold = bool(flags & 2 or "bold" in span_dict.get("font", "").lower())
                    if is_span_bold:
                        bold_count += 1
                    total_spans += 1
                    font_sizes.append(font_size)

                    span = RawSpan(
                        text=text,
                        bbox=tuple(span_dict.get("bbox", (0, 0, 0, 0))),
                        font_name=span_dict.get("font", ""),
                        font_size=font_size,
                        flags=flags,
                        color=int(span_dict.get("color", 0)),
                    )
                    spans.append(span)
                    line_text_pieces.append(text)

                if spans:
                    line_bbox = tuple(line_dict.get("bbox", (0, 0, 0, 0)))
                    line_text = " ".join(line_text_pieces).strip()
                    lines.append(RawLine(spans=spans, bbox=line_bbox, text=line_text))
                    block_text_pieces.append(line_text)

            if not lines:
                continue

            full_block_text = self._clean_block_text(" ".join(block_text_pieces))
            if not full_block_text:
                continue

            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10.0
            is_bold = (bold_count / total_spans) > 0.4 if total_spans else False

            raw_block = RawBlock(
                page_number=page_number,
                block_index=b_idx,
                bbox=tuple(block_dict.get("bbox", (0, 0, 0, 0))),
                lines=lines,
                text=full_block_text,
                avg_font_size=avg_font_size,
                is_bold=is_bold,
            )
            raw_blocks.append(raw_block)

        # Detect columns and sort blocks in natural reading order
        ordered_blocks = self.order_blocks_reading_sequence(raw_blocks, page_width, page_height)
        return ordered_blocks

    def order_blocks_reading_sequence(
        self, blocks: List[RawBlock], page_width: float, page_height: float
    ) -> List[RawBlock]:
        """Detect multi-column gutters and sort blocks: top full-width, left column, right column, bottom full-width."""
        if not blocks:
            return []

        # Margin detection
        min_x0 = min(b.bbox[0] for b in blocks)
        max_x1 = max(b.bbox[2] for b in blocks)
        content_width = max_x1 - min_x0
        mid_x = min_x0 + content_width / 2.0

        # Check if page is multi-column:
        # A 2-column page has blocks distinctly occupying left half (< mid_x - 10) and right half (> mid_x + 10)
        left_blocks = [b for b in blocks if b.bbox[2] <= mid_x + 20 and (b.bbox[2] - b.bbox[0]) < content_width * 0.7]
        right_blocks = [b for b in blocks if b.bbox[0] >= mid_x - 20 and (b.bbox[2] - b.bbox[0]) < content_width * 0.7]
        span_blocks = [
            b for b in blocks if (b.bbox[2] - b.bbox[0]) >= content_width * 0.7 or (b.bbox[0] < mid_x and b.bbox[2] > mid_x)
        ]

        is_multi_column = (len(left_blocks) >= 2 and len(right_blocks) >= 2)

        if not is_multi_column:
            # Single-column: sort purely top-to-bottom by y0, breaking ties by x0
            return sorted(blocks, key=lambda b: (round(b.bbox[1] / 5.0) * 5.0, b.bbox[0]))

        # Multi-column ordering:
        # Partition into horizontal bands divided by spanning blocks (like title, abstract, full-width figures)
        # Sort spanning blocks by y0
        span_y_cuts: List[Tuple[float, float, RawBlock]] = []
        for sb in span_blocks:
            span_y_cuts.append((sb.bbox[1], sb.bbox[3], sb))

        span_y_cuts.sort(key=lambda x: x[0])

        ordered: List[RawBlock] = []
        current_y = 0.0

        # Group column blocks between spanning blocks
        for y_top, y_bot, sb in span_y_cuts:
            # All column blocks between current_y and y_top
            inter_left = [b for b in left_blocks if b.bbox[1] >= current_y - 5 and b.bbox[3] <= y_top + 10]
            inter_right = [b for b in right_blocks if b.bbox[1] >= current_y - 5 and b.bbox[3] <= y_top + 10]

            # Sort top-to-bottom within left column, then right column
            inter_left.sort(key=lambda b: b.bbox[1])
            inter_right.sort(key=lambda b: b.bbox[1])

            for b in inter_left:
                b.column_index = 1
                ordered.append(b)
            for b in inter_right:
                b.column_index = 2
                ordered.append(b)

            sb.column_index = 0
            ordered.append(sb)
            current_y = max(current_y, y_bot)

        # Any remaining column blocks below the last spanning block
        rem_left = [b for b in left_blocks if b not in ordered]
        rem_right = [b for b in right_blocks if b not in ordered]
        rem_span = [b for b in span_blocks if b not in ordered]

        rem_left.sort(key=lambda b: b.bbox[1])
        rem_right.sort(key=lambda b: b.bbox[1])
        rem_span.sort(key=lambda b: b.bbox[1])

        for b in rem_left:
            b.column_index = 1
            ordered.append(b)
        for b in rem_right:
            b.column_index = 2
            ordered.append(b)
        for b in rem_span:
            b.column_index = 0
            ordered.append(b)

        return ordered

    @staticmethod
    def _clean_block_text(text: str) -> str:
        """Clean hyphenation and whitespace irregularities."""
        # Fix soft/hard hyphenation at line breaks (e.g. "repre-\nsentation" -> "representation")
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)  # if dehyphenate left spacing
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
