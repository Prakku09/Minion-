"""Section segmentation, canonical type classification, and bibliography resolution."""

import re
from typing import Dict, List, Optional, Set, Tuple
from minions.parser.layout import RawBlock
from minions.schema.paper import (
    CanonicalSectionType,
    ContentBlock,
    ContentBlockType,
    Equation,
    Figure,
    Reference,
    Section,
    SectionParseConfidence,
    Table,
)


class SectionSegmenter:
    """Segments paper text blocks into canonical hierarchical sections and resolves entities."""

    # Heading classification rules
    CANONICAL_PATTERNS = [
        (CanonicalSectionType.ABSTRACT, re.compile(r"\b(?:abstract|summary)\b", re.IGNORECASE)),
        (
            CanonicalSectionType.INTRODUCTION,
            re.compile(r"\b(?:introduction|motivation|overview|background)\b", re.IGNORECASE),
        ),
        (
            CanonicalSectionType.RELATED_WORK,
            re.compile(
                r"\b(?:related\s+work|prior\s+work|literature\s+review|state\s+of\s+the\s+art|background\s+and\s+related\s+work)\b",
                re.IGNORECASE,
            ),
        ),
        (
            CanonicalSectionType.METHODOLOGY,
            re.compile(
                r"\b(?:method(?:ology)?|methods|proposed|architecture|model|approach|formulation|algorithm|system\s+design|implementation|training|framework|adaptation|deep\s+residual|diffusion)\b",
                re.IGNORECASE,
            ),
        ),
        (
            CanonicalSectionType.RESULTS,
            re.compile(
                r"\b(?:results?|experiments?|evaluation|ablation|empirical|benchmarks?|translation|classification)\b",
                re.IGNORECASE,
            ),
        ),
        (
            CanonicalSectionType.DISCUSSION,
            re.compile(
                r"\b(?:discussion|analysis|limitations?|broader\s+impacts?|threats\s+to\s+validity)\b",
                re.IGNORECASE,
            ),
        ),
        (
            CanonicalSectionType.CONCLUSION,
            re.compile(
                r"\b(?:conclusions?|concluding\s+remarks|summary\s+and\s+future\s+work|future\s+work)\b",
                re.IGNORECASE,
            ),
        ),
        (
            CanonicalSectionType.REFERENCES,
            re.compile(
                r"\b(?:references?|bibliography|literature\s+cited)\b", re.IGNORECASE
            ),
        ),
        (
            CanonicalSectionType.APPENDIX,
            re.compile(
                r"\b(?:appendix|supplement(?:ary)?(?:\s+material)?)\b", re.IGNORECASE
            ),
        ),
    ]

    # Numbered heading pattern e.g. "1 Introduction", "3.2 Model Architecture", "IV. EXPERIMENTS"
    HEADING_NUM_REGEX = re.compile(
        r"^(?:(?:\d+(?:\.\d+)*\.?)|(?:[IVXLCDM]+\.?)|(?:[A-Z]\.))\s+([A-Z].*)"
    )

    def __init__(self):
        pass

    def segment_document(
        self,
        ordered_blocks: List[RawBlock],
        figures: Dict[str, Figure],
        tables: Dict[str, Table],
        equations: Dict[str, Equation],
    ) -> Tuple[List[Section], List[Reference], Dict[str, SectionParseConfidence], int]:
        """Segment ordered layout blocks into canonical hierarchical sections."""
        # Calculate average font size across document for heading thresholding
        all_font_sizes = [b.avg_font_size for b in ordered_blocks if b.text.strip()]
        base_font_size = (
            sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 10.0
        )

        # Normalize composite blocks (where a heading was grouped with following paragraph)
        normalized_blocks: List[RawBlock] = []
        for b in ordered_blocks:
            if len(b.lines) > 1:
                first_line_text = b.lines[0].text.strip()
                is_h, _, _, _ = self._classify_text_as_heading(first_line_text, base_font_size, b.lines[0].spans)
                if is_h and len(b.lines) > 1:
                    h_block = RawBlock(
                        page_number=b.page_number,
                        block_index=b.block_index,
                        bbox=b.lines[0].bbox,
                        lines=[b.lines[0]],
                        text=first_line_text,
                        avg_font_size=b.lines[0].spans[0].font_size if b.lines[0].spans else b.avg_font_size,
                        is_bold=True,
                        column_index=b.column_index,
                    )
                    rem_text = " ".join(l.text for l in b.lines[1:]).strip()
                    rem_block = RawBlock(
                        page_number=b.page_number,
                        block_index=b.block_index,
                        bbox=(b.bbox[0], b.lines[1].bbox[1], b.bbox[2], b.bbox[3]),
                        lines=b.lines[1:],
                        text=rem_text,
                        avg_font_size=b.avg_font_size,
                        is_bold=b.is_bold,
                        column_index=b.column_index,
                    )
                    normalized_blocks.append(h_block)
                    normalized_blocks.append(rem_block)
                    continue
            normalized_blocks.append(b)

        # Step 1: Detect heading boundaries
        sections_data: List[Dict] = []
        unassigned_blocks: List[RawBlock] = []
        existing_section_ids: Set[str] = set()

        # Default pre-intro section (for title/metadata blocks before Abstract or Intro)
        current_section = {
            "id": "sec_preamble",
            "canonical_type": CanonicalSectionType.OTHER,
            "heading_title": "Preamble",
            "level": 1,
            "blocks": [],
        }
        existing_section_ids.add("sec_preamble")

        for block in normalized_blocks:
            is_heading, canonical_type, level, title = self._classify_heading(
                block, base_font_size
            )

            # If we are already in References, only allow Appendix to break out
            if current_section["canonical_type"] == CanonicalSectionType.REFERENCES:
                if not (is_heading and canonical_type == CanonicalSectionType.APPENDIX):
                    current_section["blocks"].append(block)
                    continue

            if is_heading:
                # Save previous section if it has content
                if current_section["blocks"] or current_section["id"] != "sec_preamble":
                    sections_data.append(current_section)

                sec_id = self._generate_section_id(
                    canonical_type, title, len(sections_data) + 1, existing_section_ids
                )
                current_section = {
                    "id": sec_id,
                    "canonical_type": canonical_type,
                    "heading_title": title,
                    "level": level,
                    "blocks": [],
                }
            else:
                current_section["blocks"].append(block)

        if current_section["blocks"] or current_section["id"] != "sec_preamble":
            sections_data.append(current_section)

        # Remove empty preamble if present
        if sections_data and sections_data[0]["id"] == "sec_preamble" and not sections_data[0]["blocks"]:
            sections_data.pop(0)

        # Step 2: Build Section models and ContentBlocks with anchor IDs
        sections: List[Section] = []
        section_confidences: Dict[str, SectionParseConfidence] = {}
        references: List[Reference] = []

        for sec_dict in sections_data:
            sec_id = sec_dict["id"]
            c_type = sec_dict["canonical_type"]
            heading = sec_dict["heading_title"]
            level = sec_dict["level"]
            raw_blocks: List[RawBlock] = sec_dict["blocks"]

            content_blocks: List[ContentBlock] = []
            text_pieces: List[str] = []

            for b_idx, rb in enumerate(raw_blocks):
                anchor_id = f"{sec_id}_b{b_idx + 1:02d}"
                block_type = self._determine_block_type(rb.text)
                cb = ContentBlock(
                    id=anchor_id,
                    type=block_type,
                    text=rb.text,
                    page_number=rb.page_number,
                    bbox=rb.bbox,
                )
                content_blocks.append(cb)
                text_pieces.append(rb.text)

            aggregated_text = "\n\n".join(text_pieces)

            # Check if this is the References section
            if c_type == CanonicalSectionType.REFERENCES:
                references = self._parse_references(aggregated_text, content_blocks)

            # Bind associated figures, tables, and equations
            sec_fig_ids = self._bind_entities_to_section(figures, raw_blocks, aggregated_text, sec_id)
            sec_tab_ids = self._bind_entities_to_section(tables, raw_blocks, aggregated_text, sec_id)
            sec_eq_ids = self._bind_entities_to_section(equations, raw_blocks, aggregated_text, sec_id)

            # Compute Section Confidence
            conf, issues = self._evaluate_section_confidence(c_type, content_blocks, heading)
            section_confidences[sec_id] = SectionParseConfidence(
                confidence=round(conf, 2),
                blocks_count=len(content_blocks),
                issues=issues,
            )

            section_obj = Section(
                id=sec_id,
                canonical_type=c_type,
                heading_title=heading,
                level=level,
                subsections=[],
                content_blocks=content_blocks,
                raw_text=aggregated_text,
                figure_ids=sec_fig_ids,
                table_ids=sec_tab_ids,
                equation_ids=sec_eq_ids,
            )
            sections.append(section_obj)

        unassigned_count = len(unassigned_blocks)
        return sections, references, section_confidences, unassigned_count

    def _classify_heading(
        self, block: RawBlock, base_font_size: float
    ) -> Tuple[bool, CanonicalSectionType, int, str]:
        """Determine if a block is a section header and classify its canonical type."""
        text = block.text.strip()
        lines = [line.text.strip() for line in block.lines if line.text.strip()]
        if not text or len(lines) > 2 or len(text) > 120:
            return False, CanonicalSectionType.OTHER, 1, ""

        all_spans = [span for line in block.lines for span in line.spans]
        return self._classify_text_as_heading(text, base_font_size, all_spans)

    def _classify_text_as_heading(
        self, text: str, base_font_size: float, spans: List[any]
    ) -> Tuple[bool, CanonicalSectionType, int, str]:
        """Classify given text as heading based on patterns, small-caps normalization, and font metrics."""
        text = text.strip()
        if not text or len(text) > 100:
            return False, CanonicalSectionType.OTHER, 1, ""

        # Normalize LaTeX small-caps spaced letters (e.g. 'A BSTRACT' -> 'ABSTRACT', '1 I NTRODUCTION' -> '1 INTRODUCTION')
        norm_text = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", text)
        norm_text = re.sub(r"(\w+)\s+-\s*(\w+)", r"\1-\2", norm_text)
        if re.match(r"^(?:[A-Z]\s+){3,}[A-Z]$", norm_text.strip()):
            norm_text = norm_text.replace(" ", "")

        # Reject standalone numbers, punctuation fragments, author footnotes, and math expressions
        if re.match(r"^\d+$", norm_text):  # standalone page/section number
            return False, CanonicalSectionType.OTHER, 1, ""
        if any(c in norm_text for c in ["∗", "*", "†", "‡", "§", "@", "{", "}", "=", "<", ">", "\\", "|", "#", "±", "%"]):
            return False, CanonicalSectionType.OTHER, 1, ""
        if re.search(r"\b(?:arxiv|preprint|in proc|proceedings|journal|vol\.|pp\.|https?:|www\.)\b", norm_text, re.I):
            return False, CanonicalSectionType.OTHER, 1, ""
        if norm_text.startswith("Figure ") or norm_text.startswith("Fig. ") or norm_text.startswith("Table ") or norm_text.startswith("Tab. "):
            return False, CanonicalSectionType.OTHER, 1, ""

        # Reject footnote narrative sentences starting with footnote numbers e.g. "4 We still need..."
        if re.match(r"^\d+\s+[A-Z][a-z]+", norm_text) and len(norm_text.split()) > 7:
            return False, CanonicalSectionType.OTHER, 1, ""

        avg_size = sum(s.font_size for s in spans) / len(spans) if spans else base_font_size
        is_bold = any(s.flags & 2 or "bold" in s.font_name.lower() for s in spans) if spans else False
        is_font_large = avg_size >= (base_font_size + 0.8)

        # Match canonical patterns first on normalized text
        for c_type, pattern in self.CANONICAL_PATTERNS:
            if pattern.search(norm_text) and len(norm_text.split()) <= 6:
                return True, c_type, 1, norm_text

        # Match numbered section headings e.g. "1 Introduction", "3.2 Attention", "IV. Experiments"
        num_match = re.match(
            r"^(?:(?:\d+(?:\.\d+)*\.?)|(?:[IVXLCDM]+\.?)|(?:[A-Z]\.))\s+([A-Za-z].*)$", norm_text
        )
        if num_match:
            heading_body = num_match.group(1).strip()
            # Do not classify trailing sentence fragments as headings
            if heading_body.endswith(".") and not heading_body.endswith("etc."):
                return False, CanonicalSectionType.OTHER, 1, ""

            matched_c_type = CanonicalSectionType.OTHER
            for c_type, pattern in self.CANONICAL_PATTERNS:
                if pattern.search(norm_text) or pattern.search(heading_body):
                    matched_c_type = c_type
                    break

            prefix = norm_text[: num_match.start(1)].strip()
            dot_count = prefix.count(".")
            level = min(3, dot_count + 1)
            return True, matched_c_type, level, norm_text

        # Match unnumbered prominent headings (must be bold or larger font and short title-cased)
        if (is_font_large or is_bold) and len(norm_text.split()) <= 6 and not norm_text.endswith((".", ",", ";")):
            for c_type, pattern in self.CANONICAL_PATTERNS:
                if pattern.search(norm_text):
                    return True, c_type, 1, norm_text

        return False, CanonicalSectionType.OTHER, 1, ""

    def _determine_block_type(self, text: str) -> ContentBlockType:
        """Categorize content block type."""
        text = text.strip()
        if re.match(r"^(?:[•\-\*]|\d+\.)\s+", text):
            return ContentBlockType.LIST_ITEM
        if text.startswith("Figure ") or text.startswith("Fig. "):
            return ContentBlockType.FIGURE_REF
        if text.startswith("Table ") or text.startswith("Tab. "):
            return ContentBlockType.TABLE_REF
        return ContentBlockType.PARAGRAPH

    def _generate_section_id(
        self, canonical_type: CanonicalSectionType, title: str, index: int, existing_ids: Set[str]
    ) -> str:
        """Generate deterministic, unique human-readable section ID."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
        slug = slug[:25] if slug else f"sec_{index}"

        if canonical_type != CanonicalSectionType.OTHER:
            base_id = f"sec_{canonical_type.value}"
        else:
            base_id = f"sec_{slug}"

        candidate_id = base_id
        if candidate_id in existing_ids:
            candidate_id = f"{base_id}_{slug}"
        if candidate_id in existing_ids:
            candidate_id = f"{candidate_id}_{index}"

        existing_ids.add(candidate_id)
        return candidate_id

    def _bind_entities_to_section(
        self,
        entities: Dict[str, any],
        raw_blocks: List[RawBlock],
        section_text: str,
        section_id: str,
    ) -> List[str]:
        """Bind figures, tables, or equations to the section where they appear or are referenced."""
        bound_ids: Set[str] = set()
        if not raw_blocks:
            return []

        sec_page_numbers = set(b.page_number for b in raw_blocks)

        for ent_id, ent in entities.items():
            # Check if entity is on the same page as this section
            if ent.page_number in sec_page_numbers:
                bound_ids.add(ent_id)
                if not getattr(ent, "associated_section_id", None):
                    ent.associated_section_id = section_id

            # Check if entity label is mentioned in section text e.g. "Figure 1", "Table 2"
            label = getattr(ent, "label", None)
            if label and label.lower() in section_text.lower():
                bound_ids.add(ent_id)

        return sorted(list(bound_ids))

    def _parse_references(
        self, references_text: str, content_blocks: List[ContentBlock]
    ) -> List[Reference]:
        """Parse raw references block into structured Reference objects."""
        ref_entries: List[Reference] = []

        # Find all reference items matching "[1]", "[2]" or "1.", "2."
        # Use lookahead regex to segment into items
        raw_entries = re.split(r"(?=\[\d+\]|(?:\n|^)\s*\d+\.\s+)", references_text)
        cleaned_entries = [e.strip() for e in raw_entries if e.strip() and len(e.strip()) > 10]

        if len(cleaned_entries) >= 2:
            for idx, entry in enumerate(cleaned_entries):
                # Extract marker
                marker_match = re.match(r"^(\[\d+\]|\d+\.)\s*", entry)
                marker = marker_match.group(1).strip() if marker_match else f"[{idx + 1}]"
                parsed_ref = self._extract_structured_reference_fields(marker, entry)
                ref_entries.append(parsed_ref)
        else:
            # Fallback: segment by content blocks
            for idx, cb in enumerate(content_blocks):
                if len(cb.text.strip()) > 10:
                    marker_match = re.match(r"^(\[\d+\]|\d+\.)\s*", cb.text.strip())
                    marker = marker_match.group(1).strip() if marker_match else f"[{idx + 1}]"
                    parsed_ref = self._extract_structured_reference_fields(marker, cb.text.strip())
                    ref_entries.append(parsed_ref)

        return ref_entries

    def _extract_structured_reference_fields(self, ref_id: str, raw_text: str) -> Reference:
        """Extract structured authors, year, title, and venue from reference text."""
        # Year extraction (4 digits between 1900 and 2099)
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", raw_text)
        year = int(year_match.group(1)) if year_match else None

        # Title extraction heuristic (often in quotes or before journal/venue)
        title = None
        quoted = re.search(r'["“]([^"”]+)["”]', raw_text)
        if quoted:
            title = quoted.group(1).strip()

        # URL / DOI extraction
        url_match = re.search(r"(https?://\S+|doi:\S+)", raw_text, re.IGNORECASE)
        doi_or_url = url_match.group(1) if url_match else None

        # Authors heuristic (text before year or title)
        authors = []
        author_part = raw_text.split(str(year))[0] if year else raw_text[:80]
        # Clean markers
        author_part = re.sub(r"^\[\d+\]\s*|\d+\.\s*", "", author_part).strip()
        if author_part:
            raw_authors = re.split(r",| and | & ", author_part)
            authors = [a.strip() for a in raw_authors if len(a.strip()) > 2][:8]

        return Reference(
            ref_id=ref_id,
            raw_text=raw_text,
            title=title,
            authors=authors,
            year=year,
            venue=None,
            doi_or_url=doi_or_url,
        )

    def _evaluate_section_confidence(
        self, canonical_type: CanonicalSectionType, blocks: List[ContentBlock], heading: str
    ) -> Tuple[float, List[str]]:
        """Calculate confidence score and list any extraction issues for this section."""
        issues: List[str] = []
        conf = 1.0

        if not blocks:
            issues.append("Section has zero content blocks.")
            conf -= 0.40
            return max(0.0, conf), issues

        total_words = sum(len(b.text.split()) for b in blocks)
        if total_words < 20 and canonical_type not in (CanonicalSectionType.ABSTRACT, CanonicalSectionType.OTHER):
            issues.append(f"Section text is unusually short ({total_words} words).")
            conf -= 0.15

        if canonical_type == CanonicalSectionType.OTHER:
            issues.append(f"Unmapped custom heading '{heading}'.")
            conf -= 0.10

        # Check for unhyphenated split fragments
        broken_count = sum(1 for b in blocks if b.text.endswith("-"))
        if broken_count > 2:
            issues.append(f"{broken_count} blocks end with trailing hyphens.")
            conf -= 0.10

        return max(0.0, min(1.0, conf)), issues
