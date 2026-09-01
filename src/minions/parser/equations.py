"""Equation extraction engine with LaTeX reconstruction and visual image fallback."""

import os
import re
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
from minions.parser.layout import RawBlock
from minions.schema.paper import Equation


class EquationExtractor:
    """Extracts mathematical equations with LaTeX reconstruction and visual fallback under threshold."""

    LATEX_CONFIDENCE_THRESHOLD = 0.80

    # Common equation number patterns e.g. (1), (2.1), (A.3), [Eq. 4]
    EQ_NUM_REGEX = re.compile(r"^\s*\((\d+(?:\.\d+)?|[a-zA-Z]\.?\d+)\)\s*$|(?:\((\d+)\)\s*$)")

    # Math symbol indicators
    MATH_SYMBOLS = set(
        [
            "∑", "∫", "∏", "∂", "√", "±", "≤", "≥", "≠", "≈", "∈", "∉", "⊂", "⊆",
            "∪", "∩", "∧", "∨", "→", "⇒", "↔", "⇔", "∀", "∃", "∇", "∞", "∝",
            "α", "β", "γ", "δ", "ϵ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
            "ν", "ξ", "π", "ρ", "σ", "τ", "υ", "ϕ", "φ", "χ", "ψ", "ω",
            "Γ", "Δ", "Θ", "Λ", "Ξ", "Π", "Σ", "Υ", "Φ", "Ψ", "Ω",
            "argmax", "argmin", "softmax", "log", "exp", "sin", "cos", "tan",
        ]
    )

    def __init__(self, output_dir: str, dpi: int = 200):
        self.output_dir = output_dir
        self.dpi = dpi
        self.eq_dir = os.path.join(output_dir, "assets", "equations")
        os.makedirs(self.eq_dir, exist_ok=True)

    def extract_equations(
        self, doc: fitz.Document, page_blocks: Dict[int, List[RawBlock]]
    ) -> Dict[str, Equation]:
        """Scan page blocks for standalone equations, reconstruct LaTeX or generate image crop."""
        equations: Dict[str, Equation] = {}
        eq_counter = 1

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            blocks = page_blocks.get(page_num, [])

            for block in blocks:
                is_math, eq_label, confidence, latex_candidate = self._evaluate_math_block(block)
                if not is_math:
                    continue

                eq_num_str = eq_label if eq_label else str(eq_counter)
                eq_id = f"eq_{eq_counter}"
                eq_counter += 1

                label_formatted = f"({eq_label})" if eq_label and not eq_label.startswith("(") else eq_label

                if confidence >= self.LATEX_CONFIDENCE_THRESHOLD and latex_candidate:
                    # High confidence LaTeX extraction
                    equations[eq_id] = Equation(
                        id=eq_id,
                        label=label_formatted,
                        latex=latex_candidate,
                        raw_text=block.text,
                        page_number=page_num,
                        bbox=block.bbox,
                        is_image_fallback=False,
                        latex_confidence=round(confidence, 2),
                        image_path=None,
                    )
                else:
                    # Visual Fallback: crop the exact equation bounding box
                    crop_path = self._crop_equation_region(page, block.bbox, f"{eq_id}.png")
                    equations[eq_id] = Equation(
                        id=eq_id,
                        label=label_formatted,
                        latex=latex_candidate if confidence > 0.4 else None,
                        raw_text=block.text,
                        page_number=page_num,
                        bbox=block.bbox,
                        is_image_fallback=True,
                        latex_confidence=round(confidence, 2),
                        image_path=crop_path,
                    )

        return equations

    def _evaluate_math_block(
        self, block: RawBlock
    ) -> Tuple[bool, Optional[str], float, Optional[str]]:
        """Evaluate if block is an equation, calculate LaTeX confidence and generate formula."""
        text = block.text.strip()
        if not text or len(text) > 300:  # Long paragraphs are not standalone equations
            return False, None, 0.0, None

        # Check for equation label
        eq_label = None
        # Match label at end of text e.g. "E = mc^2  (1)"
        label_match = re.search(r"\((\d+(?:\.\d+)?|[a-zA-Z]\.?\d+)\)\s*$", text)
        clean_text = text
        if label_match:
            eq_label = label_match.group(1)
            clean_text = text[: label_match.start()].strip()

        # Check for narrative prose indicators
        prose_words = {
            "let", "denote", "where", "consider", "assume", "define", "we", "the", "in",
            "this", "section", "figure", "table", "method", "model", "paper", "show", "is", "are"
        }
        words = set(re.findall(r"\b[a-zA-Z]{2,}\b", clean_text.lower()))
        prose_overlap = words.intersection(prose_words)

        # Standalone equations should have very few regular English prose words unless explicit label exists
        if not eq_label and len(prose_overlap) >= 2:
            return False, None, 0.0, None

        # Math indicators
        has_math_symbol = any(sym in clean_text for sym in self.MATH_SYMBOLS)
        has_math_ops = bool(re.search(r"[=<>+\-*/^_]{2,}|\\frac|\\sqrt|\\sum|\\prod", clean_text))
        has_equals = "=" in clean_text or "≈" in clean_text or "≤" in clean_text or "≥" in clean_text

        if not (has_equals and (has_math_symbol or has_math_ops or eq_label)):
            return False, None, 0.0, None

        # Check bracket balancing
        brackets_balanced = (
            clean_text.count("(") == clean_text.count(")")
            and clean_text.count("[") == clean_text.count("]")
            and clean_text.count("{") == clean_text.count("}")
        )

        # Estimate LaTeX confidence
        confidence = 0.50
        if brackets_balanced:
            confidence += 0.20
        if has_math_symbol or has_math_ops:
            confidence += 0.15
        if eq_label:
            confidence += 0.10

        # Penalize broken words or unmapped non-ascii glyphs
        unprintable = len([c for c in clean_text if ord(c) > 1000 and c not in self.MATH_SYMBOLS])
        if unprintable > 2:
            confidence -= 0.30

        confidence = max(0.0, min(1.0, confidence))

        # Build candidate LaTeX
        latex_candidate = self._reconstruct_latex(clean_text)

        return True, eq_label, confidence, latex_candidate

    def _reconstruct_latex(self, text: str) -> str:
        """Convert standard Unicode and text math symbols to LaTeX math syntax."""
        replacements = [
            ("∑", r"\sum "),
            ("∫", r"\int "),
            ("∏", r"\prod "),
            ("∂", r"\partial "),
            ("√", r"\sqrt "),
            ("±", r"\pm "),
            ("≤", r"\le "),
            ("≥", r"\ge "),
            ("≠", r"\ne "),
            ("≈", r"\approx "),
            ("∈", r"\in "),
            ("∉", r"\notin "),
            ("⊂", r"\subset "),
            ("⊆", r"\subseteq "),
            ("∪", r"\cup "),
            ("∩", r"\cap "),
            ("→", r"\to "),
            ("⇒", r"\implies "),
            ("∀", r"\forall "),
            ("∃", r"\exists "),
            ("∇", r"\nabla "),
            ("∞", r"\infty "),
            ("α", r"\alpha "),
            ("β", r"\beta "),
            ("γ", r"\gamma "),
            ("δ", r"\delta "),
            ("θ", r"\theta "),
            ("λ", r"\lambda "),
            ("μ", r"\mu "),
            ("σ", r"\sigma "),
        ]
        res = text
        for u, ltx in replacements:
            res = res.replace(u, ltx)

        return res.strip()

    def _crop_equation_region(
        self, page: fitz.Page, bbox: Tuple[float, float, float, float], filename: str
    ) -> str:
        """Crop and save equation image for fallback."""
        rect = fitz.Rect(bbox)
        # Add slight padding around equation
        rect.x0 = max(0, rect.x0 - 5)
        rect.y0 = max(0, rect.y0 - 5)
        rect.x1 = min(page.rect.width, rect.x1 + 5)
        rect.y1 = min(page.rect.height, rect.y1 + 5)

        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        filepath = os.path.join(self.eq_dir, filename)
        pix.save(filepath)
        return os.path.relpath(filepath, self.output_dir).replace("\\", "/")
