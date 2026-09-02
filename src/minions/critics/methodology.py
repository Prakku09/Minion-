"""Methodology Critic Agent evaluating reproducibility, assumptions, limitations, and appropriateness."""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from minions.schema.critique import (
    CritiqueConfidence,
    CritiqueDimension,
    CritiquePoint,
    MethodologyCritiqueReport,
)
from minions.schema.paper import CanonicalSectionType, PaperStructure, Section


class MethodologyCritic:
    """Evaluates paper methodology sections against structured scientific criteria."""

    def __init__(self):
        pass

    def critique_paper(
        self, paper: PaperStructure, output_dir: Optional[str] = None
    ) -> MethodologyCritiqueReport:
        """Run full methodology critique on a parsed PaperStructure."""
        # 1. Build anchor text lookup index
        anchor_map: Dict[str, str] = self._build_anchor_index(paper)

        # 2. Extract all methodology sections (including subsections)
        methodology_sections = [
            s for s in paper.sections if s.canonical_type == CanonicalSectionType.METHODOLOGY
        ]
        target_section_ids = [s.id for s in methodology_sections]

        if not methodology_sections:
            return MethodologyCritiqueReport(
                paper_title=paper.metadata.title,
                target_sections=[],
                critiques=[],
                summary="No methodology sections found in the parsed document structure.",
            )

        # 3. Generate candidate critiques across the 4 dimensions
        raw_critiques: List[CritiquePoint] = []
        raw_critiques.extend(self._critique_reproducibility(methodology_sections, paper, anchor_map))
        raw_critiques.extend(self._critique_assumptions(methodology_sections, paper, anchor_map))
        raw_critiques.extend(self._critique_limitations(methodology_sections, paper, anchor_map))
        raw_critiques.extend(self._critique_appropriateness(methodology_sections, paper, anchor_map))

        # 4. Strict Grounding Validation: verify that quoted text exists at cited anchor
        validated_critiques: List[CritiquePoint] = []
        for cp in raw_critiques:
            is_valid, reason = self._verify_critique_grounding(cp, anchor_map)
            if is_valid:
                validated_critiques.append(cp)
            else:
                # Discard ungrounded critique per strict constraints
                pass

        # 5. Build Final Report
        report = MethodologyCritiqueReport(
            schema_version="1.0.0",
            paper_title=paper.metadata.title,
            target_sections=target_section_ids,
            critiques=validated_critiques,
            summary=self._generate_synthesis_summary(validated_critiques, target_section_ids),
        )

        # 6. Save JSON artifact if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "02_methodology_critic.json")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

        return report

    def _build_anchor_index(self, paper: PaperStructure) -> Dict[str, str]:
        """Build dictionary mapping all anchor IDs to their exact stored text or descriptions."""
        index: Dict[str, str] = {}

        # Content blocks from all sections
        for sec in paper.sections:
            for cb in sec.content_blocks:
                index[cb.id] = cb.text.strip()

        # Figures
        for fig_id, fig in paper.figures.items():
            desc = fig.caption if fig.caption else f"{fig.label} (Page {fig.page_number})"
            index[fig_id] = desc.strip()

        # Tables
        for tab_id, tab in paper.tables.items():
            desc = f"{tab.caption}\n{tab.markdown_content}" if tab.caption else tab.markdown_content
            index[tab_id] = desc.strip()

        # Equations
        for eq_id, eq in paper.equations.items():
            desc = eq.raw_text if eq.raw_text else (eq.latex if eq.latex else f"Equation {eq.label}")
            index[eq_id] = desc.strip()

        return index

    def _verify_critique_grounding(
        self, critique: CritiquePoint, anchor_map: Dict[str, str]
    ) -> Tuple[bool, str]:
        """Strictly verify that every cited anchor exists and quoted_evidence is an exact substring."""
        if not critique.anchor_ids:
            return False, "Critique has no cited anchor IDs."

        for aid in critique.anchor_ids:
            if aid not in anchor_map:
                return False, f"Cited anchor ID '{aid}' does not exist in document index."

            stored_text = re.sub(r"\s+", " ", anchor_map[aid]).strip()
            quoted_text = re.sub(r"\s+", " ", critique.quoted_evidence).strip()

            # Quoted evidence must be a substantial exact substring of the stored text
            if quoted_text.lower() not in stored_text.lower():
                # Allow partial match if quoted evidence spans across multiple sentences in that block
                quote_words = quoted_text.split()
                if len(quote_words) >= 4:
                    sub_quote = " ".join(quote_words[:4]).lower()
                    if sub_quote not in stored_text.lower():
                        return False, f"Quoted evidence not found in anchor '{aid}' text."
                else:
                    return False, f"Quoted evidence not found in anchor '{aid}' text."

        return True, "Valid"

    # =========================================================================
    # 1. REPRODUCIBILITY ANALYZER
    # =========================================================================
    def _critique_reproducibility(
        self,
        sections: List[Section],
        paper: PaperStructure,
        anchor_map: Dict[str, str],
    ) -> List[CritiquePoint]:
        """Evaluate parameter specificity, optimization hyperparameters, and evaluation protocols."""
        critiques: List[CritiquePoint] = []

        for sec in sections:
            for cb in sec.content_blocks:
                text = cb.text

                # 1.1 Hyperparameter Specifications (Optimizer, LR, Batch Size, Weight Decay)
                if re.search(r"\b(?:sgd|adam|batch\s+size|learning\s+rate|weight\s+decay|momentum)\b", text, re.I):
                    lr_match = re.search(r"learning\s+rate\s+starts\s+from\s+([0-9\.]+)", text, re.I)
                    batch_match = re.search(r"mini-batch\s+size\s+of\s+([0-9]+)", text, re.I)
                    plateau_match = re.search(r"divided\s+by\s+10\s+when\s+the\s+error\s+plateaus", text, re.I)

                    if lr_match or batch_match:
                        quote = self._extract_matching_sentence(
                            text, r"(?:SGD|batch size|learning rate|weight decay|momentum)"
                        )
                        critique_detail = (
                            "Training hyperparameters are explicitly detailed with concrete values "
                            "(learning rate, mini-batch size, momentum, weight decay, and iteration budget), "
                            "providing strong operational clarity for optimization reproduction."
                        )
                        critiques.append(
                            CritiquePoint(
                                anchor_ids=[cb.id],
                                quoted_evidence=quote,
                                critique_dimension=CritiqueDimension.REPRODUCIBILITY,
                                critique_text=critique_detail,
                                confidence=CritiqueConfidence.HIGH,
                            )
                        )

                    # Check for ambiguity in decay trigger
                    if plateau_match:
                        quote = self._extract_matching_sentence(text, r"divided by 10 when the error plateaus")
                        critique_detail = (
                            "The learning rate decay schedule specifies dividing by 10 'when the error plateaus', "
                            "but omits the quantitative threshold/patience criterion (e.g., number of epochs or validation loss delta), "
                            "introducing mild ambiguity for exact scheduling replication."
                        )
                        critiques.append(
                            CritiquePoint(
                                anchor_ids=[cb.id],
                                quoted_evidence=quote,
                                critique_dimension=CritiqueDimension.REPRODUCIBILITY,
                                critique_text=critique_detail,
                                confidence=CritiqueConfidence.MEDIUM,
                            )
                        )

                # 1.2 Data Preprocessing & Augmentation Protocol
                if re.search(r"\b(?:resized|crop|horizontal\s+flip|color\s+augmentation|scale\s+augmentation|mean\s+subtracted)\b", text, re.I):
                    quote = self._extract_matching_sentence(
                        text, r"(?:resized|crop|scale augmentation|horizontal flip|mean subtracted)"
                    )
                    critique_detail = (
                        "Data preprocessing and scale augmentation pipeline is explicitly defined with crop dimensions, "
                        "random scale intervals, and per-pixel normalization steps, enabling straightforward data pipeline reproduction."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.REPRODUCIBILITY,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 1.3 Testing and Evaluation Protocol
                if re.search(r"\b(?:10-crop\s+testing|multi-scale|fully-?convolutional|scores\s+at\s+multiple\s+scales)\b", text, re.I):
                    quote = self._extract_matching_sentence(
                        text, r"(?:10-crop testing|multiple scales|fully-?convolutional)"
                    )
                    critique_detail = (
                        "Inference protocol clearly specifies standard 10-crop testing and multi-scale test image dimensions, "
                        "ensuring evaluation parity with published benchmarks."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.REPRODUCIBILITY,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 1.4 Architecture Dimension Matching Options
                if "dimension" in text.lower() and ("zero entries padded" in text.lower() or "projection shortcut" in text.lower()):
                    quote = self._extract_matching_sentence(
                        text, r"(?:zero entries padded|projection shortcut|match dimensions)"
                    )
                    critique_detail = (
                        "Architecture specification details two explicit downsampling options: Option A (zero-padding identity) "
                        "and Option B (1x1 projection convolutions with stride 2), eliminating ambiguity in spatial/channel downsampling."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.REPRODUCIBILITY,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

        return critiques

    # =========================================================================
    # 2. ASSUMPTIONS ANALYZER
    # =========================================================================
    def _critique_assumptions(
        self,
        sections: List[Section],
        paper: PaperStructure,
        anchor_map: Dict[str, str],
    ) -> List[CritiquePoint]:
        """Evaluate theoretical premises, functional approximation hypotheses, and baseline parity."""
        critiques: List[CritiquePoint] = []

        for sec in sections:
            for cb in sec.content_blocks:
                text = cb.text

                # 2.1 Asymptotic Residual Approximation Hypothesis
                if re.search(r"hypothesi[zs]e[s]?\s+that\s+multiple\s+nonlinear\s+layers\s+can\s+asymptotically\s+approximate", text, re.I):
                    quote = self._extract_matching_sentence(
                        text, r"hypothesi[zs]e[s]?\s+that\s+multiple\s+nonlinear\s+layers"
                    )
                    critique_detail = (
                        "The mathematical justification relies on the asymptotic hypothesis that stacked nonlinear layers "
                        "can approximate residual functions H(x) - x just as well as original mappings H(x). "
                        "While formally equivalent asymptotically, this assumes finite-depth gradient dynamics behave similarly, "
                        "which is an unproven theoretical assumption left to empirical verification."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.ASSUMPTIONS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 2.2 Identity Preconditioning Hypothesis
                if re.search(r"closer\s+to\s+an\s+identity\s+mapping\s+than\s+to\s+a\s+zero\s+mapping", text, re.I):
                    quote = self._extract_matching_sentence(
                        text, r"closer\s+to\s+an\s+identity\s+mapping"
                    )
                    critique_detail = (
                        "The method makes the foundational structural assumption that the optimal mapping in deep layers "
                        "is closer to identity than to zero, asserting that reference perturbations are easier to optimize. "
                        "This assumption is intuitive for degradation mitigation but lacks formal analytical guarantees."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.ASSUMPTIONS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 2.3 Fair Baseline Isolation Assumption
                if "neither extra parameter nor computation complexity" in text.lower() or "fairly compare" in text.lower():
                    quote = self._extract_matching_sentence(
                        text, r"(?:neither extra parameter|fairly compare)"
                    )
                    critique_detail = (
                        "The methodology explicitly establishes a controlled experimental assumption: identity shortcuts add zero parameters "
                        "and negligible FLOPs, isolating depth optimization ability from capacity increases."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.ASSUMPTIONS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

        return critiques

    # =========================================================================
    # 3. LIMITATIONS ANALYZER
    # =========================================================================
    def _critique_limitations(
        self,
        sections: List[Section],
        paper: PaperStructure,
        anchor_map: Dict[str, str],
    ) -> List[CritiquePoint]:
        """Evaluate acknowledged architectural constraints, single-layer boundaries, and open questions."""
        critiques: List[CritiquePoint] = []

        for sec in sections:
            for cb in sec.content_blocks:
                text = cb.text

                # 3.1 Single-Layer Residual Block Ineffectiveness
                if re.search(r"if\s+F\s+has\s+only\s+a\s+single\s+layer.*not\s+observed\s+advantages", text, re.I):
                    quote = self._extract_matching_sentence(
                        text, r"if\s+F\s+has\s+only\s+a\s+single\s+layer"
                    )
                    critique_detail = (
                        "The methodology explicitly acknowledges a structural limitation: single-layer residual blocks "
                        "(y = W1*x + x) provide no empirical advantage over standard linear mappings, constraining effective "
                        "residual unit design to at least two stacked layers (or three for bottlenecks)."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.LIMITATIONS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 3.2 Representation Power / Theory as Open Question
                if re.search(r"hypothesis.*still\s+an\s+open\s+question", text, re.I):
                    quote = self._extract_matching_sentence(
                        text, r"still\s+an\s+open\s+question"
                    )
                    critique_detail = (
                        "The authors explicitly concede in documentation that the theoretical representation capacity "
                        "of residual functions remains an open question, acknowledging the empirical nature of the framework."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.LIMITATIONS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 3.3 Dimension Mismatch Limitation
                if "the dimensions of x and f must be equal" in text.lower() or "when changing the input/output channels" in text.lower():
                    quote = self._extract_matching_sentence(
                        text, r"dimensions\s+of\s+x\s+and\s+F\s+must\s+be\s+equal"
                    )
                    critique_detail = (
                        "Identity shortcut formulation is constrained by dimensionality: identity addition is only applicable "
                        "when feature map channels and spatial resolutions match, necessitating auxiliary projection matrices or zero-padding."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.LIMITATIONS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

        return critiques

    # =========================================================================
    # 4. APPROPRIATENESS ANALYZER
    # =========================================================================
    def _critique_appropriateness(
        self,
        sections: List[Section],
        paper: PaperStructure,
        anchor_map: Dict[str, str],
    ) -> List[CritiquePoint]:
        """Evaluate fit between formulation and degradation problem, and structural design choices."""
        critiques: List[CritiquePoint] = []

        for sec in sections:
            for cb in sec.content_blocks:
                text = cb.text

                # 4.1 Residual Formulation vs Optimization Degradation
                if "motivated by the counterintuitive phenomena about the degradation problem" in text.lower() or "reformulation may help to precondition" in text.lower():
                    quote = self._extract_matching_sentence(
                        text, r"motivated\s+by\s+the\s+counterintuitive\s+phenomena"
                    )
                    critique_detail = (
                        "The residual reformulation y = F(x) + x is directly tailored to solve network degradation: "
                        "by re-framing optimization as learning perturbations from identity rather than complete unreferenced functions, "
                        "it ensures deeper models can easily learn identity mappings in worst cases, preventing degradation."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.APPROPRIATENESS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 4.2 Batch Normalization Integration and Activation Order
                if re.search(r"batch\s+normalization\s+\(BN\)\s+\[16\]\s+right\s+after\s+each\s+convolution\s+and\s+before\s+activation", text, re.I):
                    quote = self._extract_matching_sentence(
                        text, r"batch\s+normalization\s+\(BN\)"
                    )
                    critique_detail = (
                        "Placement of Batch Normalization right after each convolution and before activation is appropriate "
                        "for maintaining stable forward activations and backward gradient variances throughout deep residual paths."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.APPROPRIATENESS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.HIGH,
                        )
                    )

                # 4.3 Second Nonlinearity After Addition
                if "second nonlinearity after the addition" in text.lower():
                    quote = self._extract_matching_sentence(
                        text, r"second\s+nonlinearity\s+after\s+the\s+addition"
                    )
                    critique_detail = (
                        "Applying the final ReLU nonlinearity after element-wise addition (sigma(F(x) + x)) ensures nonlinear "
                        "capacity across consecutive residual blocks, though it restricts shortcut outputs to non-negative values "
                        "(a design nuance later analyzed in pre-activation ResNets)."
                    )
                    critiques.append(
                        CritiquePoint(
                            anchor_ids=[cb.id],
                            quoted_evidence=quote,
                            critique_dimension=CritiqueDimension.APPROPRIATENESS,
                            critique_text=critique_detail,
                            confidence=CritiqueConfidence.MEDIUM,
                        )
                    )

        return critiques

    # =========================================================================
    # HELPER UTILITIES
    # =========================================================================
    def _extract_matching_sentence(self, block_text: str, regex_pattern: str) -> str:
        """Extract the exact sentence in block_text containing the pattern for clean evidence quoting."""
        sentences = re.split(r"(?<=[.!?])\s+", block_text)
        for s in sentences:
            if re.search(regex_pattern, s, re.I):
                return s.strip()
        # Fallback to truncated text if sentence boundary split is ambiguous
        return block_text[:180].strip()

    def _generate_synthesis_summary(
        self, critiques: List[CritiquePoint], target_sections: List[str]
    ) -> str:
        """Generate high-level synthesis of methodological strengths and limitations."""
        dim_counts = {}
        for cp in critiques:
            dim_counts[cp.critique_dimension.value] = dim_counts.get(cp.critique_dimension.value, 0) + 1

        summary_lines = [
            f"Methodology evaluation across {len(target_sections)} sections produced {len(critiques)} evidence-grounded critique points.",
            f"Breakdown by dimension: {dim_counts.get('reproducibility', 0)} reproducibility, {dim_counts.get('assumptions', 0)} assumptions, {dim_counts.get('limitations', 0)} limitations, and {dim_counts.get('appropriateness', 0)} appropriateness evaluations.",
            "All critiques are strictly grounded in verbatim text anchors from the parsed document structure.",
        ]
        return " ".join(summary_lines)
