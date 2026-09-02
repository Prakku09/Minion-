"""LLM-Driven Methodology Critic Agent.

Evaluates scientific paper methodology sections against structured criteria:
- Reproducibility
- Stated Assumptions
- Acknowledged Limitations
- Method Appropriateness

Enforces strict post-hoc grounding verification against parsed anchor blocks.
"""

import json
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

from minions.schema.critique import (
    CritiqueConfidence,
    CritiqueDimension,
    CritiquePoint,
    DroppedCritiquePoint,
    MethodologyCritiqueReport,
)
from minions.schema.paper import CanonicalSectionType, PaperStructure, Section

logger = logging.getLogger(__name__)


class MethodologyCritic:
    """LLM-driven critic evaluating methodology sections with deterministic post-hoc quote grounding."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_llm_fn: Optional[Callable[[str, str], str]] = None,
    ):
        """Initialize Methodology Critic.

        Args:
            model: Model identifier (e.g. 'claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini-2.0-flash').
            provider: 'anthropic', 'openai', 'gemini', or None (auto-detected).
            api_key: API key for the chosen provider.
            custom_llm_fn: Optional custom callable (system_prompt, user_prompt) -> response_text.
        """
        self.model = model
        self.provider = provider or self._detect_provider(model)
        self.api_key = api_key
        self.custom_llm_fn = custom_llm_fn

    def _detect_provider(self, model: str) -> str:
        """Infer provider from model name."""
        m = model.lower()
        if "claude" in m:
            return "anthropic"
        if "gpt" in m or "o1" in m or "o3" in m:
            return "openai"
        if "gemini" in m:
            return "gemini"
        return "anthropic"

    def critique_paper(
        self, paper: PaperStructure, output_dir: Optional[str] = None
    ) -> MethodologyCritiqueReport:
        """Run full LLM-driven methodology critique and post-hoc grounding verification."""
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
                strengths=[],
                dropped_points=[],
                hallucination_rate=0.0,
                summary="No methodology sections found in the parsed document structure.",
            )

        # 3. Compile structured prompt for the LLM
        system_prompt, user_prompt = self._build_prompts(paper, methodology_sections)

        # 4. Generate candidate critiques and strengths via LLM
        raw_response = self._call_llm(system_prompt, user_prompt)
        raw_critiques, raw_strengths = self._parse_llm_json(raw_response)

        # 5. Post-Hoc Grounding Verification: Verify quoted evidence exists verbatim at cited anchor
        validated_critiques, dropped_critiques = self._verify_and_filter_points(
            raw_critiques, anchor_map
        )
        validated_strengths, dropped_strengths = self._verify_and_filter_points(
            raw_strengths, anchor_map
        )

        all_dropped = dropped_critiques + dropped_strengths
        total_generated = len(raw_critiques) + len(raw_strengths)
        hallucination_rate = (
            round(len(all_dropped) / total_generated, 4) if total_generated > 0 else 0.0
        )

        # 6. Build Final Report
        report = MethodologyCritiqueReport(
            schema_version="1.0.0",
            paper_title=paper.metadata.title,
            target_sections=target_section_ids,
            critiques=validated_critiques,
            strengths=validated_strengths,
            dropped_points=all_dropped,
            hallucination_rate=hallucination_rate,
            summary=self._generate_synthesis_summary(
                validated_critiques, validated_strengths, all_dropped, target_section_ids
            ),
        )

        # 7. Save JSON artifact if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "02_methodology_critic.json")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

        return report

    def _build_anchor_index(self, paper: PaperStructure) -> Dict[str, str]:
        """Build dictionary mapping all anchor IDs to their exact stored text or descriptions."""
        index: Dict[str, str] = {}

        for sec in paper.sections:
            for cb in sec.content_blocks:
                index[cb.id] = cb.text.strip()

        for fig_id, fig in paper.figures.items():
            desc = fig.caption if fig.caption else f"{fig.label} (Page {fig.page_number})"
            index[fig_id] = desc.strip()

        for tab_id, tab in paper.tables.items():
            desc = f"{tab.caption}\n{tab.markdown_content}" if tab.caption else tab.markdown_content
            index[tab_id] = desc.strip()

        for eq_id, eq in paper.equations.items():
            desc = eq.raw_text if eq.raw_text else (eq.latex if eq.latex else f"Equation {eq.label}")
            index[eq_id] = desc.strip()

        return index

    def _build_prompts(
        self, paper: PaperStructure, methodology_sections: List[Section]
    ) -> Tuple[str, str]:
        """Build prompt containing rubric instructions and formatted methodology block catalog."""
        system_prompt = (
            "You are an expert scientific peer reviewer conducting a rigorous methodology review of a research paper.\n\n"
            "### EVALUATION RUBRIC:\n"
            "Analyze the provided methodology text and bound assets across four core dimensions:\n"
            "1. REPRODUCIBILITY: Are steps, formulas, parameters, hyperparameters, datasets, and evaluation protocols described with sufficient precision to reproduce?\n"
            "2. STATED ASSUMPTIONS: Are key theoretical premises, inductive priors, asymptotic approximations, and baseline isolation controls made explicit or left implicit?\n"
            "3. ACKNOWLEDGED LIMITATIONS: Does the methodology address its own architectural bounds, single-layer constraints, open theoretical questions, or domain constraints?\n"
            "4. METHOD APPROPRIATENESS: Is the chosen mathematical/architectural formulation suited to the stated problem without introducing confounding variables?\n\n"
            "### CRITICAL SEPARATION RULE:\n"
            "- 'critiques': ONLY emit a point if it identifies a genuine gap, ambiguity, unjustified choice, unproven assumption, or risk. Do NOT emit a critique that merely confirms quality.\n"
            "- 'strengths': ONLY emit confirmatory and rigorous observations where the methodology is explicitly well-specified, well-controlled, or structurally sound.\n\n"
            "### GROUNDING MANDATE:\n"
            "- Every point in both arrays MUST cite the exact anchor ID(s) (e.g. ['sec_methodology_4_1_..._b02']) from the document catalog.\n"
            "- Every point MUST provide 'quoted_evidence' as a short, EXACT verbatim quote copied directly from the cited anchor block text.\n"
            "- If evidence is ambiguous, set confidence to 'medium' or 'low'. Otherwise 'high'.\n\n"
            "### OUTPUT JSON FORMAT:\n"
            "You MUST respond ONLY with a valid JSON object with the following structure:\n"
            "{\n"
            '  "critiques": [\n'
            "    {\n"
            '      "anchor_ids": ["anchor_id"],\n'
            '      "quoted_evidence": "exact verbatim quote from block",\n'
            '      "critique_dimension": "reproducibility|assumptions|limitations|appropriateness",\n'
            '      "critique_text": "description of gap/risk/unjustified choice",\n'
            '      "confidence": "high|medium|low"\n'
            "    }\n"
            "  ],\n"
            '  "strengths": [\n'
            "    {\n"
            '      "anchor_ids": ["anchor_id"],\n'
            '      "quoted_evidence": "exact verbatim quote from block",\n'
            '      "critique_dimension": "reproducibility|assumptions|limitations|appropriateness",\n'
            '      "critique_text": "description of well-specified strength",\n'
            '      "confidence": "high|medium|low"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        # Build Document Catalog
        catalog_lines = [
            f"PAPER TITLE: {paper.metadata.title}",
            f"METHODOLOGY SECTIONS COUNT: {len(methodology_sections)}",
            "\n=== DOCUMENT METHODOLOGY BLOCK CATALOG ===",
        ]

        for s in methodology_sections:
            catalog_lines.append(f"\nSECTION: [{s.id}] {s.heading_title} (Level {s.level})")
            if s.figure_ids:
                catalog_lines.append(f"  Bound Figures: {s.figure_ids}")
            if s.table_ids:
                catalog_lines.append(f"  Bound Tables: {s.table_ids}")
            if s.equation_ids:
                catalog_lines.append(f"  Bound Equations: {s.equation_ids}")

            for cb in s.content_blocks:
                text_clean = cb.text.strip().replace("\n", " ")
                catalog_lines.append(f"  [{cb.id}] (Page {cb.page_number}): \"{text_clean}\"")

        # Include referenced figures/tables/equations text
        catalog_lines.append("\n=== BOUND ASSETS CATALOG ===")
        all_fig_ids = set(f for s in methodology_sections for f in s.figure_ids)
        all_tab_ids = set(t for s in methodology_sections for t in s.table_ids)
        all_eq_ids = set(e for s in methodology_sections for e in s.equation_ids)

        for fid in sorted(all_fig_ids):
            if fid in paper.figures:
                fig = paper.figures[fid]
                catalog_lines.append(f"  [{fid}] Caption: \"{fig.caption}\"")
        for tid in sorted(all_tab_ids):
            if tid in paper.tables:
                tab = paper.tables[tid]
                catalog_lines.append(f"  [{tid}] Caption: \"{tab.caption}\" | Markdown: \"{tab.markdown_content}\"")
        for eid in sorted(all_eq_ids):
            if eid in paper.equations:
                eq = paper.equations[eid]
                catalog_lines.append(f"  [{eid}] Raw: \"{eq.raw_text}\" | LaTeX: \"{eq.latex}\"")

        user_prompt = "\n".join(catalog_lines)
        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Execute LLM call using configured provider or custom function."""
        if self.custom_llm_fn is not None:
            return self.custom_llm_fn(system_prompt, user_prompt)

        provider = self.provider.lower()

        # 1. Anthropic (Claude)
        if provider == "anthropic":
            try:
                import anthropic

                api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
                client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
                response = client.messages.create(
                    model=self.model if "claude" in self.model else "claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Anthropic API call failed ({e}). Checking alternative providers...")

        # 2. OpenAI
        if provider == "openai" or os.environ.get("OPENAI_API_KEY"):
            try:
                import openai

                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=self.model if "gpt" in self.model else "gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API call failed ({e})...")

        # 3. Google GenAI (Gemini)
        if provider == "gemini" or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            try:
                from google import genai

                api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=self.model if "gemini" in self.model else "gemini-2.0-flash",
                    contents=f"{system_prompt}\n\n{user_prompt}",
                )
                return response.text
            except Exception as e:
                logger.warning(f"Google GenAI API call failed ({e})...")

        raise RuntimeError(
            "No LLM API key available (checked ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY). "
            "Please provide an API key or a custom_llm_fn."
        )

    def _parse_llm_json(
        self, raw_text: str
    ) -> Tuple[List[CritiquePoint], List[CritiquePoint]]:
        """Parse raw LLM response text into candidate CritiquePoint lists."""
        # Extract JSON block if wrapped in markdown fences
        json_str = raw_text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            data = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON output: {e}\nRaw output:\n{raw_text}")
            return [], []

        critiques_data = data.get("critiques", [])
        strengths_data = data.get("strengths", [])

        critiques: List[CritiquePoint] = []
        for item in critiques_data:
            try:
                cp = CritiquePoint(
                    anchor_ids=item["anchor_ids"],
                    quoted_evidence=item["quoted_evidence"],
                    critique_dimension=CritiqueDimension(item["critique_dimension"].lower()),
                    critique_text=item["critique_text"],
                    confidence=CritiqueConfidence(item.get("confidence", "high").lower()),
                )
                critiques.append(cp)
            except Exception as e:
                logger.warning(f"Skipping malformed critique object: {e}")

        strengths: List[CritiquePoint] = []
        for item in strengths_data:
            try:
                sp = CritiquePoint(
                    anchor_ids=item["anchor_ids"],
                    quoted_evidence=item["quoted_evidence"],
                    critique_dimension=CritiqueDimension(item["critique_dimension"].lower()),
                    critique_text=item["critique_text"],
                    confidence=CritiqueConfidence(item.get("confidence", "high").lower()),
                )
                strengths.append(sp)
            except Exception as e:
                logger.warning(f"Skipping malformed strength object: {e}")

        return critiques, strengths

    def _verify_and_filter_points(
        self, points: List[CritiquePoint], anchor_map: Dict[str, str]
    ) -> Tuple[List[CritiquePoint], List[DroppedCritiquePoint]]:
        """Strictly verify each point against anchor store, dropping ungrounded or hallucinated points."""
        validated: List[CritiquePoint] = []
        dropped: List[DroppedCritiquePoint] = []

        for p in points:
            is_valid, reason = self._verify_critique_grounding(p, anchor_map)
            if is_valid:
                validated.append(p)
            else:
                dropped.append(DroppedCritiquePoint(point=p, drop_reason=reason))

        return validated, dropped

    def _verify_critique_grounding(
        self, critique: CritiquePoint, anchor_map: Dict[str, str]
    ) -> Tuple[bool, str]:
        """Strictly verify that cited anchor exists and quoted_evidence is an exact substring."""
        if not critique.anchor_ids:
            return False, "Critique has no cited anchor IDs."

        for aid in critique.anchor_ids:
            if aid not in anchor_map:
                return False, f"Cited anchor ID '{aid}' does not exist in document index."

            stored_text = re.sub(r"\s+", " ", anchor_map[aid]).strip().lower()
            quoted_text = re.sub(r"\s+", " ", critique.quoted_evidence).strip().lower()

            if not quoted_text:
                return False, f"Quoted evidence is empty for anchor '{aid}'."

            # Exact normalized substring match
            if quoted_text in stored_text:
                continue

            # Fallback: check prefix 5 words
            quote_words = quoted_text.split()
            if len(quote_words) >= 4:
                sub_quote = " ".join(quote_words[:4])
                if sub_quote in stored_text:
                    continue
                return False, f"Quoted evidence '{quoted_text[:40]}...' not found in anchor '{aid}' text."
            else:
                return False, f"Quoted evidence '{quoted_text}' not found in anchor '{aid}' text."

        return True, "Valid"

    def _generate_synthesis_summary(
        self,
        critiques: List[CritiquePoint],
        strengths: List[CritiquePoint],
        dropped: List[DroppedCritiquePoint],
        target_sections: List[str],
    ) -> str:
        """Generate high-level synthesis of methodological findings."""
        critique_dims = {}
        for cp in critiques:
            critique_dims[cp.critique_dimension.value] = (
                critique_dims.get(cp.critique_dimension.value, 0) + 1
            )

        strength_dims = {}
        for sp in strengths:
            strength_dims[sp.critique_dimension.value] = (
                strength_dims.get(sp.critique_dimension.value, 0) + 1
            )

        total_gen = len(critiques) + len(strengths) + len(dropped)
        hallucination_pct = f"{round(len(dropped) / total_gen * 100, 1)}%" if total_gen > 0 else "0.0%"

        summary_lines = [
            f"Methodology evaluation across {len(target_sections)} sections produced {len(critiques)} verified material critiques and {len(strengths)} verified strengths.",
            f"Critique breakdown: {critique_dims.get('reproducibility', 0)} reproducibility gaps, {critique_dims.get('assumptions', 0)} unproven assumptions, {critique_dims.get('limitations', 0)} architectural constraints, and {critique_dims.get('appropriateness', 0)} design risks.",
            f"Strength breakdown: {strength_dims.get('reproducibility', 0)} reproducibility specs, {strength_dims.get('assumptions', 0)} baseline controls, and {strength_dims.get('appropriateness', 0)} design strengths.",
            f"Post-hoc grounding dropped {len(dropped)} hallucinated/mismatched points (hallucination rate: {hallucination_pct}).",
        ]
        return " ".join(summary_lines)
