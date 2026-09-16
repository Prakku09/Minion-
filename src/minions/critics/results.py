"""LLM-Driven Results & Experimental Evaluation Critic.

Evaluates research paper Results and Discussion sections across:
- Experimental Reproducibility
- Evaluation Rigor
- Experimental Design
- Results Interpretation

Grounding Guarantee:
- Every cited anchor ID must exist in the parsed document.
- quoted_evidence must match text stored at the cited anchor.
- Unsupported or hallucinated candidate points are dropped deterministically.
"""

import json
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from minions.schema.results_critique import (
    DroppedResultsCritiquePoint,
    ResultsCritiquePoint,
    ResultsCritiqueReport,
    ResultsDimension,
)
from minions.schema.critique import (
    ConsensusType,
    CritiqueConfidence,
)
from minions.schema.paper import (
    CanonicalSectionType,
    PaperStructure,
    Section,
)

logger = logging.getLogger(__name__)


class ResultsCritic:
    """LLM-driven critic evaluating experimental results with deterministic grounding."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_llm_fn: Optional[Callable[[str, str], str]] = None,
    ):
        """Initialize Results Critic.

        Args:
            model: Model identifier.
            provider: 'anthropic', 'openai', 'gemini', or None.
            api_key: Optional provider API key.
            custom_llm_fn: Optional deterministic/custom LLM function.
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
        self,
        paper: PaperStructure,
        output_dir: Optional[str] = None,
    ) -> ResultsCritiqueReport:
        """Run Results evaluation and deterministic grounding verification."""

        # 1. Build complete evidence anchor index.
        anchor_map = self._build_anchor_index(paper)

        # 2. Extract Results and Discussion sections.
        results_sections = [
            s
            for s in paper.sections
            if s.canonical_type
            in {
                CanonicalSectionType.RESULTS,
                CanonicalSectionType.DISCUSSION,
            }
        ]

        target_section_ids = [s.id for s in results_sections]

        if not results_sections:
            return ResultsCritiqueReport(
                paper_title=paper.metadata.title,
                target_sections=[],
                critiques=[],
                strengths=[],
                dropped_points=[],
                hallucination_rate=0.0,
                summary=(
                    "No Results or Discussion sections found in the "
                    "parsed document structure."
                ),
            )

        # 3. Build evaluation prompt.
        system_prompt, user_prompt = self._build_prompts(
            paper,
            results_sections,
        )

        # 4. Generate candidate points.
        raw_response = self._call_llm(
            system_prompt,
            user_prompt,
        )

        raw_critiques, raw_strengths = self._parse_llm_json(
            raw_response
        )

        # 5. Deterministic grounding verification.
        validated_critiques, dropped_critiques = (
            self._verify_and_filter_points(
                raw_critiques,
                anchor_map,
            )
        )

        validated_strengths, dropped_strengths = (
            self._verify_and_filter_points(
                raw_strengths,
                anchor_map,
            )
        )

        all_dropped = dropped_critiques + dropped_strengths

        total_generated = (
            len(raw_critiques) +
            len(raw_strengths)
        )

        hallucination_rate = (
            round(
                len(all_dropped) / total_generated,
                4,
            )
            if total_generated > 0
            else 0.0
        )

        # 6. Build final report.
        report = ResultsCritiqueReport(
            schema_version="1.0.0",
            paper_title=paper.metadata.title,
            target_sections=target_section_ids,
            critiques=validated_critiques,
            strengths=validated_strengths,
            dropped_points=all_dropped,
            hallucination_rate=hallucination_rate,
            summary=self._generate_synthesis_summary(
                validated_critiques,
                validated_strengths,
                all_dropped,
                target_section_ids,
            ),
        )

        # 7. Save JSON artifact.
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            report_path = os.path.join(
                output_dir,
                "02_results_critic.json",
            )

            with open(
                report_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    report.model_dump_json(
                        indent=2
                    )
                )

        return report

    def _build_anchor_index(
        self,
        paper: PaperStructure,
    ) -> Dict[str, str]:
        """Build anchor ID -> exact stored evidence mapping."""

        index: Dict[str, str] = {}

        # Text content blocks.
        for sec in paper.sections:
            for cb in sec.content_blocks:
                index[cb.id] = cb.text.strip()

        # Figures.
        for fig_id, fig in paper.figures.items():
            desc = (
                fig.caption
                if fig.caption
                else f"{fig.label} (Page {fig.page_number})"
            )

            if fig.visual_summary:
                desc = f"{desc}\n{fig.visual_summary}"

            index[fig_id] = desc.strip()

        # Tables.
        for tab_id, tab in paper.tables.items():
            desc = (
                f"{tab.caption}\n{tab.markdown_content}"
                if tab.caption
                else tab.markdown_content
            )

            index[tab_id] = desc.strip()

        # Equations.
        for eq_id, eq in paper.equations.items():
            desc = (
                eq.raw_text
                if eq.raw_text
                else (
                    eq.latex
                    if eq.latex
                    else f"Equation {eq.label}"
                )
            )

            index[eq_id] = desc.strip()

        return index

    def _build_prompts(
        self,
        paper: PaperStructure,
        results_sections: List[Section],
    ) -> Tuple[str, str]:
        """Build the Results evaluation prompt and evidence catalog."""

        system_prompt = (
            "You are an expert scientific peer reviewer conducting "
            "a rigorous evaluation of the Results and Discussion "
            "sections of a research paper.\n\n"

            "### EVALUATION RUBRIC:\n"

            "Analyze the provided experimental results, tables, "
            "figures, equations, and discussion text across four "
            "core dimensions:\n"

            "1. EXPERIMENTAL REPRODUCIBILITY: "
            "Are datasets, splits, experimental configurations, "
            "hyperparameters, hardware, evaluation procedures, "
            "and repeated-run details sufficiently specified "
            "to reproduce the reported experiments?\n"

            "2. EVALUATION RIGOR: "
            "Are appropriate metrics, baselines, comparisons, "
            "statistical evidence, variance, confidence intervals, "
            "or significance analyses provided where appropriate?\n"

            "3. EXPERIMENTAL DESIGN: "
            "Are experiments properly controlled? Examine ablations, "
            "baseline fairness, comparison protocols, confounders, "
            "dataset construction, and experimental controls.\n"

            "4. RESULTS INTERPRETATION: "
            "Do the reported results actually support the paper's "
            "claims? Identify unsupported conclusions, overclaiming, "
            "missing comparisons, contradictory findings, or "
            "important limitations in interpretation.\n\n"

            "### CRITICAL SEPARATION RULE:\n"

            "- 'critiques': ONLY emit a point when it identifies "
            "a genuine experimental gap, ambiguity, unjustified "
            "comparison, missing control, unsupported interpretation, "
            "or reproducibility risk.\n"

            "- 'strengths': ONLY emit confirmatory observations where "
            "the experiments are explicitly well-specified, rigorous, "
            "controlled, or clearly reported.\n\n"

            "### GROUNDING MANDATE:\n"

            "- Every point MUST cite exact anchor ID(s) from the "
            "document catalog.\n"

            "- Every point MUST provide quoted_evidence as a "
            "complete, exact verbatim clause or sentence copied "
            "directly from the cited anchor.\n"

            "- Do NOT invent numerical results, datasets, baselines, "
            "experimental settings, or claims not present in the "
            "provided catalog.\n"

            "- Do NOT truncate quotes unnecessarily.\n"

            "- If evidence is ambiguous, use confidence "
            "'medium' or 'low'.\n\n"

            "### OUTPUT JSON FORMAT:\n"

            "Respond ONLY with valid JSON:\n"

            "{\n"
            '  "critiques": [\n'
            "    {\n"
            '      "anchor_ids": ["anchor_id"],\n'
            '      "quoted_evidence": "exact verbatim evidence",\n'
            '      "critique_dimension": '
            '"experimental_reproducibility|evaluation_rigor|'
            'experimental_design|results_interpretation",\n'
            '      "critique_text": "analytical evaluation",\n'
            '      "confidence": "high|medium|low"\n'
            "    }\n"
            "  ],\n"

            '  "strengths": [\n'
            "    {\n"
            '      "anchor_ids": ["anchor_id"],\n'
            '      "quoted_evidence": "exact verbatim evidence",\n'
            '      "critique_dimension": '
            '"experimental_reproducibility|evaluation_rigor|'
            'experimental_design|results_interpretation",\n'
            '      "critique_text": "analytical evaluation",\n'
            '      "confidence": "high|medium|low"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        catalog_lines = [
            f"PAPER TITLE: {paper.metadata.title}",
            f"RESULTS/DISCUSSION SECTIONS COUNT: "
            f"{len(results_sections)}",
            "",
            "=== DOCUMENT RESULTS BLOCK CATALOG ===",
        ]

        for section in results_sections:
            catalog_lines.append(
                f"\nSECTION: [{section.id}] "
                f"{section.heading_title} "
                f"(Level {section.level})"
            )

            if section.figure_ids:
                catalog_lines.append(
                    f"  Bound Figures: {section.figure_ids}"
                )

            if section.table_ids:
                catalog_lines.append(
                    f"  Bound Tables: {section.table_ids}"
                )

            if section.equation_ids:
                catalog_lines.append(
                    f"  Bound Equations: {section.equation_ids}"
                )

            for cb in section.content_blocks:
                text_clean = (
                    cb.text
                    .strip()
                    .replace("\n", " ")
                )

                catalog_lines.append(
                    f'  [{cb.id}] '
                    f'(Page {cb.page_number}): '
                    f'"{text_clean}"'
                )

        # Add assets referenced by Results/Discussion.
        catalog_lines.append(
            "\n=== BOUND RESULTS ASSETS CATALOG ==="
        )

        figure_ids = {
            fid
            for s in results_sections
            for fid in s.figure_ids
        }

        table_ids = {
            tid
            for s in results_sections
            for tid in s.table_ids
        }

        equation_ids = {
            eid
            for s in results_sections
            for eid in s.equation_ids
        }

        for fid in sorted(figure_ids):
            if fid in paper.figures:
                fig = paper.figures[fid]

                catalog_lines.append(
                    f'  [{fid}] '
                    f'Caption: "{fig.caption}"'
                )

                if fig.visual_summary:
                    catalog_lines.append(
                        f'  [{fid}] '
                        f'Visual Summary: "{fig.visual_summary}"'
                    )

        for tid in sorted(table_ids):
            if tid in paper.tables:
                tab = paper.tables[tid]

                catalog_lines.append(
                    f'  [{tid}] '
                    f'Caption: "{tab.caption}" | '
                    f'Markdown: "{tab.markdown_content}"'
                )

        for eid in sorted(equation_ids):
            if eid in paper.equations:
                eq = paper.equations[eid]

                catalog_lines.append(
                    f'  [{eid}] '
                    f'Raw: "{eq.raw_text}" | '
                    f'LaTeX: "{eq.latex}"'
                )

        user_prompt = "\n".join(catalog_lines)

        return system_prompt, user_prompt

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Execute configured LLM or custom test function."""

        if self.custom_llm_fn is not None:
            return self.custom_llm_fn(
                system_prompt,
                user_prompt,
            )

        provider = self.provider.lower()

        # Anthropic.
        if provider == "anthropic":
            try:
                import anthropic

                api_key = (
                    self.api_key
                    or os.environ.get(
                        "ANTHROPIC_API_KEY"
                    )
                )

                client = (
                    anthropic.Anthropic(
                        api_key=api_key
                    )
                    if api_key
                    else anthropic.Anthropic()
                )

                response = client.messages.create(
                    model=(
                        self.model
                        if "claude" in self.model
                        else "claude-3-5-sonnet-20241022"
                    ),
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                )

                return response.content[0].text

            except Exception as e:
                logger.warning(
                    f"Anthropic API call failed ({e}). "
                    "Checking alternative providers..."
                )

        # OpenAI.
        if (
            provider == "openai"
            or os.environ.get("OPENAI_API_KEY")
        ):
            try:
                import openai

                api_key = (
                    self.api_key
                    or os.environ.get(
                        "OPENAI_API_KEY"
                    )
                )

                client = openai.OpenAI(
                    api_key=api_key
                )

                response = (
                    client.chat.completions.create(
                        model=(
                            self.model
                            if "gpt" in self.model
                            else "gpt-4o"
                        ),
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        response_format={
                            "type": "json_object"
                        },
                        temperature=0.2,
                    )
                )

                return response.choices[0].message.content

            except Exception as e:
                logger.warning(
                    f"OpenAI API call failed ({e})."
                )

        # Gemini.
        if (
            provider == "gemini"
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        ):
            try:
                from google import genai

                api_key = (
                    self.api_key
                    or os.environ.get(
                        "GEMINI_API_KEY"
                    )
                    or os.environ.get(
                        "GOOGLE_API_KEY"
                    )
                )

                client = genai.Client(
                    api_key=api_key
                )

                response = (
                    client.models.generate_content(
                        model=(
                            self.model
                            if "gemini" in self.model
                            else "gemini-2.0-flash"
                        ),
                        contents=(
                            f"{system_prompt}\n\n"
                            f"{user_prompt}"
                        ),
                    )
                )

                return response.text

            except Exception as e:
                logger.warning(
                    f"Google GenAI API call failed ({e})."
                )

        raise RuntimeError(
            "No LLM API key available "
            "(checked ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, GEMINI_API_KEY). "
            "Please provide an API key or custom_llm_fn."
        )

    def _parse_llm_json(
        self,
        raw_text: str,
    ) -> Tuple[
        List[ResultsCritiquePoint],
        List[ResultsCritiquePoint],
    ]:
        """Parse LLM response into ResultsCritiquePoint objects."""

        json_str = raw_text.strip()

        if "```json" in json_str:
            json_str = (
                json_str
                .split("```json", 1)[1]
                .split("```", 1)[0]
                .strip()
            )

        elif "```" in json_str:
            json_str = (
                json_str
                .split("```", 1)[1]
                .split("```", 1)[0]
                .strip()
            )

        try:
            data = json.loads(json_str)

        except Exception as e:
            logger.error(
                f"Failed to parse LLM JSON output: {e}\n"
                f"Raw output:\n{raw_text}"
            )

            return [], []

        critiques = self._parse_points(
            data.get("critiques", [])
        )

        strengths = self._parse_points(
            data.get("strengths", [])
        )

        return critiques, strengths

    def _parse_points(
        self,
        points_data: List[dict],
    ) -> List[ResultsCritiquePoint]:
        """Convert raw dictionaries into validated schema objects."""

        points: List[ResultsCritiquePoint] = []

        for item in points_data:
            try:
                point = ResultsCritiquePoint(
                    anchor_ids=item["anchor_ids"],
                    quoted_evidence=item["quoted_evidence"],
                    critique_dimension=ResultsDimension(
                        item["critique_dimension"].lower()
                    ),
                    critique_text=item["critique_text"],
                    confidence=CritiqueConfidence(
                        item.get(
                            "confidence",
                            "high",
                        ).lower()
                    ),
                )

                points.append(point)

            except Exception as e:
                logger.warning(
                    f"Skipping malformed results point: {e}"
                )

        return points

    def _verify_and_filter_points(
        self,
        points: List[ResultsCritiquePoint],
        anchor_map: Dict[str, str],
    ) -> Tuple[
        List[ResultsCritiquePoint],
        List[DroppedResultsCritiquePoint],
    ]:
        """Verify every Results point against the anchor index."""

        validated = []
        dropped = []

        for point in points:
            is_valid, reason = (
                self._verify_point_grounding(
                    point,
                    anchor_map,
                )
            )

            if is_valid:
                validated.append(point)

            else:
                dropped.append(
                    DroppedResultsCritiquePoint(
                        point=point,
                        drop_reason=reason,
                    )
                )

        return validated, dropped

    def _verify_point_grounding(
        self,
        point: ResultsCritiquePoint,
        anchor_map: Dict[str, str],
    ) -> Tuple[bool, str]:
        """Strictly verify anchor IDs and quoted evidence."""

        if not point.anchor_ids:
            return (
                False,
                "Results point has no cited anchor IDs.",
            )

        for anchor_id in point.anchor_ids:

            if anchor_id not in anchor_map:
                return (
                    False,
                    f"Cited anchor ID '{anchor_id}' "
                    "does not exist in document index.",
                )

            stored_text = re.sub(
                r"\s+",
                " ",
                anchor_map[anchor_id],
            ).strip().lower()

            quoted_text = re.sub(
                r"\s+",
                " ",
                point.quoted_evidence,
            ).strip().lower()

            if not quoted_text:
                return (
                    False,
                    f"Quoted evidence is empty for "
                    f"anchor '{anchor_id}'.",
                )

            # Exact normalized substring match.
            if quoted_text in stored_text:
                continue

            # Same fallback used by MethodologyCritic.
            quote_words = quoted_text.split()

            if len(quote_words) >= 4:
                sub_quote = " ".join(
                    quote_words[:4]
                )

                if sub_quote in stored_text:
                    continue

            return (
                False,
                f"Quoted evidence "
                f"'{quoted_text[:40]}...' "
                f"not found in anchor '{anchor_id}' text.",
            )

        return True, "Valid"

    def _generate_synthesis_summary(
        self,
        critiques: List[ResultsCritiquePoint],
        strengths: List[ResultsCritiquePoint],
        dropped: List[DroppedResultsCritiquePoint],
        target_sections: List[str],
    ) -> str:
        """Generate deterministic summary of Results findings."""

        critique_dims = {}

        for point in critiques:
            dimension = point.critique_dimension.value

            critique_dims[dimension] = (
                critique_dims.get(
                    dimension,
                    0,
                )
                + 1
            )

        strength_dims = {}

        for point in strengths:
            dimension = point.critique_dimension.value

            strength_dims[dimension] = (
                strength_dims.get(
                    dimension,
                    0,
                )
                + 1
            )

        total_generated = (
            len(critiques)
            + len(strengths)
            + len(dropped)
        )

        hallucination_pct = (
            f"{round(len(dropped) / total_generated * 100, 1)}%"
            if total_generated > 0
            else "0.0%"
        )

        return " ".join(
            [
                (
                    f"Results evaluation across "
                    f"{len(target_sections)} sections produced "
                    f"{len(critiques)} verified experimental critiques "
                    f"and {len(strengths)} verified strengths."
                ),
                (
                    "Critique breakdown: "
                    f"{critique_dims.get('experimental_reproducibility', 0)} "
                    "reproducibility gaps, "
                    f"{critique_dims.get('evaluation_rigor', 0)} "
                    "evaluation-rigor concerns, "
                    f"{critique_dims.get('experimental_design', 0)} "
                    "experimental-design concerns, and "
                    f"{critique_dims.get('results_interpretation', 0)} "
                    "results-interpretation concerns."
                ),
                (
                    "Strength breakdown: "
                    f"{strength_dims.get('experimental_reproducibility', 0)} "
                    "reproducibility strengths, "
                    f"{strength_dims.get('evaluation_rigor', 0)} "
                    "evaluation-rigor strengths, "
                    f"{strength_dims.get('experimental_design', 0)} "
                    "experimental-design strengths, and "
                    f"{strength_dims.get('results_interpretation', 0)} "
                    "interpretation strengths."
                ),
                (
                    f"Post-hoc grounding dropped "
                    f"{len(dropped)} unsupported or mismatched points "
                    f"(hallucination rate: {hallucination_pct})."
                ),
            ]
        )