"""Methodology scorer for dual-pass, evidence-grounded paper review."""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from minions.schema.critique import (
    ConsensusType,
    CritiqueDimension,
    CritiquePoint,
    DimensionScore,
    MethodologyCritiqueReport,
    MethodologyScores,
)

logger = logging.getLogger(__name__)


class MethodologyScorer:
    """Score methodology reports with independent LLM pass 1 and pass 2 reviews."""

    def __init__(
        self,
        custom_llm_fn: Optional[Callable[..., Any]] = None,
        model: str = "gpt-4o",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.8,
    ):
        self.custom_llm_fn = custom_llm_fn
        self.model = model
        self.provider = provider or self._detect_provider(model)
        self.api_key = api_key
        self.temperature = temperature
        self._last_pass_prompts: Dict[int, str] = {}

    def _detect_provider(self, model: str) -> str:
        model_name = (model or "").lower()
        if "claude" in model_name:
            return "anthropic"
        if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            return "openai"
        if "gemini" in model_name:
            return "gemini"
        return "openai"

    def _temperature_for_pass(self, pass_id: int) -> float:
        return 0.2 if pass_id == 1 else max(self.temperature, 0.8)

    def score_report(
        self, report: MethodologyCritiqueReport, output_dir: Optional[str] = None
    ) -> MethodologyCritiqueReport:
        """Score a methodology report with genuinely independent dual-pass LLM review."""
        if self.custom_llm_fn is None and not self._api_is_available():
            raise RuntimeError(
                "API-backed dual-pass scoring is required; deterministic fallback is disabled. "
                "Provide custom_llm_fn or configure an LLM API key."
            )

        dimension_scores: Dict[CritiqueDimension, DimensionScore] = {}
        pass_1_raw: Dict[str, int] = {}
        pass_2_raw: Dict[str, int] = {}

        for dim in CritiqueDimension:
            evidence = self._dimension_evidence(report, dim)
            pass_1_result = self._score_dimension_with_llm(report, dim, evidence, pass_id=1)
            pass_2_result = self._score_dimension_with_llm(report, dim, evidence, pass_id=2)

            if pass_1_result["rating"] == pass_2_result["rating"]:
                resolved_rating = pass_1_result["rating"]
                flag = None
            elif abs(pass_1_result["rating"] - pass_2_result["rating"]) <= 1:
                resolved_rating = round((pass_1_result["rating"] + pass_2_result["rating"]) / 2)
                flag = "near_miss"
            else:
                resolved_rating = round((pass_1_result["rating"] + pass_2_result["rating"]) / 2)
                flag = "high_disagreement"

            final_reasoning = (
                f"Pass 1: {pass_1_result['reasoning']} | Pass 2: {pass_2_result['reasoning']}"
            )
            final_score = DimensionScore(
                rating=resolved_rating,
                reasoning=final_reasoning,
                based_on_anchor_ids=list(
                    dict.fromkeys(
                        (pass_1_result["based_on_anchor_ids"] or [])
                        + (pass_2_result["based_on_anchor_ids"] or [])
                    )
                ),
                pass_1_rating=pass_1_result["rating"],
                pass_2_rating=pass_2_result["rating"],
                pass_1_reasoning=pass_1_result["reasoning"],
                pass_2_reasoning=pass_2_result["reasoning"],
                disagreement_flag=flag,
            )
            dimension_scores[dim] = final_score
            pass_1_raw[dim.value] = pass_1_result["rating"]
            pass_2_raw[dim.value] = pass_2_result["rating"]

        overall = round(sum(s.rating for s in dimension_scores.values()) / len(dimension_scores), 2)
        exact_match, near_miss, mean_delta = self._compute_scoring_agreement(pass_1_raw, pass_2_raw)

        final_scores = MethodologyScores(
            reproducibility=dimension_scores[CritiqueDimension.REPRODUCIBILITY],
            assumptions=dimension_scores[CritiqueDimension.ASSUMPTIONS],
            limitations=dimension_scores[CritiqueDimension.LIMITATIONS],
            appropriateness=dimension_scores[CritiqueDimension.APPROPRIATENESS],
            overall_rating=overall,
            pass_1_scores=pass_1_raw,
            pass_2_scores=pass_2_raw,
            exact_match_rate=exact_match,
            near_miss_rate=near_miss,
            mean_absolute_delta=mean_delta,
            scoring_pass_agreement=exact_match,
        )

        report.scores = final_scores
        return report

    def _api_is_available(self) -> bool:
        return any(
            os.environ.get(key) is not None
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
        )

    def _dimension_evidence(self, report: MethodologyCritiqueReport, dimension: CritiqueDimension) -> Dict[str, Any]:
        dim_critiques = [c for c in report.critiques if c.critique_dimension == dimension]
        dim_strengths = [s for s in report.strengths if s.critique_dimension == dimension]

        core_critiques = [c for c in dim_critiques if c.consensus == ConsensusType.CORE]
        secondary_critiques = [c for c in dim_critiques if c.consensus == ConsensusType.SECONDARY]
        core_strengths = [s for s in dim_strengths if s.consensus == ConsensusType.CORE]
        secondary_strengths = [s for s in dim_strengths if s.consensus == ConsensusType.SECONDARY]

        anchor_ids: List[str] = []
        for points in (core_critiques, secondary_critiques, core_strengths, secondary_strengths):
            for point in points:
                anchor_ids.extend(point.anchor_ids)

        return {
            "dimension": dimension.value,
            "anchor_ids": list(dict.fromkeys(anchor_ids)),
            "core_critiques": [
                {
                    "anchor_ids": c.anchor_ids,
                    "quoted_evidence": c.quoted_evidence,
                    "critique_text": c.critique_text,
                }
                for c in core_critiques
            ],
            "secondary_critiques": [
                {
                    "anchor_ids": c.anchor_ids,
                    "quoted_evidence": c.quoted_evidence,
                    "critique_text": c.critique_text,
                }
                for c in secondary_critiques
            ],
            "core_strengths": [
                {
                    "anchor_ids": s.anchor_ids,
                    "quoted_evidence": s.quoted_evidence,
                    "critique_text": s.critique_text,
                }
                for s in core_strengths
            ],
            "secondary_strengths": [
                {
                    "anchor_ids": s.anchor_ids,
                    "quoted_evidence": s.quoted_evidence,
                    "critique_text": s.critique_text,
                }
                for s in secondary_strengths
            ],
        }

    def _build_dimension_prompt(self, pass_id: int, dimension: CritiqueDimension, evidence: Dict[str, Any]) -> str:
        reviewer = "Balanced Peer Reviewer" if pass_id == 1 else "Skeptical Auditor"
        framing = (
            "You are a careful, evidence-weighted paper reviewer. Give credit for clear method specification and good controls."
            if pass_id == 1
            else "You are a skeptical reviewer. Emphasize missing assumptions, weak controls, unjustified heuristics, and risk of over-claiming."
        )

        header = [
            f"Dimension: {dimension.value}",
            f"Reviewer: {reviewer}",
            framing,
            "Score the dimension on a strict 1-5 integer scale.",
            "Use the CORE critiques/strengths as the main evidence. Secondary observations may be noted but must not dominate your reasoning.",
            "Return JSON with exactly these fields: rating, reasoning, based_on_anchor_ids",
        ]
        core_critiques = evidence["core_critiques"]
        core_strengths = evidence["core_strengths"]
        secondary_critiques = evidence["secondary_critiques"]

        lines = ["\n".join(header)]
        lines.append("CORE STRENGTHS:")
        lines.append(json.dumps(core_strengths, ensure_ascii=False, indent=2) if core_strengths else "[]")
        lines.append("CORE CRITIQUES:")
        lines.append(json.dumps(core_critiques, ensure_ascii=False, indent=2) if core_critiques else "[]")
        if secondary_critiques:
            lines.append("SECONDARY CONTEXT:")
            lines.append(json.dumps(secondary_critiques, ensure_ascii=False, indent=2))
        lines.append(
            "Rules: 1 = very poor, 5 = strong; reference the anchor IDs in reasoning; do not use any deterministic fallback formula."
        )
        return "\n".join(lines)

    def _score_dimension_with_llm(
        self,
        report: MethodologyCritiqueReport,
        dimension: CritiqueDimension,
        evidence: Dict[str, Any],
        pass_id: int,
    ) -> Dict[str, Any]:
        prompt = self._build_dimension_prompt(pass_id, dimension, evidence)
        self._last_pass_prompts[pass_id] = prompt

        if self.custom_llm_fn is not None:
            payload = self._call_custom_llm(pass_id, dimension, evidence, prompt)
            result = self._normalize_dimension_payload(payload, dimension)
            if result is None:
                raise RuntimeError(
                    f"Custom LLM scorer returned invalid structured output for pass {pass_id} / {dimension.value}."
                )
            return result

        provider = self.provider.lower()
        if provider == "openai":
            return self._call_openai_dimension(pass_id, dimension, prompt)
        if provider == "anthropic":
            return self._call_anthropic_dimension(pass_id, dimension, prompt)
        if provider == "gemini":
            return self._call_gemini_dimension(pass_id, dimension, prompt)

        raise RuntimeError(f"Unsupported scoring provider: {self.provider}")

    def _call_custom_llm(
        self, pass_id: int, dimension: CritiqueDimension, evidence: Dict[str, Any], prompt: str
    ) -> Any:
        try:
            return self.custom_llm_fn(pass_id, dimension, evidence, prompt)
        except TypeError:
            try:
                return self.custom_llm_fn(pass_id, prompt, evidence)
            except TypeError:
                return self.custom_llm_fn(pass_id, prompt)

    def _normalize_dimension_payload(self, payload: Any, dimension: CritiqueDimension) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None

        if isinstance(payload, DimensionScore):
            return {
                "rating": payload.rating,
                "reasoning": payload.reasoning,
                "based_on_anchor_ids": payload.based_on_anchor_ids,
            }

        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str):
            stripped = payload.strip()
            if "```json" in stripped:
                stripped = stripped.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in stripped:
                stripped = stripped.split("```", 1)[1].split("```", 1)[0].strip()
            try:
                data = json.loads(stripped)
            except Exception:
                return None
        else:
            return None

        try:
            rating = int(data.get("rating", data.get("score", 3)))
            anchors = data.get("based_on_anchor_ids") or data.get("anchor_ids") or []
            reasoning = str(data.get("reasoning", "No reasoning provided."))
            return {
                "rating": rating,
                "reasoning": reasoning,
                "based_on_anchor_ids": list(anchors),
            }
        except Exception:
            return None

    def _call_openai_dimension(self, pass_id: int, dimension: CritiqueDimension, prompt: str) -> Dict[str, Any]:
        import openai

        system_prompt = (
            "You are an expert scientific methodology reviewer. Output valid JSON only with fields "
            "rating, reasoning, based_on_anchor_ids."
        )
        if pass_id == 1:
            system_prompt += " Prefer a fair neutral review."
        else:
            system_prompt += " Be a skeptical critic and separate true strengths from weak or untested assumptions."

        client = openai.OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model if "gpt" in self.model.lower() else "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature_for_pass(pass_id),
            response_format={"type": "json_object"},
        )
        payload = response.choices[0].message.content
        normalized = self._normalize_dimension_payload(payload, dimension)
        if normalized is None:
            raise RuntimeError(f"OpenAI scoring for {dimension.value}, pass {pass_id} returned invalid JSON.")
        return normalized

    def _call_anthropic_dimension(self, pass_id: int, dimension: CritiqueDimension, prompt: str) -> Dict[str, Any]:
        import anthropic

        system_prompt = (
            "You are an expert scientific methodology reviewer. Return valid JSON with fields "
            "rating, reasoning, based_on_anchor_ids."
        )
        client = anthropic.Anthropic(api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model if "claude" in self.model.lower() else "claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system=system_prompt + (" Prefer a neutral review." if pass_id == 1 else " Be skeptical and critical."),
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature_for_pass(pass_id),
        )
        payload = response.content[0].text
        normalized = self._normalize_dimension_payload(payload, dimension)
        if normalized is None:
            raise RuntimeError(f"Anthropic scoring for {dimension.value}, pass {pass_id} returned invalid JSON.")
        return normalized

    def _call_gemini_dimension(self, pass_id: int, dimension: CritiqueDimension, prompt: str) -> Dict[str, Any]:
        from google import genai

        client = genai.Client(
            api_key=self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        response = client.models.generate_content(
            model=self.model if "gemini" in self.model.lower() else "gemini-2.0-flash",
            contents=(
                "You are an expert scientific methodology reviewer. Return valid JSON with keys rating, reasoning, based_on_anchor_ids. "
                + ("Prefer a neutral review." if pass_id == 1 else "Be skeptical and critical.")
                + "\n\n"
                + prompt
            ),
        )
        payload = getattr(response, "text", None)
        normalized = self._normalize_dimension_payload(payload, dimension)
        if normalized is None:
            raise RuntimeError(f"Gemini scoring for {dimension.value}, pass {pass_id} returned invalid JSON.")
        return normalized

    def _compute_scoring_agreement(
        self, pass_1: Dict[str, int], pass_2: Dict[str, int]
    ) -> Tuple[float, float, float]:
        dims = ["reproducibility", "assumptions", "limitations", "appropriateness"]
        exact_matches = sum(1 for d in dims if pass_1[d] == pass_2[d])
        near_misses = sum(1 for d in dims if abs(pass_1[d] - pass_2[d]) <= 1)
        total_delta = sum(abs(pass_1[d] - pass_2[d]) for d in dims)

        exact_match_rate = round(exact_matches / len(dims), 4)
        near_miss_rate = round(near_misses / len(dims), 4)
        mean_delta = round(total_delta / len(dims), 4)

        return exact_match_rate, near_miss_rate, mean_delta

    def _normalize_dimension_payload(self, payload: Any, dimension: CritiqueDimension) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None

        if isinstance(payload, DimensionScore):
            return {
                "rating": payload.rating,
                "reasoning": payload.reasoning,
                "based_on_anchor_ids": payload.based_on_anchor_ids,
            }

        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, str):
            stripped = payload.strip()
            if "```json" in stripped:
                stripped = stripped.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in stripped:
                stripped = stripped.split("```", 1)[1].split("```", 1)[0].strip()
            try:
                data = json.loads(stripped)
            except Exception:
                return None
        else:
            return None

        try:
            rating = int(data.get("rating", data.get("score", 3)))
            anchors = data.get("based_on_anchor_ids") or data.get("anchor_ids") or []
            reasoning = str(data.get("reasoning", "No reasoning provided."))
            return {
                "rating": rating,
                "reasoning": reasoning,
                "based_on_anchor_ids": list(anchors),
            }
        except Exception:
            return None
        import openai

        system_prompt = (
            "You are an expert scientific methodology reviewer. Output valid JSON only with fields "
            "rating, reasoning, based_on_anchor_ids."
        )
        if pass_id == 1:
            system_prompt += " Prefer a fair neutral review."
        else:
            system_prompt += " Be a skeptical critic and separate true strengths from weak or untested assumptions."

        client = openai.OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model if "gpt" in self.model.lower() else "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature_for_pass(pass_id),
            response_format={"type": "json_object"},
        )
        payload = response.choices[0].message.content
        normalized = self._normalize_dimension_payload(payload, dimension)
        if normalized is None:
            raise RuntimeError(f"OpenAI scoring for {dimension.value}, pass {pass_id} returned invalid JSON.")
        return normalized

    def _call_anthropic_dimension(self, pass_id: int, dimension: CritiqueDimension, prompt: str) -> Dict[str, Any]:
        import anthropic

        system_prompt = (
            "You are an expert scientific methodology reviewer. Return valid JSON with fields rating, reasoning, based_on_anchor_ids."
        )
        client = anthropic.Anthropic(api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model if "claude" in self.model.lower() else "claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system=system_prompt + (" Prefer a neutral review." if pass_id == 1 else " Be skeptical and critical."),
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature_for_pass(pass_id),
        )
        payload = response.content[0].text
        normalized = self._normalize_dimension_payload(payload, dimension)
        if normalized is None:
            raise RuntimeError(f"Anthropic scoring for {dimension.value}, pass {pass_id} returned invalid JSON.")
        return normalized

    def _call_gemini_dimension(self, pass_id: int, dimension: CritiqueDimension, prompt: str) -> Dict[str, Any]:
        from google import genai

        client = genai.Client(api_key=self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model=self.model if "gemini" in self.model.lower() else "gemini-2.0-flash",
            contents=(
                "You are an expert scientific methodology reviewer. Return valid JSON with keys rating, reasoning, based_on_anchor_ids. "
                + ("Prefer a neutral review." if pass_id == 1 else "Be skeptical and critical.")
                + "\n\n"
                + prompt
            ),
        )
        payload = getattr(response, "text", None)
        normalized = self._normalize_dimension_payload(payload, dimension)
        if normalized is None:
            raise RuntimeError(f"Gemini scoring for {dimension.value}, pass {pass_id} returned invalid JSON.")
        return normalized
    def _compute_scoring_agreement(
        self, pass_1: Dict[str, int], pass_2: Dict[str, int]
    ) -> Tuple[float, float, float]:
        """Compute exact match rate, near-miss rate (|delta| <= 1), and mean absolute delta."""
        dims = ["reproducibility", "assumptions", "limitations", "appropriateness"]
        exact_matches = sum(1 for d in dims if pass_1[d] == pass_2[d])
        near_misses = sum(1 for d in dims if abs(pass_1[d] - pass_2[d]) <= 1)
        total_delta = sum(abs(pass_1[d] - pass_2[d]) for d in dims)

        exact_match_rate = round(exact_matches / len(dims), 4)
        near_miss_rate = round(near_misses / len(dims), 4)
        mean_delta = round(total_delta / len(dims), 4)

        return exact_match_rate, near_miss_rate, mean_delta
