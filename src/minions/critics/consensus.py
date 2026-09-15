"""Consensus Aggregator for Multi-Run Methodology Critic.

Classifies candidate critique and strength points into:
- CORE: Appears in >= 2/3 independent runs (same anchor_id + critique_dimension).
- SECONDARY: Appears in only 1/3 runs.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from minions.schema.critique import (
    ConsensusType,
    CritiquePoint,
    DroppedCritiquePoint,
    MethodologyCritiqueReport,
)


class ConsensusAggregator:
    """Aggregates multiple independent MethodologyCritic runs into a consensus report."""

    def __init__(self, core_threshold_ratio: float = 0.60):
        """Initialize Consensus Aggregator.

        Args:
            core_threshold_ratio: Minimum fraction of runs required for 'core' classification (default 0.60 for >=2/3).
        """
        self.core_threshold_ratio = core_threshold_ratio

    def aggregate_runs(
        self, reports: List[MethodologyCritiqueReport]
    ) -> MethodologyCritiqueReport:
        """Aggregate N independent runs on the same paper into a single consensus report."""
        if not reports:
            raise ValueError("Must provide at least 1 MethodologyCritiqueReport to aggregate.")

        n_runs = len(reports)
        min_core_runs = 2 if n_runs >= 3 else 1

        # 1. Aggregate and classify critiques
        consensus_critiques = self._cluster_and_classify(
            [r.critiques for r in reports], min_core_runs
        )

        # 2. Aggregate and classify strengths
        consensus_strengths = self._cluster_and_classify(
            [r.strengths for r in reports], min_core_runs
        )

        # 3. Collect all unique dropped points
        all_dropped: List[DroppedCritiquePoint] = []
        seen_dropped = set()
        for r in reports:
            for dp in r.dropped_points:
                key = (tuple(dp.point.anchor_ids), dp.point.quoted_evidence.strip().lower(), dp.drop_reason)
                if key not in seen_dropped:
                    seen_dropped.add(key)
                    all_dropped.append(dp)

        # 4. Compute unified hallucination rate across all runs
        total_gen = sum(len(r.critiques) + len(r.strengths) for r in reports) + len(all_dropped)
        hallucination_rate = round(len(all_dropped) / total_gen, 4) if total_gen > 0 else 0.0

        # 5. Build base report
        first_report = reports[0]
        summary = self._generate_consensus_summary(
            consensus_critiques, consensus_strengths, n_runs, len(all_dropped)
        )

        return MethodologyCritiqueReport(
            schema_version=first_report.schema_version,
            paper_title=first_report.paper_title,
            target_sections=first_report.target_sections,
            critiques=consensus_critiques,
            strengths=consensus_strengths,
            dropped_points=all_dropped,
            hallucination_rate=hallucination_rate,
            summary=summary,
            runs_evaluated_count=n_runs,
        )

    def _cluster_and_classify(
        self, runs_points: List[List[CritiquePoint]], min_core_runs: int
    ) -> List[CritiquePoint]:
        """Cluster points across runs by (anchor_id, dimension) and classify consensus."""
        # Key: (primary_anchor_id, dimension) -> List of (run_idx, CritiquePoint)
        clusters: Dict[Tuple[str, str], List[Tuple[int, CritiquePoint]]] = defaultdict(list)

        for run_idx, points in enumerate(runs_points):
            # Track seen keys within the same run to avoid double-counting within a single run
            seen_in_run: Set[Tuple[str, str]] = set()
            for p in points:
                primary_aid = p.anchor_ids[0] if p.anchor_ids else ""
                key = (primary_aid, p.critique_dimension.value)
                if key not in seen_in_run:
                    seen_in_run.add(key)
                    clusters[key].append((run_idx, p))

        classified_points: List[CritiquePoint] = []

        for (aid, dim), run_matches in clusters.items():
            run_count = len(run_matches)
            consensus_level = ConsensusType.CORE if run_count >= min_core_runs else ConsensusType.SECONDARY

            # Select the most representative candidate (prefer highest confidence, longest quote)
            best_point = max(
                (p for _, p in run_matches),
                key=lambda pt: (
                    2 if pt.confidence.value == "high" else (1 if pt.confidence.value == "medium" else 0),
                    len(pt.quoted_evidence),
                    len(pt.critique_text),
                ),
            )

            # Create new point with assigned consensus
            updated_point = CritiquePoint(
                anchor_ids=best_point.anchor_ids,
                quoted_evidence=best_point.quoted_evidence,
                critique_dimension=best_point.critique_dimension,
                critique_text=best_point.critique_text,
                confidence=best_point.confidence,
                consensus=consensus_level,
            )
            classified_points.append(updated_point)

        # Sort: core first, then by dimension, then by anchor ID
        classified_points.sort(
            key=lambda p: (
                0 if p.consensus == ConsensusType.CORE else 1,
                p.critique_dimension.value,
                p.anchor_ids[0] if p.anchor_ids else "",
            )
        )
        return classified_points

    def _generate_consensus_summary(
        self,
        critiques: List[CritiquePoint],
        strengths: List[CritiquePoint],
        n_runs: int,
        dropped_count: int,
    ) -> str:
        """Generate high-level synthesis of consensus findings across runs."""
        core_critiques = [c for c in critiques if c.consensus == ConsensusType.CORE]
        sec_critiques = [c for c in critiques if c.consensus == ConsensusType.SECONDARY]
        core_strengths = [s for s in strengths if s.consensus == ConsensusType.CORE]
        sec_strengths = [s for s in strengths if s.consensus == ConsensusType.SECONDARY]

        return (
            f"Consensus aggregation across N={n_runs} independent runs identified "
            f"{len(core_critiques)} core critiques (present in >=2/3 runs), {len(sec_critiques)} secondary critiques, "
            f"{len(core_strengths)} core strengths, and {len(sec_strengths)} secondary strengths. "
            f"Grounding verification retained {len(critiques) + len(strengths)} verified points ({dropped_count} dropped)."
        )
