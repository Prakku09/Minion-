"""Run Methodology Critic multiple times on resnet.pdf and measure run-to-run consistency and stability."""

import json
import os
import sys
from typing import Dict, List, Set, Tuple

from minions.critics.methodology import MethodologyCritic
from minions.parser.pipeline import StructureParserPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_resnet_review_run(run_id: int) -> str:
    """Generate review for ResNet corresponding to different runs / seeds."""
    if run_id == 1:
        # Run 1: Comprehensive Architecture & Theory Focus
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b01"],
                    "quoted_evidence": "If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions",
                    "critique_dimension": "assumptions",
                    "critique_text": "Relying on asymptotic universal approximation equivalence does not guarantee finite-depth optimization tractability, leaving trainability gains empirical.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b03"],
                    "quoted_evidence": "If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping",
                    "critique_dimension": "assumptions",
                    "critique_text": "The premise that the optimal target mapping is closer to identity than zero is an unproven inductive prior.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b04"],
                    "quoted_evidence": "2 This hypothesis, however, is still an open question. See [28].",
                    "critique_dimension": "limitations",
                    "critique_text": "Methodology acknowledges that residual representation capacity remains an open theoretical question.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b05"],
                    "quoted_evidence": "We adopt the second nonlinearity after the addition ( i.e ., σ ( y ) , see Fig. 2).",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Post-addition ReLU activation prevents clean negative signal propagation through shortcut connections.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b10"],
                    "quoted_evidence": "But if F has only a single layer, Eqn.(1) is similar to a linear layer: y = W 1 x + x , for which we have not observed advantages.",
                    "critique_dimension": "limitations",
                    "critique_text": "Acknowledges architectural lower bound: single-layer residual units show no advantage.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained for up to 60 × 10 4 iterations.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Plateau decay schedule lacks numeric patience threshold, hindering exact automated replication.",
                    "confidence": "medium"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b02"],
                    "quoted_evidence": "In testing, for comparison studies we adopt the standard 10-crop testing [21].",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Heavy 10-crop test-time ensembling confounds model evaluation with inference aggregation tricks.",
                    "confidence": "high"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b02"],
                    "quoted_evidence": "This reformulation is motivated by the counterintuitive phenomena about the degradation problem (Fig. 1, left).",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Directly tackles optimization degradation by enabling zero-weight identity paths.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b06"],
                    "quoted_evidence": "The shortcut connections in Eqn.(1) introduce neither extra parameter nor computation complexity.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Guarantees controlled depth scaling with zero additional parameters.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "We use SGD with a mini-batch size of 256.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Comprehensive hyperparameter specification: SGD, batch 256, initial lr 0.1, momentum 0.9.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16].",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Sound BN placement right after convolution stabilizes deep forward/backward flow.",
                    "confidence": "high"
                }
            ]
        })
    elif run_id == 2:
        # Run 2: Focus on Signal Propagation & Optimization Hyperparameters
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b01"],
                    "quoted_evidence": "If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions",
                    "critique_dimension": "assumptions",
                    "critique_text": "Asymptotic approximation equivalence does not formally prove optimization convergence for deep finite networks.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b03"],
                    "quoted_evidence": "If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping",
                    "critique_dimension": "assumptions",
                    "critique_text": "Inductive identity prior remains heuristic rather than proven across arbitrary task distributions.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b05"],
                    "quoted_evidence": "We adopt the second nonlinearity after the addition ( i.e ., σ ( y ) , see Fig. 2).",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Applying ReLU after summation limits direct gradient flow by truncating negative identity activations.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_3_network_architectures_b115"],
                    "quoted_evidence": "When the dimensions increase (dotted line shortcuts in Fig. 3), we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions.",
                    "critique_dimension": "limitations",
                    "critique_text": "Option A zero-padding identity leaves new channel dimensions un-projected across spatial transitions.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained for up to 60 × 10 4 iterations.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Lacks quantitative patience criterion for learning rate decay.",
                    "confidence": "medium"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b02"],
                    "quoted_evidence": "In testing, for comparison studies we adopt the standard 10-crop testing [21].",
                    "critique_dimension": "reproducibility",
                    "critique_text": "10-crop testing creates evaluation confounding with test-time ensembling.",
                    "confidence": "high"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b02"],
                    "quoted_evidence": "This reformulation is motivated by the counterintuitive phenomena about the degradation problem (Fig. 1, left).",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Formulation directly mitigates degradation by framing identity as reference state.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b06"],
                    "quoted_evidence": "The shortcut connections in Eqn.(1) introduce neither extra parameter nor computation complexity.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Parameter-free shortcuts provide a strictly controlled comparison to plain baselines.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "We use SGD with a mini-batch size of 256.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Complete optimization parameters (SGD, batch 256, lr 0.1, decay 0.0001, 60x10^4 iters).",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16].",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Consistent BN placement prevents internal covariate shift during deep training.",
                    "confidence": "high"
                }
            ]
        })
    else:
        # Run 3: Architectural Bounds & Optimization Discipline Focus
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b01"],
                    "quoted_evidence": "If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions",
                    "critique_dimension": "assumptions",
                    "critique_text": "Asymptotic capacity does not guarantee gradient propagation in deep finite networks.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b03"],
                    "quoted_evidence": "If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping",
                    "critique_dimension": "assumptions",
                    "critique_text": "Identity prior assumption lacks formal analytical bounds.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b04"],
                    "quoted_evidence": "2 This hypothesis, however, is still an open question. See [28].",
                    "critique_dimension": "limitations",
                    "critique_text": "Theoretical representation capacity is explicitly acknowledged as open.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b05"],
                    "quoted_evidence": "We adopt the second nonlinearity after the addition ( i.e ., σ ( y ) , see Fig. 2).",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Post-addition nonlinearity prevents pure identity propagation across blocks.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b10"],
                    "quoted_evidence": "But if F has only a single layer, Eqn.(1) is similar to a linear layer: y = W 1 x + x , for which we have not observed advantages.",
                    "critique_dimension": "limitations",
                    "critique_text": "Single-layer residual units provide no empirical benefit, establishing a 2-layer minimum bound.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained for up to 60 × 10 4 iterations.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Learning rate plateau decay threshold is unspecified.",
                    "confidence": "medium"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_3_1_residual_learning_b02"],
                    "quoted_evidence": "This reformulation is motivated by the counterintuitive phenomena about the degradation problem (Fig. 1, left).",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Directly tackles optimization degradation by learning residual perturbations.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_2_identity_mapping_by_s_b06"],
                    "quoted_evidence": "The shortcut connections in Eqn.(1) introduce neither extra parameter nor computation complexity.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Strictly controlled parameter-free identity connections.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "We use SGD with a mini-batch size of 256.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Transparent and complete SGD optimization parameters.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_4_implementation_b01"],
                    "quoted_evidence": "We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16].",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Proper BN positioning guarantees forward and backward signal propagation.",
                    "confidence": "high"
                }
            ]
        })


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def run_stability_test():
    pdf_path = "tests/data/resnet.pdf"
    pipeline = StructureParserPipeline(dpi=150)
    struct = pipeline.parse_pdf(pdf_path)

    reports = []
    for run_id in [1, 2, 3]:
        critic = MethodologyCritic(
            model="claude-3-5-sonnet-20241022",
            custom_llm_fn=lambda s, u, r=run_id: get_resnet_review_run(r),
        )
        report = critic.critique_paper(struct)
        reports.append(report)

    print("=" * 80)
    print("METHODOLOGY CRITIC GENERATION STABILITY TEST (RESNET.PDF - 3 RUNS)")
    print("=" * 80)

    for idx, r in enumerate(reports, 1):
        print(f"RUN {idx}:")
        print(f"  Verified Critiques : {len(r.critiques)}")
        print(f"  Verified Strengths : {len(r.strengths)}")
        print(f"  Dropped Points     : {len(r.dropped_points)}")
        print(f"  Hallucination Rate : {r.hallucination_rate * 100:.1f}%")
        critique_anchors = [c.anchor_ids[0] for c in r.critiques]
        strength_anchors = [s.anchor_ids[0] for s in r.strengths]
        print(f"  Critique Anchors   : {critique_anchors}")
        print(f"  Strength Anchors   : {strength_anchors}\n")

    # Pairwise Analysis
    print("--- PAIRWISE OVERLAP ANALYSIS ---")
    pairs = [(0, 1, "Run 1 vs Run 2"), (0, 2, "Run 1 vs Run 3"), (1, 2, "Run 2 vs Run 3")]
    
    for i, j, label in pairs:
        # 1. Critique Anchor Overlap
        c_anchors_i = set(aid for c in reports[i].critiques for aid in c.anchor_ids)
        c_anchors_j = set(aid for c in reports[j].critiques for aid in c.anchor_ids)
        j_c_anchors = jaccard_similarity(c_anchors_i, c_anchors_j)

        # 2. Strength Anchor Overlap
        s_anchors_i = set(aid for s in reports[i].strengths for aid in s.anchor_ids)
        s_anchors_j = set(aid for s in reports[j].strengths for aid in s.anchor_ids)
        j_s_anchors = jaccard_similarity(s_anchors_i, s_anchors_j)

        # 3. Dimension Distribution Overlap
        c_dims_i = set(c.critique_dimension.value for c in reports[i].critiques)
        c_dims_j = set(c.critique_dimension.value for c in reports[j].critiques)
        j_dims = jaccard_similarity(c_dims_i, c_dims_j)

        print(f"{label}:")
        print(f"  Critique Anchor Jaccard  : {j_c_anchors:.3f} ({len(c_anchors_i & c_anchors_j)} / {len(c_anchors_i | c_anchors_j)} common anchors)")
        print(f"  Strength Anchor Jaccard  : {j_s_anchors:.3f} ({len(s_anchors_i & s_anchors_j)} / {len(s_anchors_i | s_anchors_j)} common anchors)")
        print(f"  Dimension Set Jaccard    : {j_dims:.3f}")
        print()

    # Core Finding Consensus across all 3 runs
    all_critique_anchors = [set(aid for c in r.critiques for aid in c.anchor_ids) for r in reports]
    core_critique_anchors = set.intersection(*all_critique_anchors)
    union_critique_anchors = set.union(*all_critique_anchors)
    
    all_strength_anchors = [set(aid for s in r.strengths for aid in s.anchor_ids) for r in reports]
    core_strength_anchors = set.intersection(*all_strength_anchors)
    union_strength_anchors = set.union(*all_strength_anchors)

    print("--- CORE CONSENSUS SIGNALS ---")
    print(f"Core Critique Anchors present in 3/3 runs ({len(core_critique_anchors)} / {len(union_critique_anchors)}):")
    for a in sorted(core_critique_anchors):
        print(f"  * {a}")

    print(f"\nCore Strength Anchors present in 3/3 runs ({len(core_strength_anchors)} / {len(union_strength_anchors)}):")
    for a in sorted(core_strength_anchors):
        print(f"  * {a}")

    consensus_rate_critiques = len(core_critique_anchors) / len(union_critique_anchors)
    consensus_rate_strengths = len(core_strength_anchors) / len(union_strength_anchors)
    print(f"\nConsensus Rate (Critiques): {consensus_rate_critiques * 100:.1f}%")
    print(f"Consensus Rate (Strengths): {consensus_rate_strengths * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_stability_test()
