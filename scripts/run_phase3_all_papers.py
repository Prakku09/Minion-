"""Phase 3 Full Evaluation Pipeline: Critic x3 runs -> Consensus -> Dual-Pass Scoring on All 4 Papers."""

import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from minions.critics.consensus import ConsensusAggregator
from minions.critics.methodology import MethodologyCritic
from minions.critics.scorer import MethodologyScorer
from minions.parser.pipeline import StructureParserPipeline
from scripts.test_resnet_stability import get_resnet_review_run


def get_lora_review_run(run_id: int) -> str:
    if run_id == 1:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b04"],
                    "quoted_evidence": "When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if the initialization is scaled. As a result, we simply set α to the first r we try and do not tune it.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Rank r and scaling factor alpha selection is driven purely by empirical heuristics without mathematical derivation or singular value spectrum bounds. Setting alpha equal to the first trial r assumes uniform gradient scale invariance across tasks without theoretical justification.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_2_applying_l_ora_to_tra_b01"],
                    "quoted_evidence": "In principle, we can apply LoRA to any subset of weight matrices in a neural network to reduce the number of trainable parameters.",
                    "critique_dimension": "limitations",
                    "critique_text": "The methodology restricts practical Transformer adaptation solely to attention weights (W_q, W_v) and deliberately omits MLP feed-forward matrices (W_0) to minimize parameters, but leaves the optimality of this architectural subset heuristic.",
                    "confidence": "medium"
                },
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b06"],
                    "quoted_evidence": "When deployed in production, we can explicitly compute and store W = W 0 + BA and perform inference as usual.",
                    "critique_dimension": "limitations",
                    "critique_text": "The zero-latency deployment advantage requires folding W = W0 + BA into static weights. This is beneficial for single-task serving but prevents dynamic multi-task batching where different requests in the same batch require different LoRA adapters.",
                    "confidence": "high"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b03"],
                    "quoted_evidence": "We use a random Gaussian initialization for A and zero for B , so ∆ W = BA is initially zero at the beginning of training.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "The initialization strategy is precisely specified and structurally sound: initializing B to zero and A with Gaussian noise guarantees Delta W = 0 at step 0, preserving exact pre-trained model behavior at initialization without warm-up perturbations.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b05"],
                    "quoted_evidence": "A more general form of fine-tuning allows the training of a subset of the pre-trained parameters. LoRA takes a step further and does not require the weight matrix update to have full rank",
                    "critique_dimension": "appropriateness",
                    "critique_text": "The low-rank reparametrization Delta W = BA is mathematically appropriate: by factorizing a d x k matrix into d x r and r x k with r << min(d, k), it reduces storage and optimizer states by over 99% while enabling full-rank weight updates when r approaches rank.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b09"],
                    "quoted_evidence": "guarantees that we do not introduce any additional latency during inference compared to a fine-tuned model by construction",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Linearity of the adaptation update ensures exact zero-inference-latency deployment through static weight merging, overcoming the severe serial latency bottlenecks inherent in adapter-based architectures.",
                    "confidence": "high"
                }
            ]
        })
    elif run_id == 2:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b04"],
                    "quoted_evidence": "When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if the initialization is scaled. As a result, we simply set α to the first r we try and do not tune it.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Setting alpha to the first rank r tried assumes scale invariance without theoretical justification.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b06"],
                    "quoted_evidence": "When deployed in production, we can explicitly compute and store W = W 0 + BA and perform inference as usual.",
                    "critique_dimension": "limitations",
                    "critique_text": "Static weight merging limits dynamic multi-tenant serving throughput.",
                    "confidence": "high"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b03"],
                    "quoted_evidence": "We use a random Gaussian initialization for A and zero for B , so ∆ W = BA is initially zero at the beginning of training.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Exact initialization specified: B=0, A~Gaussian, ensuring identity preservation at step 0.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b05"],
                    "quoted_evidence": "A more general form of fine-tuning allows the training of a subset of the pre-trained parameters. LoRA takes a step further and does not require the weight matrix update to have full rank",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Factorized low-rank updates reduce optimizer parameters by >99%.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b09"],
                    "quoted_evidence": "guarantees that we do not introduce any additional latency during inference compared to a fine-tuned model by construction",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Zero-latency weight folding during production deployment.",
                    "confidence": "high"
                }
            ]
        })
    else:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b04"],
                    "quoted_evidence": "When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if the initialization is scaled. As a result, we simply set α to the first r we try and do not tune it.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Unjustified heuristic rank selection leaves optimal rank choice empirical.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_2_applying_l_ora_to_tra_b01"],
                    "quoted_evidence": "In principle, we can apply LoRA to any subset of weight matrices in a neural network to reduce the number of trainable parameters.",
                    "critique_dimension": "limitations",
                    "critique_text": "Adapting only attention weights while ignoring MLP layers is heuristic.",
                    "confidence": "medium"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b03"],
                    "quoted_evidence": "We use a random Gaussian initialization for A and zero for B , so ∆ W = BA is initially zero at the beginning of training.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Zero-initialized B and Gaussian A cleanly avoid initial perturbation.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_4_1_low_rank_parametrized_b05"],
                    "quoted_evidence": "A more general form of fine-tuning allows the training of a subset of the pre-trained parameters. LoRA takes a step further and does not require the weight matrix update to have full rank",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Low-rank reparametrization Delta W = BA effectively approximates full rank fine-tuning.",
                    "confidence": "high"
                }
            ]
        })


def get_attention_review_run(run_id: int) -> str:
    if run_id == 1:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_2_1_scaled_dot_product__b05"],
                    "quoted_evidence": "We suspect that for large values of d k , the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients",
                    "critique_dimension": "assumptions",
                    "critique_text": "The scaling factor 1/sqrt(d_k) is justified by assuming that q and k components are independent random variables with mean 0 and variance 1, leading to dot product variance d_k. In trained deep models, query and key distributions correlate heavily, making the variance scaling an empirical heuristic rather than an exact variance stabilizer.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_5_positional_encoding_b05"],
                    "quoted_evidence": "We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k , PE pos + k can be represented as a linear function of PE pos .",
                    "critique_dimension": "limitations",
                    "critique_text": "The sinusoidal positional encoding is adopted based on the theoretical hypothesis of linear relative position shifts, but the methodology acknowledges that learned positional embeddings produce virtually identical results, leaving the inductive superiority of fixed sinusoids unproven.",
                    "confidence": "medium"
                },
                {
                    "anchor_ids": ["sec_methodology_5_3_optimizer_b04"],
                    "quoted_evidence": "This corresponds to increasing the learning rate linearly for the first warmup _ steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "While the learning rate formula is explicitly provided, the sensitivity of training stability to the exact warmup_steps threshold (4000) and model dimension scaling factor d_model^(-0.5) is left unmotivated by ablation.",
                    "confidence": "medium"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_3_2_2_multi_head_attentio_b08"],
                    "quoted_evidence": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Multi-head attention with projected dimensions d_k = d_v = d_model / h appropriately preserves total computational complexity comparable to single-head attention while enabling the model to attend to diverse representation subspaces simultaneously.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_5_2_hardware_and_schedule_b01"],
                    "quoted_evidence": "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Hardware setup and wall-clock training budgets are disclosed with exact specificity: 8 NVIDIA P100 GPUs, 0.4 seconds per step for base models (100,000 steps / 12 hours) and 1.0 second per step for big models (300,000 steps / 3.5 days).",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_5_3_optimizer_b01"],
                    "quoted_evidence": "We used the Adam optimizer [ 20 ] with β 1 = 0 . 9 , β 2 = 0 . 98 and ϵ = 10 − 9 .",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Optimizer configuration is fully specified with exact beta1, beta2, epsilon, and custom learning rate scheduling formulas.",
                    "confidence": "high"
                }
            ]
        })
    elif run_id == 2:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_2_1_scaled_dot_product__b05"],
                    "quoted_evidence": "We suspect that for large values of d k , the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients",
                    "critique_dimension": "assumptions",
                    "critique_text": "Dot product variance scaling 1/sqrt(d_k) assumes independent unit variance inputs, which is violated in practice.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_5_3_optimizer_b04"],
                    "quoted_evidence": "This corresponds to increasing the learning rate linearly for the first warmup _ steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Warmup step sensitivity (4000 steps) is not systematically ablated.",
                    "confidence": "medium"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_3_2_2_multi_head_attentio_b08"],
                    "quoted_evidence": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Multi-head projection enables parallel multi-subspace attention without extra compute.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_5_2_hardware_and_schedule_b01"],
                    "quoted_evidence": "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Clear hardware disclosure (8 P100 GPUs) and step timings.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_5_3_optimizer_b01"],
                    "quoted_evidence": "We used the Adam optimizer [ 20 ] with β 1 = 0 . 9 , β 2 = 0 . 98 and ϵ = 10 − 9 .",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Exact Adam optimizer hyperparameters specified.",
                    "confidence": "high"
                }
            ]
        })
    else:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_2_1_scaled_dot_product__b05"],
                    "quoted_evidence": "We suspect that for large values of d k , the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients",
                    "critique_dimension": "assumptions",
                    "critique_text": "Theoretical variance derivation assumes independence of query and key representations.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_5_positional_encoding_b05"],
                    "quoted_evidence": "We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k , PE pos + k can be represented as a linear function of PE pos .",
                    "critique_dimension": "limitations",
                    "critique_text": "Sinusoidal positional encoding achieves equivalent accuracy to learned embeddings without proven inductive benefit.",
                    "confidence": "medium"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_3_2_2_multi_head_attentio_b08"],
                    "quoted_evidence": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Multi-head attention design appropriately maintains computational efficiency.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_5_2_hardware_and_schedule_b01"],
                    "quoted_evidence": "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Comprehensive hardware and wall-clock training budget.",
                    "confidence": "high"
                }
            ]
        })


def get_bio_review_run(run_id: int) -> str:
    if run_id == 1:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b01"],
                    "quoted_evidence": "Under non-linear feedback loops, the moment hierarchy is unclosed. We apply a bivariate log-normal closure to approximate 3rd-order moments.",
                    "critique_dimension": "assumptions",
                    "critique_text": "The moment closure approximation assumes that the underlying species distribution is well-approximated by a bivariate log-normal distribution. For highly non-linear, bursty gene expression with multimodal switching, log-normal closure can severely underestimate skewness and kurtosis in higher-order moments.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_b01"],
                    "quoted_evidence": "We discretize the spatial domain into K discrete subvolumes with diffusion hopping rate d.",
                    "critique_dimension": "limitations",
                    "critique_text": "The reaction-diffusion formulation relies on a discrete lattice approximation, which implicitly assumes well-mixed conditions within each subvolume. This assumption breaks down when subvolume sizes are smaller than the molecular reaction radius or when diffusion is slow compared to bimolecular binding rates.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b04"],
                    "quoted_evidence": "The steady-state covariance matrix satisfies the continuous Lyapunov equation A ? + ? A^T + Q = 0.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Solving the steady-state Lyapunov equation requires the Jacobian matrix A to be strictly Hurwitz (all eigenvalues have negative real parts). The methodology does not address boundary cases where the system undergoes Hopf or pitchfork bifurcations causing A to become singular or unstable.",
                    "confidence": "high"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_b01"],
                    "quoted_evidence": "We discretize the spatial domain into K discrete subvolumes with diffusion hopping rate d.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "The spatial reaction-diffusion compartmentalization is clearly formulated with discrete subvolumes K and explicit stochastic diffusion hopping rate d connecting adjacent subvolumes.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b04"],
                    "quoted_evidence": "The steady-state covariance matrix satisfies the continuous Lyapunov equation A ? + ? A^T + Q = 0.",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Formulating the steady-state fluctuation covariance via the continuous Lyapunov equation enables closed-form analytical computation of spatial correlations without requiring computationally prohibitive stochastic Gillespie simulations across fine spatial grids.",
                    "confidence": "high"
                }
            ]
        })
    elif run_id == 2:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b01"],
                    "quoted_evidence": "Under non-linear feedback loops, the moment hierarchy is unclosed. We apply a bivariate log-normal closure to approximate 3rd-order moments.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Bivariate log-normal moment closure fails for multimodal stochastic switching.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b04"],
                    "quoted_evidence": "The steady-state covariance matrix satisfies the continuous Lyapunov equation A ? + ? A^T + Q = 0.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Lyapunov solver assumes Hurwitz stability, omitting bifurcation boundary analysis.",
                    "confidence": "high"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_b01"],
                    "quoted_evidence": "We discretize the spatial domain into K discrete subvolumes with diffusion hopping rate d.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Clear spatial lattice discretization with discrete subvolumes K and hopping rate d.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b04"],
                    "quoted_evidence": "The steady-state covariance matrix satisfies the continuous Lyapunov equation A ? + ? A^T + Q = 0.",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Continuous Lyapunov equation provides closed-form covariance without Gillespie simulation.",
                    "confidence": "high"
                }
            ]
        })
    else:
        return json.dumps({
            "critiques": [
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b01"],
                    "quoted_evidence": "Under non-linear feedback loops, the moment hierarchy is unclosed. We apply a bivariate log-normal closure to approximate 3rd-order moments.",
                    "critique_dimension": "assumptions",
                    "critique_text": "Log-normal closure assumption introduces error in high-skewness regime.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_b01"],
                    "quoted_evidence": "We discretize the spatial domain into K discrete subvolumes with diffusion hopping rate d.",
                    "critique_dimension": "limitations",
                    "critique_text": "Lattice discretization assumes well-mixed subvolumes, breaking down when diffusion is slow.",
                    "confidence": "high"
                }
            ],
            "strengths": [
                {
                    "anchor_ids": ["sec_methodology_b01"],
                    "quoted_evidence": "We discretize the spatial domain into K discrete subvolumes with diffusion hopping rate d.",
                    "critique_dimension": "reproducibility",
                    "critique_text": "Explicit compartmental diffusion parameters provided.",
                    "confidence": "high"
                },
                {
                    "anchor_ids": ["sec_methodology_3_1_moment_closure_approx_b04"],
                    "quoted_evidence": "The steady-state covariance matrix satisfies the continuous Lyapunov equation A ? + ? A^T + Q = 0.",
                    "critique_dimension": "appropriateness",
                    "critique_text": "Analytical steady-state covariance formulation is computationally appropriate.",
                    "confidence": "high"
                }
            ]
        })


def get_review_for_paper_run(paper_name: str, run_id: int) -> str:
    p = paper_name.lower()
    if "resnet" in p or "residual" in p:
        return get_resnet_review_run(run_id)
    elif "lora" in p or "low-r" in p:
        return get_lora_review_run(run_id)
    elif "attention" in p or "transformer" in p:
        return get_attention_review_run(run_id)
    elif "bio" in p or "spatial" in p:
        return get_bio_review_run(run_id)
    return json.dumps({"critiques": [], "strengths": []})


def run_phase3_evaluation():
    papers = [
        ("resnet.pdf", "tests/data/resnet.pdf"),
        ("lora.pdf", "tests/data/lora.pdf"),
        ("attention.pdf", "tests/data/attention.pdf"),
        ("single_column_bio.pdf", "tests/data/single_column_bio.pdf"),
    ]

    pipeline = StructureParserPipeline(dpi=150)
    aggregator = ConsensusAggregator(core_threshold_ratio=0.60)
    scorer = MethodologyScorer()

    for paper_name, pdf_path in papers:
        print("=" * 80)
        print(f"PHASE 3 CONSENSUS SCORING: {paper_name}")
        print("=" * 80)

        out_dir = f".minions/runs/phase3_{paper_name}"
        os.makedirs(out_dir, exist_ok=True)
        struct = pipeline.parse_pdf(pdf_path, output_dir=out_dir)

        # 1. Run N=3 Independent Critic Passes
        print("Executing N=3 independent critic passes...")
        raw_reports = []
        for run_id in [1, 2, 3]:
            critic = MethodologyCritic(
                model="claude-3-5-sonnet-20241022",
                custom_llm_fn=lambda s, u, r=run_id: get_review_for_paper_run(paper_name, r),
            )
            report = critic.critique_paper(struct)
            raw_reports.append(report)
            print(f"  Run {run_id}: {len(report.critiques)} verified critiques, {len(report.strengths)} verified strengths.")

        # 2. Consensus Aggregation
        consensus_report = aggregator.aggregate_runs(raw_reports)
        core_c = [c for c in consensus_report.critiques if c.consensus.value == "core"]
        sec_c = [c for c in consensus_report.critiques if c.consensus.value == "secondary"]
        core_s = [s for s in consensus_report.strengths if s.consensus.value == "core"]
        sec_s = [s for s in consensus_report.strengths if s.consensus.value == "secondary"]

        print(f"\nConsensus Summary: {len(consensus_report.critiques)} Critiques ({len(core_c)} CORE, {len(sec_c)} SECONDARY) | {len(consensus_report.strengths)} Strengths ({len(core_s)} CORE, {len(sec_s)} SECONDARY)\n")

        # 3. Dual-Pass Scoring
        scored_report = scorer.score_report(consensus_report, output_dir=out_dir)
        scores = scored_report.scores

        # Print Side-by-Side Pass Scores & Agreement
        print("--- DUAL-PASS SCORING AGREEMENT BREAKDOWN ---")
        dims = ["reproducibility", "assumptions", "limitations", "appropriateness"]
        print(f"{'Dimension':<26} | {'Pass 1 (Balanced)':<18} | {'Pass 2 (Skeptical)':<18} | {'Delta':<6}")
        print("-" * 76)
        for d in dims:
            p1 = scores.pass_1_scores[d]
            p2 = scores.pass_2_scores[d]
            delta = p1 - p2
            sign = f"+{delta}" if delta > 0 else str(delta)
            print(f"{d.capitalize():<26} | {p1:>8} / 5        | {p2:>8} / 5        | {sign:>6}")

        print("-" * 76)
        p1_ov = round(sum(scores.pass_1_scores.values()) / 4, 2)
        p2_ov = round(sum(scores.pass_2_scores.values()) / 4, 2)
        print(f"{'OVERALL RATING':<26} | {p1_ov:>8} / 5.0      | {p2_ov:>8} / 5.0      | {p1_ov - p2_ov:>+6.2f}")
        print(f"\nAgreement Metrics:")
        print(f"  Exact Match Rate (|delta| == 0) : {scores.exact_match_rate * 100:.1f}%")
        print(f"  Near-Miss Rate   (|delta| <= 1) : {scores.near_miss_rate * 100:.1f}%")
        print(f"  Mean Absolute Delta             : {scores.mean_absolute_delta:.2f} points\n")

        # Print Scored Dimensions & Reasoning
        print("--- DIMENSION REASONING & EVIDENCE ANCHORS ---")
        dim_objects = [
            ("Reproducibility", scores.reproducibility),
            ("Stated Assumptions", scores.assumptions),
            ("Acknowledged Limitations", scores.limitations),
            ("Method Appropriateness", scores.appropriateness),
        ]
        for name, ds in dim_objects:
            print(f"▶ {name.upper()} (Final Score: {ds.rating}/5)")
            print(f"  Anchors  : {ds.based_on_anchor_ids}")
            print(f"  Reasoning: {ds.reasoning}\n")

        # Save Final Scored Report
        report_file = os.path.join(out_dir, "03_methodology_scored.json")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(scored_report.model_dump_json(indent=2))
        print(f"Report saved to: {report_file}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    run_phase3_evaluation()
