from minions.critics.results import ResultsCritic


PAPER_TEXT = """
We evaluated the proposed model on the CIFAR-10 dataset.
The dataset was divided into training and test sets.
We report accuracy and top-5 accuracy for evaluation.
"""


class FakeMetadata:
    title = "Fake Results Paper"


class FakeBlock:
    id = "results_b01"
    text = PAPER_TEXT
    page_number = 1


class FakeSection:
    id = "section_results"
    canonical_type = "results"
    level = 1
    heading_title = "Results"
    content_blocks = [FakeBlock()]
    raw_text = PAPER_TEXT
    figure_ids = []
    table_ids = []
    equation_ids = []
    subsections = []


class FakePaper:
    metadata = FakeMetadata()
    sections = [FakeSection()]
    figures = {}
    tables = {}
    equations = {}


def valid_llm(prompt, system_prompt):
    return """
{
    "critiques": [
        {
            "anchor_ids": ["results_b01"],
            "quoted_evidence": "We report accuracy and top-5 accuracy",
            "critique_dimension": "evaluation_rigor",
            "critique_text": "The evaluation reports multiple accuracy metrics, which provides useful evidence for model performance.",
            "confidence": "high"
        }
    ],
    "strengths": []
}
"""


def hallucinated_llm(prompt, system_prompt):
    return """
{
    "critiques": [
        {
            "anchor_ids": ["results_b01"],
            "quoted_evidence": "We used Adam optimizer with weight decay 0.05",
            "critique_dimension": "experimental_reproducibility",
            "critique_text": "The paper clearly reports the optimizer and weight decay.",
            "confidence": "high"
        }
    ],
    "strengths": []
}
"""


def test_results_critic_grounding():
    critic = ResultsCritic(custom_llm_fn=valid_llm)

    report = critic.critique_paper(FakePaper())

    print()
    print("=== VALID GROUNDING DEBUG ===")
    print("Critiques:", report.critiques)
    print("Dropped:", report.dropped_points)
    print("Hallucination rate:", report.hallucination_rate)
    print("==============================")
    print()

    assert len(report.critiques) == 1
    assert len(report.dropped_points) == 0
    assert report.hallucination_rate == 0.0

    print("Valid grounding test: PASS")


def test_results_critic_drops_hallucination():
    critic = ResultsCritic(custom_llm_fn=hallucinated_llm)

    report = critic.critique_paper(FakePaper())

    print()
    print("=== HALLUCINATION DEBUG ===")
    print("Critiques:", report.critiques)
    print("Dropped:", report.dropped_points)
    print("Hallucination rate:", report.hallucination_rate)
    print("============================")
    print()

    assert len(report.critiques) == 0
    assert len(report.dropped_points) == 1
    assert report.hallucination_rate == 1.0

    print("Hallucination filtering test: PASS")


if __name__ == "__main__":
    test_results_critic_grounding()
    test_results_critic_drops_hallucination()

    print("All Results Critic tests passed.")