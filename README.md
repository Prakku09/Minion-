# Minions 🤖🔬

**Minions** is a multi-agent research assistant pipeline for automated, evidence-grounded scientific paper review. Instead of producing generic commentary, Minions parses academic PDFs into a structured multimodal map and critiques each section based on verifiable document evidence.

---

## 🏗️ Architecture & Pipeline

Minions is built as a pipeline of specialized agents and steps:

```mermaid
flowchart LR
    PDF["Input Research PDF"] --> Parser["1. Structure Parser\n(Multimodal & Layout Extraction)"]
    Parser --> Methodology["2. Phase 2\nMethodology Critic"]
    Methodology --> Consensus["3. Phase 3\nConsensus + Scoring"]
    Parser --> Results["4. Phase 4\nResults Critic"]
    Consensus --> Review["Evidence-Grounded\nReview Outputs"]
    Results --> Review
```

### 1. Structure Parser (Phase 1)
- **Multimodal Document Processing**: High-resolution page rendering, figure cropping, and tabular Markdown extraction.
- **Multi-Column Reading Order Sorting**: Robust column gutter detection and full-width block handling to prevent interleaved text.
- **Equation Engine**: LaTeX reconstruction with confidence threshold ($\ge 0.80$) and automatic high-resolution visual fallback.
- **Grounding Index**: Deterministic anchor IDs (`sec_methodology_b01`, `fig_1`, `tab_1`, `eq_1`) with page coordinates and verbatim text.
- **Canonical Section Mapping**: Standardizes paper sections (`abstract`, `introduction`, `methodology`, `results`, `discussion`, `conclusion`, `references`).
- **Structured Schema (v1.0.0)**: Pydantic-validated JSON representation with per-section diagnostic confidences.

### 2. Methodology Critic (Phase 2)
- **LLM-Driven Review**: Evaluates methodology sections across reproducibility, assumptions, limitations, and appropriateness.
- **Evidence Separation**: Distinguishes genuine methodological gaps from confirmatory strengths.
- **Grounding Verification**: Checks every cited anchor and quoted passage against the parsed document, dropping unsupported points.

### 3. Consensus and Scoring (Phase 3)
- **Independent Runs**: Aggregates multiple critic runs while counting each run at most once per clustered point.
- **Configurable Consensus**: Uses `ceil(n_runs * core_threshold_ratio)` to classify core findings, with a minimum threshold of one run.
- **Semantic Clustering**: Groups points by normalized critique text within each dimension, so equivalent findings with nearby block anchors are not falsely split.
- **Dual-Pass Scoring**: Uses independent balanced and skeptical LLM passes to produce 1–5 ratings, reasoning, anchor references, and agreement metrics.

### 4. Results Critic (Phase 4)
Phase 4 extends the pipeline from method review to experimental-result review. It operates independently from the methodology critic and focuses only on parsed `results` and `discussion` sections.

```mermaid
flowchart TD
    Structure["01_structure_parser.json"] --> Sections["Select Results + Discussion sections"]
    Structure --> Index["Build evidence index\n(text, figures, tables, equations)"]
    Sections --> Prompt["Build rubric prompt"]
    Index --> Prompt
    Prompt --> LLM["ResultsCritic LLM pass"]
    LLM --> Candidates["Candidate critiques + strengths"]
    Candidates --> Verify["Deterministic anchor/quote verification"]
    Verify --> Report["02_results_critic.json"]
    Verify --> Dropped["Dropped points + hallucination rate"]
```

The Phase 4 rubric has four dimensions:
- **Experimental Reproducibility**: datasets, splits, configurations, hyperparameters, hardware, evaluation procedures, and repeated runs.
- **Evaluation Rigor**: metrics, baselines, statistical evidence, variance, confidence intervals, and significance analysis.
- **Experimental Design**: ablations, baseline fairness, controls, confounders, and dataset construction.
- **Results Interpretation**: whether reported evidence supports the claims without overclaiming or ignoring contradictions.

Every Phase 4 point must cite an existing document anchor and an exact evidence quote. Validated reports are written to `02_results_critic.json`; rejected candidates are retained with their failure reason, and the report records the resulting hallucination rate. The implementation is in `src/minions/critics/results.py`, with schemas in `src/minions/schema/results_critique.py` and grounding tests in `tests/test_results_critic.py`.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Prakku09/Minion-robo.git
cd Minion-robo

# Install dependencies (Python 3.10+)
pip install -e ".[dev]"
```

---

## 🚀 Quickstart & Usage

### Parse a Scientific PDF
```bash
# Run via CLI
minions-parse path/to/paper.pdf --out .minions/runs/my_paper

# Or run via Python module
python -m minions.cli path/to/paper.pdf --out .minions/runs/my_paper
```

### Output Artifacts
The parser outputs:
- `.minions/runs/<paper_id>/01_structure_parser.json`: Complete structured JSON map.
- `.minions/runs/<paper_id>/assets/pages/`: 150–200 DPI rendered page images.
- `.minions/runs/<paper_id>/assets/figures/`: High-resolution cropped figures and plots.
- `.minions/runs/<paper_id>/assets/tables/`: Cropped tables and Markdown grids.
- `.minions/runs/<paper_id>/assets/equations/`: Cropped visual equations for fallback cases.

---

## 🧪 Running the Test Suite

Run the full automated test suite (including multi-column sorting, grounding round-trip, and validation on real papers):

```bash
pytest -v
```

### Test Coverage
1. `tests/test_schema.py`: Schema v1.0.0 construction, validation, and strict extra field rejection.
2. `tests/test_reading_order.py`: Multi-column layout reading sequence validation across vertical gutters.
3. `tests/test_grounding_roundtrip.py`: Verifies anchor ID bounding boxes and text retrieval against raw PDF.
4. `tests/test_pipeline_real_papers.py`: Validated on:
   - **Attention Is All You Need** (Vaswani et al., 11 pages, NeurIPS)
   - **LoRA: Low-Rank Adaptation** (Hu et al., 14 pages, ICLR)
   - **Deep Residual Learning (ResNet)** (He et al., 12 pages, CVPR)
   - **Spatial Stochastic Dynamics** (Single-column mathematical biology)

---

## 📄 License

MIT License.
