"""Generate single-column mathematical biology paper fixture for multimodal parser testing."""

import os
import fitz  # PyMuPDF


def create_single_column_bio_paper(out_path: str = "tests/data/single_column_bio.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = fitz.open()

    # Page 1: Single column Title, Abstract, Introduction
    p1 = doc.new_page(width=612, height=792)
    p1.insert_textbox(
        fitz.Rect(60, 50, 552, 90),
        "Spatial Stochastic Dynamics of Gene Regulatory Networks",
        fontsize=16,
        fontname="helv",
    )
    p1.insert_textbox(
        fitz.Rect(60, 100, 552, 130),
        "Elena Vance, Marcus Thorne, David S. Geller\nDepartment of Computational Biology, BioInstitute",
        fontsize=10,
        fontname="helv",
    )
    p1.insert_textbox(
        fitz.Rect(60, 150, 552, 230),
        "Abstract\nSpatial heterogeneity plays a crucial role in stochastic gene expression. In this paper, we formulate a reaction-diffusion master equation and derive rigorous analytical bounds for molecular noise propagation across cellular compartments.",
        fontsize=10,
        fontname="helv",
    )
    p1.insert_textbox(
        fitz.Rect(60, 260, 552, 400),
        "1 Introduction\nCellular decision making relies on precise spatial localization of transcription factors. Classical models assume well-mixed cytoplasmic conditions, ignoring diffusion delays and local crowding. Recent single-molecule fluorescence tracking experiments demonstrate substantial concentration gradients across micro-domains.",
        fontsize=10,
        fontname="helv",
    )
    p1.insert_textbox(
        fitz.Rect(60, 420, 552, 550),
        "2 Related Work\nEarly foundational work by Gillespie established the chemical master equation framework. More recently, spatial stochastic simulation algorithms have been developed for reaction-diffusion systems.",
        fontsize=10,
        fontname="helv",
    )

    # Page 2: Methodology & Equations & Figure
    p2 = doc.new_page(width=612, height=792)
    p2.insert_textbox(
        fitz.Rect(60, 50, 552, 100),
        "3 Methodology and Reaction-Diffusion Formulation\nWe discretize the spatial domain into K discrete subvolumes with diffusion hopping rate d.",
        fontsize=11,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(80, 110, 532, 150),
        "∂P(x, t)/∂t = ∑_{k=1}^K D ∇^2 P(x, t) + f(x) (1)",
        fontsize=10,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(60, 160, 552, 240),
        "3.1 Moment Closure Approximations\nUnder non-linear feedback loops, the moment hierarchy is unclosed. We apply a bivariate log-normal closure to approximate higher-order cumulants.",
        fontsize=10,
        fontname="helv",
    )
    # Draw a diagram box representing Figure 1
    p2.draw_rect(fitz.Rect(100, 260, 512, 420), color=(0.2, 0.4, 0.8), fill=(0.9, 0.95, 1.0), width=1.5)
    p2.insert_textbox(
        fitz.Rect(120, 320, 492, 360),
        "[Compartmental Diffusion Diagram: Subvolume v_k <-> v_{k+1}]",
        fontsize=10,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(60, 430, 552, 470),
        "Figure 1: Schematic illustration of the discretized spatial reaction-diffusion lattice.",
        fontsize=9,
        fontname="helv",
    )
    p2.insert_textbox(
        fitz.Rect(60, 490, 552, 600),
        "The steady-state covariance matrix satisfies the continuous Lyapunov equation A Σ + Σ A^T + Q = 0.",
        fontsize=10,
        fontname="helv",
    )

    # Page 3: Results, Table, Discussion, Conclusion, References
    p3 = doc.new_page(width=612, height=792)
    p3.insert_textbox(
        fitz.Rect(60, 50, 552, 100),
        "4 Results and Simulation Benchmarks\nWe evaluated our closed-form moment equations against 10,000 Monte Carlo Gillespie trajectories.",
        fontsize=11,
        fontname="helv",
    )
    # Draw Table
    p3.draw_rect(fitz.Rect(60, 110, 552, 220), color=(0.5, 0.5, 0.5), width=1)
    p3.insert_textbox(
        fitz.Rect(60, 115, 552, 200),
        "Table 1: Simulation error and runtime comparison across lattice resolutions.\nResolution | Gillespie (s) | Our Method (s) | Relative Error (%)\n10x10 | 142.5 | 0.12 | 0.84\n50x50 | 3820.1 | 0.85 | 1.12",
        fontsize=9,
        fontname="helv",
    )
    p3.insert_textbox(
        fitz.Rect(60, 230, 552, 320),
        "5 Discussion and Limitations\nWhile our analytical closure provides orders-of-magnitude speedups, strong bistability may cause unimodal assumptions to break down.",
        fontsize=10,
        fontname="helv",
    )
    p3.insert_textbox(
        fitz.Rect(60, 330, 552, 400),
        "6 Conclusion\nWe have introduced a fast, rigorous spatial stochastic modeling pipeline with verified accuracy across diverse biological regimes.",
        fontsize=10,
        fontname="helv",
    )
    p3.insert_textbox(
        fitz.Rect(60, 420, 552, 450),
        "References",
        fontsize=11,
        fontname="helv",
    )
    p3.insert_textbox(
        fitz.Rect(60, 460, 552, 650),
        "[1] D. T. Gillespie. Exact stochastic simulation of coupled chemical reactions. J. Phys. Chem., 81(25):2340-2361, 1977.\n[2] J. Elf and M. Ehrenberg. Fast evaluation of fluctuations in biochemical networks. Genome Res., 13(11):2475-2484, 2003.\n[3] S. Engblom et al. Simulation of stochastic reaction-diffusion processes on unstructured meshes. SIAM J. Sci. Comput., 31(3):1774-1797, 2009.\n[4] R. Grima. An effective rate equation approach to reaction kinetics in crowded environments. J. Chem. Phys., 132(18):185102, 2010.\n[5] M. Ullah and O. Wolkenhauer. Stochastic Approaches for Systems Biology. Springer, 2011.",
        fontsize=9,
        fontname="helv",
    )

    doc.save(out_path)
    doc.close()
    return out_path


if __name__ == "__main__":
    create_single_column_bio_paper()
    print("Created single column bio paper fixture.")
