"""MiniCert Streamlit research dashboard application entry point.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.comparison import (
    ComparisonItem,
    build_comparison_analysis,
    build_comparison_metric_table,
    get_comparison_series_for_plotting,
)
from dashboard.components import (
    HISTORY_TABLE_COLUMNS,
    PER_SAMPLE_COLUMNS,
    certification_rate_series,
    format_optional,
    history_table_rows,
    improvement_series,
    parse_epsilon_input,
    records_table_rows,
    runtime_series,
    status_display,
    width_series,
)
from dashboard.services import (
    IndividualOutcome,
    execute_and_save_experiment,
    predict_image,
    run_baseline_verification,
    run_refined_verification,
)
from dashboard.storage import (
    ExperimentMetadata,
    import_legacy_experiments,
    list_experiments,
    load_experiment_result,
)
from dashboard.visualizations import (
    plot_certification_rates,
    plot_comparison_certification,
    plot_comparison_improvement,
    plot_comparison_runtime,
    plot_comparison_widths,
    plot_improvement,
    plot_interval_widths,
    plot_runtime,
)
from evaluation import (
    ExperimentConfig,
    ExperimentResult,
    load_mnist_test_dataset,
    load_trained_network,
)
from evaluation.dataset import InputSample, flatten_image
from models.model_loader import MathematicalNetwork

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "saved_model.pth"
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(
    page_title="MiniCert · Neural Network Verification Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_research_theme() -> None:
    """Inject the unified research workspace design tokens, typography, and styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --mc-bg: #F7F3ED;
            --mc-sidebar: #F1EAE2;
            --mc-panel: #FFFCF8;
            --mc-ink: #30282A;
            --mc-body: #51494A;
            --mc-muted: #827875;
            --mc-line: #E5DCD3;
            --mc-line-light: #EFE9E1;
            --mc-wine: #702B3A;
            --mc-wine-hover: #56212D;
            --mc-cherry: #963E50;
            --mc-green: #1E7A34;
            --mc-green-bg: rgba(30, 122, 52, 0.08);
            --mc-amber: #9A6700;
            --mc-amber-bg: rgba(154, 103, 0, 0.08);
            --mc-red: #B3261E;
            --mc-red-bg: rgba(179, 38, 30, 0.08);
            --mc-shadow-sm: 0 1px 3px rgba(48, 40, 42, 0.05);
            --mc-shadow-md: 0 4px 12px rgba(48, 40, 42, 0.07);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .stApp {
            background-color: var(--mc-bg);
            color: var(--mc-body);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 243, 237, 0.94);
            backdrop-filter: blur(8px);
        }

        /* Typography */
        h1, h2, .mc-serif {
            font-family: 'DM Serif Display', Georgia, Cambria, 'Times New Roman', serif !important;
            color: var(--mc-ink) !important;
            letter-spacing: -0.025em;
            font-weight: 400;
        }
        h1 {
            font-size: 2.45rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.35rem !important;
        }
        h2 {
            font-size: 1.75rem !important;
            line-height: 1.25 !important;
            margin-top: 1.8rem !important;
            margin-bottom: 0.6rem !important;
        }
        h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: var(--mc-ink) !important;
            font-weight: 600 !important;
            letter-spacing: -0.015em;
        }
        p, li, label, span {
            color: var(--mc-body);
            line-height: 1.62;
        }

        /* Sidebar Research Drawer */
        [data-testid="stSidebar"] {
            background-color: var(--mc-sidebar) !important;
            border-right: 1px solid var(--mc-line) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div {
            gap: 4px;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background: transparent;
            border-radius: 8px;
            padding: 9px 14px;
            margin: 2px 0;
            transition: all 0.16s ease;
            cursor: pointer;
            border-left: 3px solid transparent;
            color: var(--mc-body);
            font-size: 0.92rem;
            font-weight: 500;
            user-select: none;
            -webkit-user-select: none;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label:focus,
        [data-testid="stSidebar"] div[data-testid="stRadio"] label:focus-visible,
        [data-testid="stSidebar"] div[data-testid="stRadio"] input:focus,
        [data-testid="stSidebar"] div[data-testid="stRadio"] input:focus-visible {
            outline: none !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background: #E8E0D6;
            color: var(--mc-ink);
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked),
        [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
            background: var(--mc-panel);
            color: var(--mc-wine) !important;
            font-weight: 600;
            border-left: 3px solid var(--mc-wine);
            box-shadow: var(--mc-shadow-sm);
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] label input[type="radio"] {
            accent-color: var(--mc-wine);
        }

        /* Hero Banner */
        .mc-hero {
            background: linear-gradient(135deg, #FFFCF8 0%, #F3EDE5 100%);
            border: 1px solid var(--mc-line);
            border-left: 5px solid var(--mc-wine);
            border-radius: 16px;
            padding: 32px 36px;
            margin: 6px 0 24px;
            box-shadow: var(--mc-shadow-sm);
        }
        .mc-kicker {
            color: var(--mc-wine);
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .mc-research-title {
            color: var(--mc-ink);
            font-family: 'DM Serif Display', Georgia, serif;
            font-size: 1.42rem;
            line-height: 1.4;
            max-width: 950px;
            margin: 4px 0 14px;
        }
        .mc-hero p {
            max-width: 920px;
            margin: 0;
            font-size: 0.98rem;
            line-height: 1.65;
            color: var(--mc-body);
        }

        /* Research Section Header */
        .mc-section-header {
            margin-bottom: 22px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--mc-line-light);
        }
        .mc-section-header h1 {
            margin: 4px 0 6px !important;
        }
        .mc-section-header p {
            font-size: 1.02rem;
            color: var(--mc-muted);
            margin: 0;
            max-width: 860px;
        }

        /* Analytical Cards */
        .mc-card {
            background: var(--mc-panel);
            border: 1px solid var(--mc-line);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: var(--mc-shadow-sm);
            transition: all 0.18s ease;
        }
        .mc-card:hover {
            box-shadow: var(--mc-shadow-md);
        }
        .mc-card-kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--mc-wine);
            margin-bottom: 8px;
        }
        .mc-card h4 {
            margin: 0 0 8px;
            color: var(--mc-ink);
        }

        /* Metric styling */
        [data-testid="stMetric"] {
            background: var(--mc-panel) !important;
            border: 1px solid var(--mc-line) !important;
            border-radius: 12px !important;
            padding: 16px 18px !important;
            box-shadow: var(--mc-shadow-sm) !important;
        }
        [data-testid="stMetricLabel"] {
            color: var(--mc-muted) !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--mc-wine) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-variant-numeric: tabular-nums !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
        }

        /* Interactive controls & buttons */
        div.stButton > button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            border: 1px solid var(--mc-line) !important;
            background: var(--mc-panel) !important;
            color: var(--mc-ink) !important;
            transition: all 0.16s ease !important;
            padding: 7px 18px !important;
        }
        div.stButton > button:hover {
            border-color: var(--mc-wine) !important;
            color: var(--mc-wine) !important;
            background: #FFF8F6 !important;
            transform: translateY(-1px);
        }
        div.stButton > button[data-testid="baseButton-primary"],
        div.stButton > button[kind="primary"],
        div.stButton > button[type="primary"] {
            background: var(--mc-wine) !important;
            border-color: var(--mc-wine) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        div.stButton > button[data-testid="baseButton-primary"]:hover,
        div.stButton > button[kind="primary"]:hover,
        div.stButton > button[type="primary"]:hover {
            background: var(--mc-wine-hover) !important;
            border-color: var(--mc-wine-hover) !important;
            color: #FFFFFF !important;
        }
        div.stDownloadButton > button {
            border-radius: 8px !important;
            border: 1px solid var(--mc-line) !important;
            background: var(--mc-panel) !important;
            color: var(--mc-ink) !important;
            font-size: 0.86rem !important;
            transition: all 0.16s ease !important;
        }
        div.stDownloadButton > button:hover {
            border-color: var(--mc-wine) !important;
            color: var(--mc-wine) !important;
            transform: translateY(-1px);
        }

        /* Sidebar Cards & Tags */
        .mc-sidebar-brand {
            padding: 12px 6px 18px;
            border-bottom: 1px solid var(--mc-line);
            margin-bottom: 16px;
        }
        .mc-sidebar-title {
            font-family: 'DM Serif Display', Georgia, serif;
            font-size: 1.68rem;
            color: var(--mc-ink);
            line-height: 1.2;
            margin: 2px 0 4px;
        }
        .mc-sidebar-sub {
            font-size: 0.8rem;
            color: var(--mc-muted);
            letter-spacing: 0.02em;
        }
        .mc-sidebar-card {
            background: var(--mc-panel);
            border: 1px solid var(--mc-line);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 14px;
            box-shadow: var(--mc-shadow-sm);
        }
        .mc-sidebar-card-label {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--mc-muted);
            margin-bottom: 4px;
        }
        .mc-sidebar-card-value {
            font-weight: 600;
            color: var(--mc-ink);
            font-size: 0.95rem;
        }
        .mc-sidebar-card-sub {
            font-size: 0.78rem;
            color: var(--mc-muted);
            margin-top: 2px;
        }
        .mc-sidebar-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 8px;
        }
        .mc-tag {
            font-size: 0.68rem;
            font-weight: 600;
            color: var(--mc-wine);
            background: #F4ECE7;
            border: 1px solid #E8DDD6;
            border-radius: 4px;
            padding: 2px 6px;
        }
        .mc-sidebar-footer {
            font-size: 0.74rem;
            color: var(--mc-muted);
            padding: 18px 4px 6px;
            border-top: 1px solid var(--mc-line);
            margin-top: 24px;
            line-height: 1.5;
        }

        /* Chart container */
        .mc-chart-box {
            background: var(--mc-panel);
            border: 1px solid var(--mc-line);
            border-radius: 14px;
            padding: 20px 22px 14px;
            margin: 10px 0 20px;
            box-shadow: var(--mc-shadow-sm);
        }
        .mc-chart-header {
            margin-bottom: 12px;
        }
        .mc-chart-title {
            font-size: 1.12rem;
            font-weight: 600;
            color: var(--mc-ink);
            margin-bottom: 4px;
        }
        .mc-chart-desc {
            font-size: 0.86rem;
            color: var(--mc-muted);
            margin: 0;
        }

        /* Step workflow grid */
        .mc-step-card {
            background: var(--mc-panel);
            border: 1px solid var(--mc-line);
            border-radius: 12px;
            padding: 18px 20px;
            height: 100%;
            box-shadow: var(--mc-shadow-sm);
            transition: all 0.16s ease;
        }
        .mc-step-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--mc-shadow-md);
            border-color: var(--mc-wine);
        }
        .mc-step-num {
            display: inline-block;
            color: var(--mc-wine);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }
        .mc-step-title {
            color: var(--mc-ink);
            font-size: 1.0rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .mc-step-desc {
            font-size: 0.86rem;
            color: var(--mc-body);
            line-height: 1.55;
            margin: 0;
        }

        /* Scope & Limitation Boxes */
        .mc-scope-box {
            background: var(--mc-panel);
            border: 1px solid var(--mc-line);
            border-top: 4px solid var(--mc-wine);
            border-radius: 12px;
            padding: 20px 24px;
            height: 100%;
            box-shadow: var(--mc-shadow-sm);
        }
        .mc-limits-box {
            background: var(--mc-panel);
            border: 1px solid var(--mc-line);
            border-top: 4px solid var(--mc-cherry);
            border-radius: 12px;
            padding: 20px 24px;
            height: 100%;
            box-shadow: var(--mc-shadow-sm);
        }

        /* Callout notes */
        .mc-callout {
            border-left: 4px solid var(--mc-wine);
            background: #F2ECE5;
            padding: 14px 18px;
            border-radius: 0 10px 10px 0;
            margin: 14px 0;
            font-size: 0.92rem;
            color: var(--mc-ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def cached_network(path: Path) -> MathematicalNetwork:
    """Load and cache the trained feedforward model abstraction."""
    return load_trained_network(path)


@st.cache_resource
def cached_dataset(data_dir: Path) -> Any:
    """Load and cache the canonical MNIST test dataset."""
    return load_mnist_test_dataset(data_dir)


def status_badge(status: str | None) -> str:
    """Render an accessible, high-contrast status pill badge."""
    label = status_display(status) if status else "NOT RUN"
    colors = {
        "CERTIFIED": ("#1E7A34", "rgba(30, 122, 52, 0.09)", "#1E7A34"),
        "INCONCLUSIVE": ("#9A6700", "rgba(154, 103, 0, 0.09)", "#9A6700"),
        "ERROR": ("#B3261E", "rgba(179, 38, 30, 0.09)", "#B3261E"),
        "COMPLETED": ("#1E7A34", "rgba(30, 122, 52, 0.09)", "#1E7A34"),
        "FAILED": ("#B3261E", "rgba(179, 38, 30, 0.09)", "#B3261E"),
        "INCOMPLETE": ("#9A6700", "rgba(154, 103, 0, 0.09)", "#9A6700"),
        "CORRUPTED": ("#B3261E", "rgba(179, 38, 30, 0.09)", "#B3261E"),
        "TIGHTENED": ("#1E7A34", "rgba(30, 122, 52, 0.09)", "#1E7A34"),
        "UNCHANGED": ("#51494A", "rgba(81, 73, 74, 0.08)", "#827875"),
    }
    text_color, bg_color, border_color = colors.get(
        label.upper(), ("#51494A", "rgba(81, 73, 74, 0.08)", "#827875")
    )
    return (
        f"<span style='color:{text_color}; background:{bg_color}; font-weight:600; "
        f"font-size:0.78rem; letter-spacing:0.05em; text-transform:uppercase; "
        f"border:1px solid {border_color}; border-radius:6px; padding:3px 10px; display:inline-block;'>"
        f"{label}</span>"
    )


def render_bounds_table(
    lower: np.ndarray,
    upper: np.ndarray,
    target_class: int | None = None,
    comp_lower: np.ndarray | None = None,
    comp_upper: np.ndarray | None = None,
) -> None:
    """Render a publication-quality logit bounds table with tabular numbers."""
    if lower.size == 0 or upper.size == 0:
        st.caption("Output bounds are not available.")
        return

    rows = []
    has_comp = (
        comp_lower is not None
        and comp_upper is not None
        and comp_lower.size == lower.size
        and comp_upper.size == upper.size
    )

    for idx in range(lower.size):
        is_target = target_class is not None and idx == target_class
        base_w = upper[idx] - lower[idx]
        row_dict = {
            "Class": f"{idx} {' ★ (Target)' if is_target else ''}",
            "Lower Bound": f"{lower[idx]:.4f}",
            "Upper Bound": f"{upper[idx]:.4f}",
            "Interval Width": f"{base_w:.4f}",
        }
        if has_comp:
            assert comp_lower is not None and comp_upper is not None
            ref_w = comp_upper[idx] - comp_lower[idx]
            reduction = (
                ((base_w - ref_w) / base_w * 100.0) if base_w > 0.0 else 0.0
            )
            row_dict["Refined Lower"] = f"{comp_lower[idx]:.4f}"
            row_dict["Refined Upper"] = f"{comp_upper[idx]:.4f}"
            row_dict["Refined Width"] = f"{ref_w:.4f}"
            row_dict["Width Tightening (%)"] = f"{reduction:+.1f}%"
        rows.append(row_dict)

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_chart_panel(
    figure: Any,
    data: Any,
    *,
    key: str,
    title: str,
    explanation: str,
    unit: str = "",
) -> None:
    """Render a publication chart with a visible control toolbar and accessible data table."""
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    st.markdown(
        f"""
        <div class="mc-chart-box">
            <div class="mc-chart-header">
                <div class="mc-chart-title">{title} {f'({unit})' if unit else ''}</div>
                <div class="mc-chart-desc">{explanation}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Control Toolbar above chart
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 2, 1])
    with ctrl_col1:
        show_legend = st.checkbox(
            "Show Legend", value=True, key=f"{key}_show_legend"
        )
    with ctrl_col2:
        show_grid = st.checkbox("Show Grid", value=True, key=f"{key}_show_grid")

    axis = figure.axes[0] if figure.axes else None
    if axis is not None:
        if axis.get_legend() is not None:
            axis.get_legend().set_visible(show_legend)
        axis.grid(
            show_grid, axis="y", color="#E5DCD3", linewidth=0.65, alpha=0.9
        )

    # Export PNG Buffer
    png_buffer = io.BytesIO()
    figure.savefig(
        png_buffer,
        format="png",
        dpi=200,
        bbox_inches="tight",
        facecolor="#FFFCF8",
    )

    with ctrl_col4:
        st.download_button(
            "Download PNG",
            data=png_buffer.getvalue(),
            file_name=f"{key}.png",
            mime="image/png",
            key=f"{key}_btn_png",
            width="stretch",
        )

    # Chart Display
    with st.container(border=True):
        st.pyplot(figure, width="stretch", clear_figure=False)

    # Accessible Collapsible Data Table
    with st.expander("Accessible Numerical Data & CSV Export", expanded=False):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button(
            "Download Chart Data (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{key}_data.csv",
            mime="text/csv",
            key=f"{key}_btn_csv",
        )


# ==============================================================================
# Section 1: Overview
# ==============================================================================


def render_overview_section(network: MathematicalNetwork) -> None:
    """Render the research foundation, methodology, architecture, and empirical evidence."""
    # Section A: Research Identity Hero
    st.markdown(
        """
        <section class="mc-hero">
            <div class="mc-kicker">NEURAL NETWORK ROBUSTNESS RESEARCH</div>
            <h1>MiniCert</h1>
            <div class="mc-research-title">
                Formal Verification of Neural Networks Using Interval Bound Propagation and LP-Based Refinement
            </div>
            <p>
                MiniCert is a neural network robustness verification framework that combines Interval Bound Propagation
                with linear programming-based bound refinement to determine whether a neural network's prediction
                remains stable under bounded input perturbations.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Section B: The Research Problem
    st.markdown("## The Research Problem")
    st.markdown(
        """
        Deep neural networks achieve high empirical accuracy on validation datasets, yet remain notoriously susceptible
        to adversarial perturbations—imperceptible input changes that cause catastrophic misclassifications.
        Standard test accuracy provides **no formal guarantee** that a prediction will remain constant across an entire
        continuous neighborhood around a test input.

        To guarantee safety, formal verification methods compute provable enclosures of reachable network outputs under
        bounded perturbations ($L_\\infty$-norm ball $\\mathcal{B}_\\epsilon(x_0)$).
        **Interval Bound Propagation (IBP)** propagates intervals layer-by-layer in $O(\\text{weights})$ time.
        While computationally efficient, IBP suffers from the *wrapping effect*, over-approximating reachable sets and
        producing loose bounds that fail to verify genuinely robust inputs.

        **MiniCert investigates whether linear programming (LP) relaxations of non-linear ReLU activations can tighten
        these output bounds, improving certification rates while maintaining computational tractability.**
        """
    )

    # Section C: Proposed Methodology
    st.markdown("## Proposed Verification Methodology")
    st.markdown(
        "MiniCert implements a principled two-stage verification pipeline combining fast interval propagation with targeted convex relaxations:"
    )

    steps = [
        (
            "01 · INPUT REGION",
            "Bounded Norm Ball",
            "Construct the L-infinity perturbation region [x₀ - ε, x₀ + ε] clipped to valid MNIST pixel bounds [0, 1].",
        ),
        (
            "02 · ABSTRACTION",
            "Mathematical Network",
            "Abstract sequential feedforward layers into explicit Linear affine transforms and piecewise-linear ReLUs.",
        ),
        (
            "03 · BASELINE IBP",
            "Interval Propagation",
            "Compute coarse intermediate and output bounds [L, U] using forward interval arithmetic.",
        ),
        (
            "04 · CONVEX RELAXATION",
            "Planet Triangle LP",
            "Formulate linear programs for unstable ReLUs (l < 0 < u) using convex hull constraints and solve with HiGHS.",
        ),
        (
            "05 · LOGIT REFINEMENT",
            "Output Bound Tightening",
            "Solve targeted LPs to compute refined lower and upper bounds directly for the final classification logits.",
        ),
        (
            "06 · VERDICT",
            "Provable Certificate",
            "Evaluate whether the target class provably dominates all competing classes throughout the perturbation space.",
        ),
    ]

    for row_start in (0, 3):
        cols = st.columns(3)
        for col, (kicker, title, desc) in zip(
            cols, steps[row_start : row_start + 3], strict=True
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="mc-step-card">
                        <span class="mc-step-num">{kicker}</span>
                        <div class="mc-step-title">{title}</div>
                        <p class="mc-step-desc">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Formal Certification Condition")
    st.markdown(
        "A classification decision for target class $t$ is provably **CERTIFIED** robust across the entire perturbation region if and only if:"
    )
    st.latex(r"L_t > \max_{j \ne t} U_j")
    st.markdown(
        r"""
        Where:
        - $L_t$ is the certified lower bound of the target (predicted) logit.
        - $U_j$ is the certified upper bound of competing class logit $j \ne t$.
        - The **Verification Margin** is defined as $M = L_t - \max_{j \ne t} U_j$. If $M > 0$, the target class dominates all competitors for all inputs in the region.
        """
    )
    st.markdown(
        """
        <div class="mc-callout">
            <strong>Important Scientific Clarification:</strong> An <strong>INCONCLUSIVE</strong> verdict means the sufficient
            certification condition could not be proven using the computed bounds. It does <em>not</em> imply that an adversarial
            counterexample exists, nor does it prove that the network is unsafe. It simply indicates that the relaxation was too loose to guarantee robustness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section D: Technical Architecture
    st.markdown("## Implemented Technical Architecture")
    st.markdown(
        "MiniCert is engineered with clean mathematical separation between the neural model abstraction, the abstract domain, and the numerical solver backend:"
    )

    arch_cols = st.columns(4)
    arch_modules = [
        (
            "MODEL INGESTION",
            "MathematicalNetwork",
            "PyTorch checkpoint loader extracting weights and biases into sequential Linear and ReLU mathematical operators.",
            "models/model_loader.py",
        ),
        (
            "ABSTRACT DOMAIN",
            "Interval Arithmetic",
            "IEEE float64 Box domain propagation computing intermediate pre- and post-activation neuron enclosures.",
            "abstract_domain/bounds.py",
        ),
        (
            "LP FORMULATION",
            "Planet ReLU Relaxation",
            "Symbolic LP builder constructing immutable CSC sparse constraint matrices with reusable bounds and dual checks.",
            "verification/lp.py",
        ),
        (
            "SOLVER BACKEND",
            "SciPy HiGHS Optimizer",
            "High-performance C++ simplex solver with vectorized numerical feasibility, duality-gap validation, and L4 refinement.",
            "verification/lp_solver.py",
        ),
    ]
    for col, (mod_kicker, mod_title, mod_desc, mod_file) in zip(
        arch_cols, arch_modules, strict=True
    ):
        with col:
            st.markdown(
                f"""
                <div class="mc-card">
                    <div class="mc-card-kicker">{mod_kicker}</div>
                    <h4>{mod_title}</h4>
                    <p style="font-size:0.86rem; margin-bottom:8px;">{mod_desc}</p>
                    <code style="font-size:0.75rem; color:var(--mc-wine);">{mod_file}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Section E: Scope & Limitations
    st.markdown("## Research Scope & Boundary Limitations")
    scope_col, limit_col = st.columns(2)
    with scope_col:
        st.markdown(
            """
            <div class="mc-scope-box">
                <div class="mc-card-kicker">CURRENTLY SUPPORTED</div>
                <h4>Verified Capabilities</h4>
                <ul style="font-size:0.9rem; padding-left:18px; margin:0;">
                    <li><strong>Architecture:</strong> Fully-connected sequential feedforward networks (MLP).</li>
                    <li><strong>Layer Types:</strong> Linear affine maps and piecewise-linear ReLU activations.</li>
                    <li><strong>Data Domain:</strong> Normalized image feature tensors (e.g. MNIST 28×28 → 784).</li>
                    <li><strong>Perturbation Norm:</strong> Bounded L-infinity ball (||x - x₀||∞ ≤ ε).</li>
                    <li><strong>Numerical Engine:</strong> IEEE 754 float64 arithmetic with strict residual tolerances (10⁻⁶).</li>
                    <li><strong>Refinement:</strong> SciPy HiGHS LP dual-simplex optimizer with Layer-4 logit specialization.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with limit_col:
        st.markdown(
            """
            <div class="mc-limits-box">
                <div class="mc-card-kicker">SOUNDNESS & ARCHITECTURAL LIMITATIONS</div>
                <h4>Out-of-Scope Boundaries</h4>
                <ul style="font-size:0.9rem; padding-left:18px; margin:0;">
                    <li><strong>No CNN / Attention:</strong> Convolutional, pooling, and Transformer layers are not supported.</li>
                    <li><strong>No Non-Linear Activations:</strong> Sigmoid, Tanh, GELU, and SiLU activations are unsupported.</li>
                    <li><strong>No Exact Rationals:</strong> Soundness relies on float64 solver tolerances, not infinite-precision arithmetic.</li>
                    <li><strong>Incompleteness:</strong> As a convex relaxation, failure to certify does not generate adversarial examples.</li>
                    <li><strong>Scalability Horizon:</strong> LP solving scales with active neurons; very large networks require MILP/SMT.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section F: Experimental Evidence (Live Stored Artifacts)
    st.markdown("## Empirical Experimental Evidence")
    st.markdown(
        "MiniCert has been evaluated through reproducible large-sample benchmarks and systematic optimization audits. "
        "The figures below reflect real artifacts persisted in `results/`:"
    )

    evidence_path = (
        PROJECT_ROOT / "results" / "equivalence_seed39_n100_summary.json"
    )
    if evidence_path.is_file():
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            overall = evidence["overall"]
            cfg = evidence["config"]
            solves_ratio = (
                overall["mode_a_total_lp_solves"]
                / overall["mode_b_total_lp_solves"]
            )

            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Evaluated Samples",
                evidence["sample_selection"]["selected"],
                f"Seed {cfg['seed']}",
            )
            metric_cols[1].metric(
                "Total Paired Queries",
                overall["completed_pairs"],
                f"{len(cfg['epsilons'])} Perturbation Radii",
            )
            metric_cols[2].metric(
                "Verdict Discrepancies",
                f"{overall['verdict_mismatches']} / 400",
                "Exact Equivalence",
            )
            metric_cols[3].metric(
                "Mean Runtime Speedup",
                f"{overall['mean_runtime_speedup']:.2f}×",
                f"Median {overall['median_runtime_speedup']:.2f}×",
            )

            st.markdown(
                f"""
                <div class="mc-card" style="margin-top:12px;">
                    <div class="mc-card-kicker">100-SAMPLE OPTIMIZATION AUDIT SUMMARY (SEED 39)</div>
                    <p style="margin:0; font-size:0.92rem;">
                        Rigorous comparison between <strong>all-target refinement</strong> (Mode A: hidden + output neurons)
                        and <strong>optimized Layer 4-only refinement</strong> (Mode B: output logits only) across 400 comparisons:
                        <br>&bull; <strong>LP Solves Reduced:</strong> 44,686 solves down to 8,000 solves (<strong>{solves_ratio:.2f}× fewer LP solves</strong>, up to 8.74× fewer at ε = 0.10).
                        <br>&bull; <strong>Bound Identity:</strong> Maximum lower bound diff: <code>0.000e+00</code>, maximum upper bound diff: <code>0.000e+00</code>.
                        <br>&bull; <strong>Verdict Discrepancies:</strong> 0 mismatches across all 400 comparisons.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Note on comparison scope: The speedup above compares all-target refinement against optimized Layer 4-only refinement. "
                "In contrast, certification rate improvements (e.g., 5.0% → 90.0% at ε = 0.01) reflect the benefit of LP refinement over baseline IBP."
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not parse equivalence audit evidence: {exc}")
    else:
        st.info(
            "The 100-sample equivalence validation artifact (`equivalence_seed39_n100_summary.json`) is not present."
        )


# ==============================================================================
# Section 2: Individual Verification
# ==============================================================================


def dataset_sample(
    index: int, vector: np.ndarray, label: int, predicted: int
) -> Any:
    return InputSample(
        index=index, image=vector, label=label, predicted=predicted
    )


def render_individual_section(
    network: MathematicalNetwork, dataset: Any
) -> None:
    """Render the single-sample verification inspection interface."""
    st.markdown(
        """
        <div class="mc-section-header">
            <div class="mc-kicker">SAMPLE-LEVEL VERIFICATION INSPECTION</div>
            <h1>Individual Sample Verification</h1>
            <p>
                Inspect an individual MNIST test digit, configure its perturbation radius ε, and compare baseline IBP
                bounds with LP-refined bounds across all ten output logits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Select Sample & Perturbation Radius")
    input_c1, input_c2 = st.columns([1, 1])
    with input_c1:
        sample_index = int(
            st.number_input(
                "MNIST Test Sample Index",
                min_value=0,
                max_value=len(dataset) - 1,
                value=0,
                step=1,
            )
        )
    with input_c2:
        epsilon = float(
            st.slider(
                "Perturbation Radius ε (L-infinity norm)",
                min_value=0.0,
                max_value=0.20,
                value=0.03,
                step=0.005,
                format="%.3f",
            )
        )

    # Extract sample and prediction
    image_tensor, label = dataset[sample_index]
    vector = flatten_image(image_tensor)

    pred_key = ("pred", sample_index)
    if st.session_state.get(pred_key) is None:
        logits, predicted = predict_image(network, vector)
        st.session_state[pred_key] = (predicted, logits)
    predicted, logits = st.session_state[pred_key]

    # Sample info card
    preview_col1, preview_col2 = st.columns([1, 3])
    with preview_col1:
        st.image(
            vector.reshape(28, 28),
            caption=f"MNIST Test Digit #{sample_index}",
            width=150,
            clamp=True,
        )
    with preview_col2:
        st.markdown(
            f"""
            <div class="mc-card">
                <div class="mc-card-kicker">CLASSIFICATION SUMMARY</div>
                <h4>Ground Truth Label: <strong>{label}</strong> &nbsp;|&nbsp; Model Prediction: <strong>{predicted}</strong></h4>
                <p style="margin:0; font-size:0.9rem;">
                    Target class for verification: <strong>Class {predicted}</strong>.
                    <br>Perturbation space: <code>[{vector.min():.2f} - {epsilon:g}, {vector.max():.2f} + {epsilon:g}] ∩ [0, 1]⁷⁸⁴</code>.
                    <br>The network prediction is certified robust if the lower bound of class {predicted} exceeds all competing upper bounds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 2. Execute Verification")
    btn_col1, btn_col2 = st.columns(2)
    cache_key = (sample_index, epsilon)
    with btn_col1:
        run_baseline = st.button(
            "Run Baseline IBP Verification", width="stretch"
        )
    with btn_col2:
        run_refined = st.button(
            "Run LP-Refined Verification", type="primary", width="stretch"
        )

    if run_baseline:
        with st.spinner("Computing Interval Bound Propagation..."):
            st.session_state.setdefault("baseline_cache", {})[cache_key] = (
                run_baseline_verification(
                    network,
                    dataset_sample(sample_index, vector, label, predicted),
                    epsilon,
                )
            )

    if run_refined:
        with st.spinner("Solving LP relaxations via SciPy HiGHS..."):
            st.session_state.setdefault("refined_cache", {})[cache_key] = (
                run_refined_verification(
                    network,
                    dataset_sample(sample_index, vector, label, predicted),
                    epsilon,
                )
            )

    baseline_outcome: IndividualOutcome | None = st.session_state.get(
        "baseline_cache", {}
    ).get(cache_key)
    refined_outcome: IndividualOutcome | None = st.session_state.get(
        "refined_cache", {}
    ).get(cache_key)

    st.markdown("### 3. Verification Outcomes & Bound Enclosures")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("#### Baseline Interval Bound Propagation")
        if baseline_outcome is None and refined_outcome is None:
            st.info("Click 'Run Baseline IBP' above to evaluate interval bounds.")
        else:
            base = (
                baseline_outcome
                if baseline_outcome is not None
                else refined_outcome
            )
            assert base is not None
            st.markdown(
                f"**Verdict:** {status_badge(base.baseline_status)}",
                unsafe_allow_html=True,
            )
            b_m1, b_m2, b_m3 = st.columns(3)
            b_m1.metric(
                "Margin",
                format_optional(base.baseline_margin),
                help="Lt - max(Uj for j != t). Must be > 0 to certify.",
            )
            b_m2.metric("Target", base.target_class)
            b_m3.metric(
                "IBP Time",
                f"{base.baseline_time_seconds:.4f}s"
                if base.baseline_time_seconds
                else "—",
            )
            if base.baseline_status == "ERROR" and base.error:
                st.error(f"IBP execution error: {base.error}")

    with res_col2:
        st.markdown("#### LP-Refined Verification (IBP + HiGHS)")
        if refined_outcome is None:
            st.info(
                "Click 'Run LP-Refined Verification' to solve linear programs."
            )
        else:
            st.markdown(
                f"**Verdict:** {status_badge(refined_outcome.refined_status)}",
                unsafe_allow_html=True,
            )
            r_m1, r_m2, r_m3, r_m4 = st.columns(4)
            r_m1.metric("Margin", format_optional(refined_outcome.refined_margin))
            r_m2.metric(
                "LP Time",
                f"{refined_outcome.lp_refinement_time_seconds:.3f}s"
                if refined_outcome.lp_refinement_time_seconds
                else "—",
            )
            r_m3.metric(
                "Total Time",
                f"{refined_outcome.refined_time_seconds:.3f}s"
                if refined_outcome.refined_time_seconds
                else "—",
            )
            r_m4.metric(
                "LP Solves",
                f"{refined_outcome.refinement_accepted or 0} / {refined_outcome.refinement_targets or 0}",
            )
            if (
                refined_outcome.refined_status == "ERROR"
                and refined_outcome.error
            ):
                st.error(f"Refinement error: {refined_outcome.error}")

    st.markdown("#### Output Logit Bounds (Class-by-Class)")
    if (
        refined_outcome is not None
        and baseline_outcome is not None
        and refined_outcome.refined_lower.size > 0
        and baseline_outcome.baseline_lower.size > 0
    ):
        render_bounds_table(
            lower=baseline_outcome.baseline_lower,
            upper=baseline_outcome.baseline_upper,
            target_class=predicted,
            comp_lower=refined_outcome.refined_lower,
            comp_upper=refined_outcome.refined_upper,
        )
    elif refined_outcome is not None and refined_outcome.refined_lower.size > 0:
        render_bounds_table(
            lower=refined_outcome.refined_lower,
            upper=refined_outcome.refined_upper,
            target_class=predicted,
        )
    elif (
        baseline_outcome is not None and baseline_outcome.baseline_lower.size > 0
    ):
        render_bounds_table(
            lower=baseline_outcome.baseline_lower,
            upper=baseline_outcome.baseline_upper,
            target_class=predicted,
        )


# ==============================================================================
# Section 3: New Experiment
# ==============================================================================


def render_new_experiment_section(
    network: MathematicalNetwork, dataset: Any
) -> None:
    """Render the experiment runner and configuration workflow."""
    st.markdown(
        """
        <div class="mc-section-header">
            <div class="mc-kicker">EVALUATION BENCHMARK RUNNER</div>
            <h1>Configure New Verification Experiment</h1>
            <p>
                Configure, execute, and persist systematic verification benchmarks comparing baseline IBP against
                LP-refined verification across multiple perturbation radii.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Benchmark Parameters")
    st.caption(
        "Only correctly classified test samples are evaluated. All runs use deterministic seeds and streaming JSONL checkpointing."
    )

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        sample_count = int(
            st.number_input(
                "Sample Count (correctly classified)",
                min_value=1,
                max_value=10000,
                value=20,
                step=5,
                help="Number of correctly classified MNIST test digits to sample.",
            )
        )
    with param_col2:
        epsilon_text = st.text_input(
            "Perturbation Radii ε (comma-separated)",
            value="0.01, 0.03, 0.05, 0.1",
            help="List of perturbation radii to benchmark (e.g. 0.01, 0.03, 0.05, 0.1).",
        )
    with param_col3:
        seed = int(
            st.number_input(
                "Random Seed",
                min_value=0,
                max_value=2**31 - 1,
                value=42,
                step=1,
                help="Seed for reproducible sample selection.",
            )
        )

    st.markdown("### 2. Execution Design")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        run_dev = st.button(
            "Run Fast Development Run (5 samples, ε = 0.03)", width="stretch"
        )
    with action_col2:
        run_conf = st.button(
            "Run Configured Experiment", type="primary", width="stretch"
        )

    if run_dev:
        cfg = ExperimentConfig(sample_count=5, epsilons=(0.03,), seed=42)
        _execute_and_register_experiment(
            cfg, network, dataset, experiment_type="development"
        )
    elif run_conf:
        try:
            epsilons = parse_epsilon_input(epsilon_text)
            cfg = ExperimentConfig(
                sample_count=sample_count,
                epsilons=epsilons,
                seed=seed,
                data_dir=DATA_DIR,
                checkpoint=CHECKPOINT_PATH,
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Configuration error: {exc}")
        else:
            total_pairs = sample_count * len(epsilons)
            if total_pairs >= 200:
                st.warning(
                    f"Configuration will verify {total_pairs} sample-epsilon pairs ({total_pairs * 20:,} LP solves). "
                    "This run may take several minutes."
                )
            _execute_and_register_experiment(
                cfg, network, dataset, experiment_type="configured"
            )


def _execute_and_register_experiment(
    config: ExperimentConfig,
    network: MathematicalNetwork,
    dataset: Any,
    experiment_type: str,
) -> None:
    """Run an evaluation experiment with progress indicators and register the result."""
    with st.spinner(
        f"Executing {experiment_type} verification experiment (this may take some time)..."
    ):
        try:
            result, meta = execute_and_save_experiment(
                config,
                network,
                dataset,
                experiment_type=experiment_type,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Experiment execution failed: {exc}")
            return

    st.session_state["experiment_result"] = result
    st.session_state["current_experiment_id"] = meta.experiment_id
    st.session_state["current_experiment_metadata"] = meta

    st.success(
        f"Experiment completed successfully and persisted as `{meta.experiment_id}`! "
        "Navigate to **Experiment Results** to inspect charts and metrics."
    )


# ==============================================================================
# Section 4: Experiment Results
# ==============================================================================


def render_results_section() -> None:
    """Render the active experiment's metrics, comparative charts, and per-sample tables."""
    st.markdown(
        """
        <div class="mc-section-header">
            <div class="mc-kicker">ACTIVE RUN ANALYTICS</div>
            <h1>Experiment Results</h1>
            <p>
                In-depth analytical evaluation of certification rates, interval width tightening, solver runtime,
                and per-sample verification records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result: ExperimentResult | None = st.session_state.get("experiment_result")
    exp_id = st.session_state.get("current_experiment_id")

    if result is None:
        st.info(
            "No experiment loaded or executed yet. Run an experiment in 'New Experiment' "
            "or load an existing one from 'Experiment History'."
        )
        return

    sel = result.selection
    metadata = st.session_state.get("current_experiment_metadata")
    exp_type = (
        metadata.experiment_type.capitalize()
        if metadata and getattr(metadata, "experiment_type", None)
        else "Configured"
    )
    st.markdown(
        f"""
        <div class="mc-card">
            <div class="mc-card-kicker">ACTIVE EXPERIMENT IDENTIFIER</div>
            <h4><code>{exp_id or result.config.experiment_name}</code> ({exp_type})</h4>
            <p style="margin:0; font-size:0.9rem;">
                Evaluated <strong>{sel.selected_count} correctly classified samples</strong> (excluded {sel.incorrect_excluded} misclassified).
                Tested across <strong>{len(result.metrics)} perturbation radii</strong> with random seed <code>{result.config.seed}</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Aggregated Summary by Perturbation Radius")
    for entry in result.metrics:
        cert = entry.certification
        st.markdown(f"**Perturbation Radius ε = {entry.epsilon:g}**")
        m_cols = st.columns(4)
        m_cols[0].metric(
            "Baseline Certified",
            f"{cert.baseline_rate * 100:.1f}%"
            if cert.baseline_rate is not None
            else "—",
        )
        m_cols[1].metric(
            "Refined Certified",
            f"{cert.refined_rate * 100:.1f}%"
            if cert.refined_rate is not None
            else "—",
        )
        m_cols[2].metric(
            "Improvement Delta",
            f"{cert.improvement_pp:+.1f} pp"
            if cert.improvement_pp is not None
            else "—",
        )
        m_cols[3].metric(
            "Verification Errors",
            cert.baseline_errors + cert.refined_errors,
        )

    st.markdown("### Publication Visualizations")
    tab_c1, tab_c2, tab_c3, tab_c4 = st.tabs([
        "Certification Rates",
        "Interval Widths",
        "Runtime Comparison",
        "Certification Improvement",
    ])

    with tab_c1:
        cert_data = certification_rate_series(result.metrics)
        render_chart_panel(
            plot_certification_rates(cert_data),
            cert_data,
            key="results_certification",
            title="Certification Rate Comparison",
            explanation="Percentage of evaluated samples certified robust at each perturbation radius. Missing observations remain unavailable rather than zero.",
            unit="%",
        )

    with tab_c2:
        width_data = width_series(result.metrics)
        render_chart_panel(
            plot_interval_widths(width_data),
            width_data,
            key="results_width",
            title="Output Interval Width Comparison",
            explanation="Mean output logit interval width (upper - lower bound). Narrower widths reflect tighter enclosures.",
            unit="Logit units",
        )

    with tab_c3:
        runtime_data = runtime_series(result.metrics)
        render_chart_panel(
            plot_runtime(runtime_data),
            runtime_data,
            key="results_runtime",
            title="Mean Verification Runtime Comparison",
            explanation="Mean verification runtime per sample in seconds. Uses logarithmic scaling when runtimes span multiple orders of magnitude.",
            unit="Seconds",
        )

    with tab_c4:
        improvement_data = improvement_series(result.metrics)
        render_chart_panel(
            plot_improvement(improvement_data),
            improvement_data,
            key="results_improvement",
            title="Certification Rate Improvement",
            explanation="Refined certification percentage minus baseline certification percentage (in percentage points). Values > 0 favor LP refinement.",
            unit="Percentage Points (pp)",
        )

    st.markdown("### Per-Sample Verification Records")
    rows = records_table_rows(result)
    st.dataframe(
        pd.DataFrame(rows, columns=PER_SAMPLE_COLUMNS),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Download Artifacts")
    try:
        export_paths = result.save()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to export files: {exc}")
        return

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        samples_path = export_paths["samples_csv"]
        st.download_button(
            "Download Per-Sample Records (CSV)",
            data=samples_path.read_bytes(),
            file_name=samples_path.name,
            mime="text/csv",
            width="stretch",
        )
    with exp_col2:
        summary_path = export_paths["summary_json"]
        st.download_button(
            "Download Aggregated Summary (JSON)",
            data=summary_path.read_bytes(),
            file_name=summary_path.name,
            mime="application/json",
            width="stretch",
        )


# ==============================================================================
# Section 5: Experiment History
# ==============================================================================


def render_history_section() -> None:
    """Render the persistent benchmark repository and historical inspection drawer."""
    hist_hdr, hist_btn = st.columns([3, 1])
    with hist_hdr:
        st.markdown(
            """
            <div class="mc-section-header">
                <div class="mc-kicker">BENCHMARK REPOSITORY</div>
                <h1>Experiment History</h1>
                <p>
                    Browse, inspect, and export previously completed evaluation runs. Historical results are loaded
                    directly from persistent disk storage without re-evaluating verification queries.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hist_btn:
        st.write("")
        if st.button(
            "🔄 Refresh History",
            width="stretch",
            help="Scan storage and discover newly completed runs",
        ):
            newly_imported = import_legacy_experiments()
            if newly_imported:
                st.success(
                    f"Discovered and imported {len(newly_imported)} experiment(s)!"
                )
            else:
                st.info("Experiment repository is up to date.")
            st.rerun()

    all_experiments = list_experiments()
    if not all_experiments:
        st.info("No experiment records found in persistent storage.")
        return

    # Summary table
    rows = history_table_rows(all_experiments)
    st.dataframe(
        pd.DataFrame(rows, columns=HISTORY_TABLE_COLUMNS),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Inspect Stored Experiment")
    options = [m.experiment_id for m in all_experiments]
    selected_id = st.selectbox(
        "Choose an experiment to inspect details and charts:",
        options=options,
        index=0,
        format_func=lambda exp_id: f"{exp_id} ({next((getattr(m, 'experiment_type', 'run').capitalize() for m in all_experiments if m.experiment_id == exp_id), '')})",
    )

    selected_meta = next(
        (m for m in all_experiments if m.experiment_id == selected_id), None
    )
    if selected_meta is None:
        return

    st.markdown(
        f"**Experiment Status:** {status_badge(selected_meta.status)}",
        unsafe_allow_html=True,
    )
    if selected_meta.error_message:
        st.warning(f"Error note: {selected_meta.error_message}")

    if selected_meta.status not in ("Completed", "Failed"):
        st.info("This experiment is incomplete or corrupted and cannot be fully loaded.")
        return

    try:
        loaded_res = load_experiment_result(selected_id)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load experiment result: {exc}")
        return

    load_col1, _ = st.columns([1, 3])
    with load_col1:
        if st.button("Load into Active Results View", width="stretch"):
            st.session_state["experiment_result"] = loaded_res
            st.session_state["current_experiment_id"] = selected_id
            st.session_state["current_experiment_metadata"] = selected_meta
            st.session_state["workspace_nav"] = "Experiment Results"
            st.success(f"Experiment `{selected_id}` is now active in Results!")
            st.rerun()

    with st.expander("Stored Configuration JSON", expanded=False):
        st.json(loaded_res.config.as_config_dict())

    st.markdown("#### Aggregated Metrics by Epsilon")
    for entry in loaded_res.metrics:
        cert = entry.certification
        cols = st.columns(4)
        cols[0].metric(
            f"Baseline Rate (ε={entry.epsilon:g})",
            f"{cert.baseline_rate * 100:.1f}%"
            if cert.baseline_rate is not None
            else "—",
        )
        cols[1].metric(
            f"Refined Rate (ε={entry.epsilon:g})",
            f"{cert.refined_rate * 100:.1f}%"
            if cert.refined_rate is not None
            else "—",
        )
        cols[2].metric(
            f"Improvement (ε={entry.epsilon:g})",
            f"{cert.improvement_pp:+.1f} pp"
            if cert.improvement_pp is not None
            else "—",
        )
        cols[3].metric(
            f"Inconclusive (ε={entry.epsilon:g})",
            f"{cert.baseline_inconclusive} / {cert.refined_inconclusive}",
        )

    with st.expander("Historical Per-Sample Records Table", expanded=False):
        sample_rows = records_table_rows(loaded_res)
        st.dataframe(
            pd.DataFrame(sample_rows, columns=PER_SAMPLE_COLUMNS),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Historical Visualizations")
    h_tab1, h_tab2, h_tab3, h_tab4 = st.tabs([
        "Certification Rates",
        "Interval Widths",
        "Runtime Comparison",
        "Certification Improvement",
    ])
    with h_tab1:
        c_data = certification_rate_series(loaded_res.metrics)
        render_chart_panel(
            plot_certification_rates(c_data),
            c_data,
            key="hist_cert",
            title="Certification Rate Comparison",
            explanation="Certification rates across perturbation radii.",
            unit="%",
        )
    with h_tab2:
        w_data = width_series(loaded_res.metrics)
        render_chart_panel(
            plot_interval_widths(w_data),
            w_data,
            key="hist_width",
            title="Output Interval Width Comparison",
            explanation="Mean output logit interval widths.",
            unit="Logit units",
        )
    with h_tab3:
        r_data = runtime_series(loaded_res.metrics)
        render_chart_panel(
            plot_runtime(r_data),
            r_data,
            key="hist_runtime",
            title="Mean Runtime Comparison",
            explanation="Mean verification runtime in seconds per sample.",
            unit="Seconds",
        )
    with h_tab4:
        i_data = improvement_series(loaded_res.metrics)
        render_chart_panel(
            plot_improvement(i_data),
            i_data,
            key="hist_imp",
            title="Certification Improvement",
            explanation="Refined minus baseline certification percentage.",
            unit="pp",
        )

    st.markdown("#### Download Historical Files")
    exp_dir = selected_meta.path
    s_file = exp_dir / "samples.csv"
    sum_file = exp_dir / "summary.json"
    dl_c1, dl_c2 = st.columns(2)
    if s_file.is_file():
        with dl_c1:
            st.download_button(
                "Download samples.csv",
                data=s_file.read_bytes(),
                file_name=f"{selected_id}_samples.csv",
                mime="text/csv",
                width="stretch",
            )
    if sum_file.is_file():
        with dl_c2:
            st.download_button(
                "Download summary.json",
                data=sum_file.read_bytes(),
                file_name=f"{selected_id}_summary.json",
                mime="application/json",
                width="stretch",
            )


# ==============================================================================
# Section 6: Experiment Comparison
# ==============================================================================


def render_comparison_section() -> None:
    """Render cross-experiment side-by-side comparability analysis, tables, and charts."""
    st.markdown(
        """
        <div class="mc-section-header">
            <div class="mc-kicker">CROSS-EXPERIMENT ANALYSIS</div>
            <h1>Multi-Experiment Comparison</h1>
            <p>
                Compare two or more completed evaluation runs across matching perturbation radii. Rigorously inspects
                sample-set comparability, certification improvements, bound widths, and execution runtimes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_experiments = list_experiments()
    completed_experiments = [
        m for m in all_experiments if m.status == "Completed"
    ]

    if len(completed_experiments) < 2:
        st.info(
            "At least two completed experiments are required for cross-experiment comparison. "
            "Execute additional benchmarks in 'New Experiment'."
        )
        return

    options = [m.experiment_id for m in completed_experiments]
    selected_ids = st.multiselect(
        "Select experiments to compare (minimum 2):",
        options=options,
        default=options[:2],
        format_func=lambda exp_id: f"{exp_id} ({next((getattr(m, 'experiment_type', 'run').capitalize() for m in completed_experiments if m.experiment_id == exp_id), '')})",
    )

    if len(selected_ids) < 2:
        st.warning("Please select at least two experiments to compare.")
        return

    items: list[ComparisonItem] = []
    for exp_id in selected_ids:
        meta = next(
            (m for m in completed_experiments if m.experiment_id == exp_id),
            None,
        )
        if meta:
            try:
                res = load_experiment_result(exp_id)
                items.append(ComparisonItem(metadata=meta, result=res))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to load experiment '{exp_id}': {exc}")
                return

    analysis = build_comparison_analysis(items)

    st.markdown("### Comparability Analysis & Validation Notices")
    for notice in analysis.notices:
        if (
            "differ" in notice.lower()
            or "not evaluate identical" in notice.lower()
        ):
            st.warning(notice)
        else:
            st.info(notice)

    if not analysis.matching_epsilons:
        st.error(
            "No common perturbation radii (ε) found across selected experiments. "
            "Comparison requires at least one matching epsilon value."
        )
        return

    st.markdown(
        f"**Matching perturbation radii compared:** `{list(analysis.matching_epsilons)}`"
    )

    # 10-Metric Comparison Table per matching epsilon
    st.markdown("### 10-Metric Side-by-Side Comparison Table")
    if len(analysis.matching_epsilons) == 1:
        eps = analysis.matching_epsilons[0]
        st.markdown(f"**Perturbation Radius ε = {eps:g}:**")
        table_rows = build_comparison_metric_table(analysis, eps)
        st.dataframe(
            pd.DataFrame(table_rows), hide_index=True, width="stretch"
        )
    else:
        eps_tabs = st.tabs(
            [f"ε = {eps:g}" for eps in analysis.matching_epsilons]
        )
        for eps, tab in zip(analysis.matching_epsilons, eps_tabs, strict=True):
            with tab:
                table_rows = build_comparison_metric_table(analysis, eps)
                st.dataframe(
                    pd.DataFrame(table_rows), hide_index=True, width="stretch"
                )

    # Multi-Experiment Visualizations
    st.markdown("### Comparative Publication Charts")
    comp_series = get_comparison_series_for_plotting(analysis)

    st.markdown("#### 1. Certification Rate Comparison (Refined IBP + LP)")
    render_chart_panel(
        plot_comparison_certification(comp_series),
        pd.json_normalize(comp_series.get("experiments", [])),
        key="comp_cert",
        title="Refined Certification Rate Comparison",
        explanation="Refined certification rates across comparable experiments at matching perturbation radii.",
        unit="%",
    )

    st.markdown("#### 2. Certification Improvement Comparison (Refined - Baseline)")
    render_chart_panel(
        plot_comparison_improvement(comp_series),
        pd.json_normalize(comp_series.get("experiments", [])),
        key="comp_imp",
        title="Certification Improvement Comparison",
        explanation="Refined minus baseline certification rate in percentage points for each experiment.",
        unit="pp",
    )

    st.markdown("#### 3. Output Interval Width Comparison")
    render_chart_panel(
        plot_comparison_widths(comp_series),
        pd.json_normalize(comp_series.get("experiments", [])),
        key="comp_width",
        title="Mean Refined Output Interval Width Comparison",
        explanation="Mean refined output-logit interval width for each experiment (narrower bounds are tighter).",
        unit="Logit units",
    )

    st.markdown("#### 4. Total Refined Verification Runtime Comparison")
    render_chart_panel(
        plot_comparison_runtime(comp_series),
        pd.json_normalize(comp_series.get("experiments", [])),
        key="comp_runtime",
        title="Total Refined Verification Runtime Comparison",
        explanation="Mean total refined verification runtime per sample in seconds.",
        unit="Seconds",
    )


# ==============================================================================
# Main Orchestration & Application Shell
# ==============================================================================


def main() -> None:
    """Initialize resources, theme, navigation drawer, and dispatch the active view."""
    apply_research_theme()

    if not CHECKPOINT_PATH.is_file():
        st.error(
            f"Trained model checkpoint not found at '{CHECKPOINT_PATH}'. Please ensure saved_model.pth exists."
        )
        st.stop()

    try:
        network = cached_network(CHECKPOINT_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the trained model checkpoint: {exc}")
        st.stop()

    try:
        dataset = cached_dataset(DATA_DIR)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the MNIST test dataset: {exc}")
        st.stop()

    # Discover and import legacy experiment directories on startup
    try:
        import_legacy_experiments()
    except Exception:
        pass

    # Populate active experiment session state with the latest completed run if empty
    if st.session_state.get("experiment_result") is None:
        all_exps = list_experiments()
        completed = [e for e in all_exps if e.status == "Completed"]
        if completed:
            latest = completed[0]
            try:
                st.session_state["experiment_result"] = load_experiment_result(
                    latest.experiment_id
                )
                st.session_state["current_experiment_id"] = latest.experiment_id
                st.session_state["current_experiment_metadata"] = latest
            except Exception:
                pass

    # Navigation choices
    nav_options = [
        "Overview",
        "Individual Verification",
        "New Experiment",
        "Experiment Results",
        "Experiment History",
        "Experiment Comparison",
    ]

    # Sidebar Drawer Navigation
    with st.sidebar:
        st.markdown(
            """
            <div class="mc-sidebar-brand">
                <div class="mc-kicker">FORMAL METHODS RESEARCH</div>
                <div class="mc-sidebar-title">MiniCert</div>
                <div class="mc-sidebar-sub">Neural Network Robustness Verifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="mc-sidebar-card">
                <div class="mc-sidebar-card-label">VERIFIED NETWORK</div>
                <div class="mc-sidebar-card-value">MiniCertNet</div>
                <div class="mc-sidebar-card-sub">
                    {network.input_features} &rarr; 100 &rarr; 50 &rarr; {network.output_features} &middot; 4 Layers
                </div>
                <div class="mc-sidebar-tags">
                    <span class="mc-tag">PyTorch</span>
                    <span class="mc-tag">Float64</span>
                    <span class="mc-tag">HiGHS LP</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_id = st.session_state.get("current_experiment_id", "None loaded")
        st.markdown(
            f"""
            <div class="mc-sidebar-card">
                <div class="mc-sidebar-card-label">ACTIVE BENCHMARK RUN</div>
                <div class="mc-sidebar-card-value" style="font-size:0.84rem; font-family:'JetBrains Mono', monospace; word-break:break-all;">
                    {active_id}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "workspace_nav" not in st.session_state:
            st.session_state["workspace_nav"] = "Overview"

        selected_page = st.radio(
            "Workspace Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state["workspace_nav"])
            if st.session_state["workspace_nav"] in nav_options
            else 0,
            key="workspace_nav",
            format_func=lambda p: {
                "Overview": "📋  Overview",
                "Individual Verification": "🔬  Individual Verification",
                "New Experiment": "⚙️  New Experiment",
                "Experiment Results": "📊  Experiment Results",
                "Experiment History": "📜  Experiment History",
                "Experiment Comparison": "⚖️  Experiment Comparison",
            }.get(p, p),
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="mc-sidebar-footer">
                <div style="font-weight:600; color:var(--mc-ink); margin-bottom:4px;">
                    MiniCert Research Workspace
                </div>
                <div>Interval Bound Propagation &bull; Planet Triangle LP &bull; Layer-4 Refinement</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Route view
    if selected_page == "Overview":
        render_overview_section(network)
    elif selected_page == "Individual Verification":
        render_individual_section(network, dataset)
    elif selected_page == "New Experiment":
        render_new_experiment_section(network, dataset)
    elif selected_page == "Experiment Results":
        render_results_section()
    elif selected_page == "Experiment History":
        render_history_section()
    elif selected_page == "Experiment Comparison":
        render_comparison_section()


main()
