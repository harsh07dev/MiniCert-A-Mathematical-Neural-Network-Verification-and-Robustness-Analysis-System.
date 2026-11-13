"""MiniCert Streamlit das   board application entry point.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

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
from evaluation import ExperimentConfig, ExperimentResult, load_mnist_test_dataset, load_trained_network
from evaluation.dataset import flatten_image
from models.model_loader import MathematicalNetwork

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "saved_model.pth"
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="MiniCert Verification Dashboard", layout="wide")


@st.cache_resource
def cached_network(path: Path) -> MathematicalNetwork:
    return load_trained_network(path)


@st.cache_resource
def cached_dataset(data_dir: Path) -> Any:
    return load_mnist_test_dataset(data_dir)


def status_badge(status: str | None) -> str:
    label = status_display(status) if status else "NOT RUN"
    colors = {
        "CERTIFIED": "#1E7A34",
        "INCONCLUSIVE": "#9A6700",
        "ERROR": "#B3261E",
        "COMPLETED": "#1E7A34",
        "FAILED": "#B3261E",
        "INCOMPLETE": "#9A6700",
        "CORRUPTED": "#B3261E",
        "TIGHTENED": "#1E7A34",
        "UNCHANGED": "#555555",
    }
    color = colors.get(label.upper(), "#555555")
    return (
        f"<span style='color:{color}; font-weight:600; "
        f"border:1px solid {color}; border-radius:8px; padding:4px 10px;'>"
        f"{label}</span>"
    )


def render_bounds(lower: np.ndarray, upper: np.ndarray) -> None:
    if lower.size == 0 or upper.size == 0:
        st.caption("Output bounds are not available.")
        return
    rows = [
        {"class": index, "lower": lower[index], "upper": upper[index], "width": upper[index] - lower[index]}
        for index in range(lower.size)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ==============================================================================
# Section 1: Overview
# ==============================================================================


def render_overview_section(network: MathematicalNetwork) -> None:
    overview_columns = st.columns(4)
    overview_columns[0].metric("Architecture", "784 → 64 → 32 → 10")
    overview_columns[1].metric("Dataset", "MNIST test")
    overview_columns[2].metric("Input / output features", f"{network.input_features} / {network.output_features}")
    overview_columns[3].metric("Model", network.source_model)
    st.markdown(
        "- **Verification method:** Interval Bound Propagation (IBP)\n"
        "- **Refinement method:** LP-based Planet/triangle relaxation\n"
        "- **Nominal accuracy reference:** the trained MiniCertNet achieves 96.44% MNIST test accuracy.\n"
        "- **Active experiment:** "
        + (
            f"`{st.session_state.get('current_experiment_id', 'Session run')}`"
            if st.session_state.get("experiment_result")
            else "no experiment loaded yet (see New Experiment or Experiment History)."
        )
    )
    with st.expander("What does certification mean?"):
        st.markdown(
            "For a chosen sample and epsilon, the verifier computes certified lower and upper "
            "bounds on every output logit for **all** inputs inside the L-infinity ball of radius "
            "epsilon. If the lower bound of the target class strictly exceeds the upper bound of "
            "every other class, the prediction is **CERTIFIED** robust for that region. If the "
            "bounds overlap, the result is **INCONCLUSIVE** — the property was not proven, but "
            "the network is never labeled unsafe. LP-based refinement tightens the IBP bounds "
            "using a Planet/triangle relaxation of the ReLU neurons. Bounds remain sound outer "
            "approximations under the floating-point assumptions documented by the verification "
            "engine; this dashboard makes no additional formal claims."
        )


# ==============================================================================
# Section 2: Individual Verification
# ==============================================================================


def dataset_sample(index: int, vector: np.ndarray, label: int, predicted: int) -> Any:
    from evaluation.dataset import InputSample

    return InputSample(index=index, image=vector, label=label, predicted=predicted)


def render_individual_section(network: MathematicalNetwork, dataset: Any) -> None:
    st.header("Individual verification")
    left, right = st.columns(2)
    with left:
        sample_index = int(
            st.number_input("MNIST test sample index", min_value=0, max_value=len(dataset) - 1, value=0, step=1)
        )
    with right:
        epsilon = float(st.slider("Epsilon (L-infinity radius)", min_value=0.0, max_value=0.2, value=0.05, step=0.005))

    image_tensor, label = dataset[sample_index]
    vector = flatten_image(image_tensor)

    prediction_key = ("prediction", sample_index)
    if st.session_state.get(prediction_key) is None:
        logits, predicted = predict_image(network, vector)
        st.session_state[prediction_key] = (predicted, logits)
    predicted, logits = st.session_state[prediction_key]

    image_column, info_column = st.columns([1, 3])
    with image_column:
        st.image(vector.reshape(28, 28), caption=f"Sample {sample_index}", width=160, clamp=True)
    with info_column:
        st.markdown(f"**Actual label:** {label}")
        st.markdown(f"**Model prediction:** {predicted}")
        st.markdown(f"**Target class for verification:** {predicted}")
        st.caption(
            "CERTIFIED means the target class provably dominates all competitors "
            "for every input in the perturbation region. INCONCLUSIVE means the "
            "bounds are too loose to prove that; it is never a claim that the "
            "network is unsafe."
        )

    baseline_button, refined_button = st.columns(2)
    cache_key = (sample_index, epsilon)
    with baseline_button:
        run_baseline = st.button("Run baseline IBP verification", width="stretch")
    with refined_button:
        run_refined = st.button("Run refined IBP + LP verification", width="stretch")

    if run_baseline:
        with st.spinner("Running baseline IBP verification..."):
            st.session_state.setdefault("baseline_cache", {})[cache_key] = run_baseline_verification(
                network, dataset_sample(sample_index, vector, label, predicted), epsilon
            )
    if run_refined:
        st.warning("LP refinement can take a while for a single sample.")
        with st.spinner("Running IBP + LP refinement..."):
            st.session_state.setdefault("refined_cache", {})[cache_key] = run_refined_verification(
                network, dataset_sample(sample_index, vector, label, predicted), epsilon
            )

    baseline_outcome: IndividualOutcome | None = st.session_state.get("baseline_cache", {}).get(cache_key)
    refined_outcome: IndividualOutcome | None = st.session_state.get("refined_cache", {}).get(cache_key)

    st.subheader("Baseline IBP result")
    if baseline_outcome is None and refined_outcome is None:
        st.info("Run one of the verification buttons to see results.")
    else:
        baseline = baseline_outcome if baseline_outcome is not None else refined_outcome
        assert baseline is not None
        st.markdown(status_badge(baseline.baseline_status), unsafe_allow_html=True)
        metric_columns = st.columns(3)
        metric_columns[0].metric("Certification margin", format_optional(baseline.baseline_margin))
        metric_columns[1].metric("Target class", baseline.target_class)
        metric_columns[2].metric("Baseline time (s)", format_optional(baseline.baseline_time_seconds))
        if baseline.baseline_status == "ERROR" and baseline.error:
            st.error(f"Baseline verification failed: {baseline.error}")
        st.markdown("**Baseline output bounds**")
        render_bounds(baseline.baseline_lower, baseline.baseline_upper)

    st.subheader("Refined IBP + LP result")
    if refined_outcome is None:
        st.info("Run the refined verification button to see LP results.")
    else:
        st.markdown(status_badge(refined_outcome.refined_status), unsafe_allow_html=True)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Certification margin", format_optional(refined_outcome.refined_margin))
        metric_columns[1].metric("LP refinement time (s)", format_optional(refined_outcome.lp_refinement_time_seconds))
        metric_columns[2].metric("Total refined time (s)", format_optional(refined_outcome.refined_time_seconds))
        metric_columns[3].metric(
            "Refinements applied",
            f"{refined_outcome.refinement_accepted if refined_outcome.refinement_accepted is not None else '—'}"
            f"/{refined_outcome.refinement_targets if refined_outcome.refinement_targets is not None else '—'}",
        )
        if refined_outcome.refinement_status:
            st.caption(f"Refinement status: {status_display(refined_outcome.refinement_status)}")
        if refined_outcome.refined_status == "ERROR" and refined_outcome.error:
            st.error(f"Refined verification failed: {refined_outcome.error}")
        st.markdown("**Refined output bounds**")
        render_bounds(refined_outcome.refined_lower, refined_outcome.refined_upper)


# ==============================================================================
# Section 3: New Experiment
# ==============================================================================


def render_new_experiment_section(network: MathematicalNetwork, dataset: Any) -> None:
    st.header("New Experiment")
    st.markdown(
        "Configure and execute reproducible experiments comparing baseline IBP against LP-refined verification. "
        "Every run is automatically persisted to persistent storage and added to Experiment History."
    )

    control_columns = st.columns(3)
    with control_columns[0]:
        sample_count = int(st.number_input("Sample count", min_value=1, max_value=10000, value=100, step=1))
    with control_columns[1]:
        epsilon_text = st.text_input("Epsilons (comma-separated)", value="0.01, 0.03, 0.05, 0.1")
    with control_columns[2]:
        seed = int(st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1))

    st.caption(
        "Changing parameters above will NOT trigger execution. Execution starts ONLY when a button below is clicked. "
        "Completed experiments remain stored persistently across reruns and page refreshes."
    )
    development_column, configured_column = st.columns(2)
    with development_column:
        run_development = st.button("Run development experiment (5 samples, epsilon 0.03)", width="stretch")
    with configured_column:
        run_configured = st.button("Run configured experiment", width="stretch")

    if run_development:
        config = ExperimentConfig(sample_count=5, epsilons=(0.03,), seed=42)
        _run_and_persist(config, network, dataset, experiment_type="development")
    elif run_configured:
        try:
            epsilons = parse_epsilon_input(epsilon_text)
            config = ExperimentConfig(
                sample_count=sample_count,
                epsilons=epsilons,
                seed=seed,
                data_dir=DATA_DIR,
                checkpoint=CHECKPOINT_PATH,
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Invalid experiment configuration: {exc}")
        else:
            if sample_count * len(epsilons) >= 200:
                st.warning("This configuration verifies 200 or more sample-epsilon pairs and may take several minutes.")
            _run_and_persist(config, network, dataset, experiment_type="configured")


def _run_and_persist(
    config: ExperimentConfig,
    network: MathematicalNetwork,
    dataset: Any,
    experiment_type: str,
) -> None:
    with st.spinner(f"Running {experiment_type} experiment (this may take a while)..."):
        try:
            result, meta = execute_and_save_experiment(
                config,
                network,
                dataset,
                experiment_type=experiment_type,
            )
        except Exception as exc:  # noqa: BLE001 - dashboard boundary
            st.error(f"Experiment execution failed: {exc}")
            return

    st.session_state["experiment_result"] = result
    st.session_state["current_experiment_id"] = meta.experiment_id
    st.session_state["current_experiment_metadata"] = meta
    st.success(
        f"Experiment completed successfully and saved as `{meta.experiment_id}`! "
        "You can inspect full metrics and charts in **Experiment Results** or browse all runs in **Experiment History**."
    )


# ==============================================================================
# Section 4: Experiment Results
# ==============================================================================


def render_results_section() -> None:
    st.header("Experiment Results")
    result: ExperimentResult | None = st.session_state.get("experiment_result")
    exp_id = st.session_state.get("current_experiment_id")

    if result is None:
        st.info(
            "No experiment loaded or executed yet. Run a new experiment in 'New Experiment' "
            "or select an experiment from 'Experiment History'."
        )
        return

    selection = result.selection
    excluded_label = "sample" if selection.incorrect_excluded == 1 else "samples"
    exp_title = f"`{exp_id}` ({result.config.experiment_name})" if exp_id else f"`{result.config.experiment_name}`"
    st.markdown(f"### Active Experiment: {exp_title}")
    st.caption(
        f"Evaluated {selection.selected_count} correctly classified samples, excluded {selection.incorrect_excluded} "
        f"incorrectly classified {excluded_label}, tested across {len(result.metrics)} epsilon value(s)."
    )

    for entry in result.metrics:
        certification = entry.certification
        columns = st.columns(4)
        columns[0].metric(
            f"Baseline certified @ eps {entry.epsilon:g}",
            format_optional(None if certification.baseline_rate is None else certification.baseline_rate * 100, 3) + "%",
        )
        columns[1].metric(
            f"Refined certified @ eps {entry.epsilon:g}",
            format_optional(None if certification.refined_rate is None else certification.refined_rate * 100, 3) + "%",
        )
        columns[2].metric(
            f"Improvement @ eps {entry.epsilon:g}",
            format_optional(certification.improvement_pp, 3) + " pp",
        )
        columns[3].metric(
            f"Errors @ eps {entry.epsilon:g}",
            certification.baseline_errors + certification.refined_errors,
        )

    st.subheader("Certification rate comparison")
    st.pyplot(plot_certification_rates(certification_rate_series(result.metrics)))

    st.subheader("Output interval width")
    st.pyplot(plot_interval_widths(width_series(result.metrics)))

    st.subheader("Runtime comparison")
    st.pyplot(plot_runtime(runtime_series(result.metrics)))

    st.subheader("Certification improvement")
    st.pyplot(plot_improvement(improvement_series(result.metrics)))

    st.subheader("Per-sample results")
    rows = records_table_rows(result)
    st.dataframe(pd.DataFrame(rows, columns=PER_SAMPLE_COLUMNS), hide_index=True, width="stretch")

    st.subheader("Exports")
    try:
        export_paths = result.save()
    except Exception as exc:  # noqa: BLE001 - dashboard boundary
        st.error(f"Export failed: {exc}")
        return
    export_columns = st.columns(2)
    for column, (key, label) in zip(
        export_columns,
        [("samples_csv", "Download per-sample CSV"), ("summary_json", "Download aggregated JSON")],
        strict=True,
    ):
        path = export_paths[key]
        with column:
            st.download_button(
                label=label,
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv" if key == "samples_csv" else "application/json",
                width="stretch",
            )


# ==============================================================================
# Section 5: Experiment History
# ==============================================================================


def render_history_section() -> None:
    hdr_col, btn_col = st.columns([3, 1])
    with hdr_col:
        st.header("Experiment History")
    with btn_col:
        if st.button("🔄 Refresh History", width="stretch", help="Rescan storage and discover new completed runs"):
            newly_imported = import_legacy_experiments()
            if newly_imported:
                st.success(f"Discovered and imported {len(newly_imported)} experiment(s)!")
            else:
                st.info("History is up to date.")
            st.rerun()

    st.markdown(
        "Browse, inspect, and export previously completed evaluation runs. "
        "Historical results are loaded directly from disk without recomputing any verification queries."
    )

    all_experiments = list_experiments()
    if not all_experiments:
        st.info("No experiment records found in persistent storage.")
        return

    # Table view of all experiments
    rows = history_table_rows(all_experiments)
    st.dataframe(pd.DataFrame(rows, columns=HISTORY_TABLE_COLUMNS), hide_index=True, width="stretch")

    st.subheader("Inspect Experiment")
    options = [m.experiment_id for m in all_experiments]
    selected_id = st.selectbox(
        "Choose an experiment to inspect details and historical charts",
        options=options,
        index=0,
        format_func=lambda exp_id: f"{exp_id} ({next((m.experiment_type.capitalize() for m in all_experiments if m.experiment_id == exp_id), '')})",
    )

    selected_meta = next((m for m in all_experiments if m.experiment_id == selected_id), None)
    if selected_meta is None:
        return

    st.markdown(f"**Status:** {status_badge(selected_meta.status)}", unsafe_allow_html=True)
    if selected_meta.error_message:
        st.warning(f"Note: {selected_meta.error_message}")

    if selected_meta.status not in ("Completed", "Failed"):
        st.info("This experiment folder is incomplete or corrupted and cannot be fully loaded.")
        return

    try:
        loaded_result = load_experiment_result(selected_id)
    except Exception as exc:
        st.error(f"Failed to load experiment result: {exc}")
        return

    action_col1, action_col2 = st.columns([1, 3])
    with action_col1:
        if st.button("Load into Experiment Results", width="stretch"):
            st.session_state["experiment_result"] = loaded_result
            st.session_state["current_experiment_id"] = selected_id
            st.session_state["current_experiment_metadata"] = selected_meta
            st.success(f"Experiment `{selected_id}` is now active in **Experiment Results**!")

    with st.expander("Configuration", expanded=False):
        st.json(loaded_result.config.as_config_dict())

    st.markdown("#### Aggregated Metrics")
    for entry in loaded_result.metrics:
        cert = entry.certification
        cols = st.columns(4)
        cols[0].metric(
            f"Baseline rate @ eps {entry.epsilon:g}",
            format_optional(None if cert.baseline_rate is None else cert.baseline_rate * 100, 3) + "%",
        )
        cols[1].metric(
            f"Refined rate @ eps {entry.epsilon:g}",
            format_optional(None if cert.refined_rate is None else cert.refined_rate * 100, 3) + "%",
        )
        cols[2].metric(
            f"Improvement @ eps {entry.epsilon:g}",
            format_optional(cert.improvement_pp, 3) + " pp",
        )
        cols[3].metric(
            f"Inconclusive @ eps {entry.epsilon:g}",
            f"{cert.baseline_inconclusive} / {cert.refined_inconclusive}",
        )

    with st.expander("Per-Sample Results Table", expanded=False):
        sample_rows = records_table_rows(loaded_result)
        st.dataframe(pd.DataFrame(sample_rows, columns=PER_SAMPLE_COLUMNS), hide_index=True, width="stretch")

    st.markdown("#### Historical Visualizations")
    tab_chart1, tab_chart2, tab_chart3, tab_chart4 = st.tabs([
        "Certification Rates",
        "Interval Widths",
        "Runtime Comparison",
        "Certification Improvement",
    ])
    with tab_chart1:
        st.pyplot(plot_certification_rates(certification_rate_series(loaded_result.metrics)))
    with tab_chart2:
        st.pyplot(plot_interval_widths(width_series(loaded_result.metrics)))
    with tab_chart3:
        st.pyplot(plot_runtime(runtime_series(loaded_result.metrics)))
    with tab_chart4:
        st.pyplot(plot_improvement(improvement_series(loaded_result.metrics)))

    st.markdown("#### Download Exports")
    exp_folder = selected_meta.path
    samples_file = exp_folder / "samples.csv"
    summary_file = exp_folder / "summary.json"
    dl_col1, dl_col2 = st.columns(2)
    if samples_file.is_file():
        with dl_col1:
            st.download_button(
                label="Download samples.csv",
                data=samples_file.read_bytes(),
                file_name=f"{selected_id}_samples.csv",
                mime="text/csv",
                width="stretch",
            )
    if summary_file.is_file():
        with dl_col2:
            st.download_button(
                label="Download summary.json",
                data=summary_file.read_bytes(),
                file_name=f"{selected_id}_summary.json",
                mime="application/json",
                width="stretch",
            )


# ==============================================================================
# Section 6: Experiment Comparison
# ==============================================================================


def render_comparison_section() -> None:
    st.header("Experiment Comparison")
    st.markdown(
        "Compare two or more completed evaluation experiments across matching epsilon values. "
        "Evaluates certification rates, interval-width tightening, and solver runtimes side by side."
    )

    all_experiments = list_experiments()
    completed_experiments = [m for m in all_experiments if m.status == "Completed"]

    if len(completed_experiments) < 2:
        st.info(
            "At least two completed experiments are required for comparison. "
            "Run additional experiments in the 'New Experiment' section."
        )
        return

    options = [m.experiment_id for m in completed_experiments]
    default_selection = options[:2]

    selected_ids = st.multiselect(
        "Select experiments to compare (minimum 2)",
        options=options,
        default=default_selection,
        format_func=lambda exp_id: f"{exp_id} ({next((m.experiment_type.capitalize() for m in completed_experiments if m.experiment_id == exp_id), '')})",
    )

    if len(selected_ids) < 2:
        st.warning("Please select at least two experiments to compare.")
        return

    items: list[ComparisonItem] = []
    for exp_id in selected_ids:
        meta = next((m for m in completed_experiments if m.experiment_id == exp_id), None)
        if meta:
            try:
                res = load_experiment_result(exp_id)
                items.append(ComparisonItem(metadata=meta, result=res))
            except Exception as exc:
                st.error(f"Failed to load experiment '{exp_id}': {exc}")
                return

    analysis = build_comparison_analysis(items)

    st.markdown("### Comparability Analysis & Notices")
    for notice in analysis.notices:
        if "differ" in notice.lower() or "not evaluate identical" in notice.lower():
            st.warning(notice)
        else:
            st.info(notice)

    if not analysis.matching_epsilons:
        st.error(
            "No common epsilon values found across the selected experiments. "
            "Comparison requires at least one matching perturbation radius."
        )
        return

    st.markdown(f"**Matching perturbation radii compared:** `{list(analysis.matching_epsilons)}`")

    # 10-Metric Comparison Table per matching epsilon
    st.subheader("10-Metric Detailed Comparison Table")
    if len(analysis.matching_epsilons) == 1:
        eps = analysis.matching_epsilons[0]
        st.markdown(f"**Perturbation radius $\\epsilon = {eps:g}$:**")
        table_rows = build_comparison_metric_table(analysis, eps)
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
    else:
        eps_tabs = st.tabs([f"Epsilon {eps:g}" for eps in analysis.matching_epsilons])
        for eps, tab in zip(analysis.matching_epsilons, eps_tabs, strict=True):
            with tab:
                table_rows = build_comparison_metric_table(analysis, eps)
                st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    # Visualizations
    st.subheader("Visual Comparisons across Matching Epsilons")
    comp_series = get_comparison_series_for_plotting(analysis)

    st.markdown("#### 1. Certification Rate Comparison (Refined IBP + LP)")
    st.pyplot(plot_comparison_certification(comp_series))

    st.markdown("#### 2. Certification Improvement Comparison (Refined - Baseline)")
    st.pyplot(plot_comparison_improvement(comp_series))

    st.markdown("#### 3. Output Interval Width Comparison")
    st.pyplot(plot_comparison_widths(comp_series))

    st.markdown("#### 4. Total Refined Verification Runtime Comparison")
    st.pyplot(plot_comparison_runtime(comp_series))


# ==============================================================================
# Main Orchestration
# ==============================================================================


def main() -> None:
    st.title("MiniCert Verification Dashboard")
    st.markdown(
        "Mathematical neural-network verification, robustness certification, "
        "and empirical refinement evaluation."
    )

    if not CHECKPOINT_PATH.is_file():
        st.error(f"Trained model checkpoint not found at '{CHECKPOINT_PATH}'.")
        st.stop()

    try:
        network = cached_network(CHECKPOINT_PATH)
    except Exception as exc:  # noqa: BLE001 - dashboard boundary
        st.error(f"Could not load the trained model: {exc}")
        st.stop()

    try:
        dataset = cached_dataset(DATA_DIR)
    except Exception as exc:  # noqa: BLE001 - dashboard boundary
        st.error(f"Could not load the MNIST test dataset: {exc}")
        st.stop()

    # Import legacy experiments on first startup if results/ has them
    try:
        import_legacy_experiments()
    except Exception:
        pass

    # Populate session_state with latest completed experiment if empty
    if st.session_state.get("experiment_result") is None:
        all_exps = list_experiments()
        completed = [e for e in all_exps if e.status == "Completed"]
        if completed:
            latest = completed[0]
            try:
                st.session_state["experiment_result"] = load_experiment_result(latest.experiment_id)
                st.session_state["current_experiment_id"] = latest.experiment_id
                st.session_state["current_experiment_metadata"] = latest
            except Exception:
                pass

    tab_overview, tab_individual, tab_new_exp, tab_results, tab_history, tab_comparison = st.tabs([
        "1. Overview",
        "2. Individual Verification",
        "3. New Experiment",
        "4. Experiment Results",
        "5. Experiment History",
        "6. Experiment Comparison",
    ])

    with tab_overview:
        render_overview_section(network)

    with tab_individual:
        render_individual_section(network, dataset)

    with tab_new_exp:
        render_new_experiment_section(network, dataset)

    with tab_results:
        render_results_section()

    with tab_history:
        render_history_section()

    with tab_comparison:
        render_comparison_section()


main()
