"""Matplotlib figures for the dashboard, built strictly from real results.

Streamlit is not imported here. Only finite measurements are plotted: missing
values (None or NaN) are marked as ``n/a`` and are never converted to zero.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised only without matplotlib
    plt = None  # type: ignore[assignment]

COLOR_BASELINE = "#786B67"
COLOR_REFINED = "#702B3A"
COLOR_LP = "#627D68"
COLOR_IMPROVEMENT = "#627D68"
COLOR_NEGATIVE = "#963E50"
COLOR_ZERO = "#A7793D"
TEXT_COLOR = "#30282A"
MUTED_TEXT_COLOR = "#827875"
MISSING_COLOR = "#827875"
GRID_COLOR = "#E5DCD3"
SPINE_COLOR = "#C9BEB5"
FIGURE_SIZE = (8.0, 4.6)


def _figure() -> Any:
    if plt is None:
        raise ImportError("matplotlib is required for dashboard visualizations")
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=120)
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    axis.grid(True, color=GRID_COLOR, linewidth=0.65, alpha=0.9)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(SPINE_COLOR)
    axis.tick_params(colors=TEXT_COLOR, labelsize=10)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.xaxis.label.set_size(11)
    axis.yaxis.label.set_size(11)
    return figure, axis


def _set_titles(axis: Any, title: str, ylabel: str) -> None:
    axis.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12)
    axis.set_xlabel("Epsilon")
    axis.set_ylabel(ylabel)


def _epsilon_labels(entries: Sequence[dict[str, Any]]) -> list[str]:
    return [f"{float(entry['epsilon']):g}" for entry in entries]


def _configure_x_axis(axis: Any, entries: Sequence[dict[str, Any]]) -> None:
    positions = list(range(len(entries)))
    axis.set_xticks(positions)
    axis.set_xticklabels(_epsilon_labels(entries))
    axis.set_xlim(-0.6, max(0.6, len(entries) - 0.4))


def _is_present(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _present(values: Sequence[Any]) -> list[float]:
    return [float(value) for value in values if _is_present(value)]


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.0001 or abs(value) >= 10000:
        return f"{value:.2e}"
    return f"{value:.3g}"


def _format_percentage(value: float) -> str:
    return f"{_format_number(value)}%"


def _format_improvement(value: float) -> str:
    prefix = "+" if value > 0.0 else ""
    return f"{prefix}{_format_number(value)} pp"


def _value_label(
    axis: Any,
    x: float,
    y: float,
    text: str,
    *,
    dx: int = 0,
    above: bool = True,
    horizontal_alignment: str = "center",
) -> None:
    axis.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, 8 if above else -15),
        ha=horizontal_alignment,
        va="bottom" if above else "top",
        fontsize=9,
        color=TEXT_COLOR,
        annotation_clip=False,
    )


def _missing_label(axis: Any, x: float, *, level: int = 0) -> None:
    axis.text(
        x,
        0.025 + 0.055 * level,
        "n/a",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color=MISSING_COLOR,
    )


def _empty_message(axis: Any, message: str) -> None:
    axis.text(
        0.5,
        0.55,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED_TEXT_COLOR,
    )


def _add_legend(axis: Any) -> None:
    handles, _ = axis.get_legend_handles_labels()
    if not handles:
        return
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(3, len(handles)),
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)


def _finish(figure: Any, axis: Any, *, legend: bool = True) -> Any:
    if legend:
        _add_legend(axis)
    figure.tight_layout()
    return figure


def plot_certification_rates(series: Sequence[dict[str, Any]]) -> Any:
    """Line chart of certification percentage versus epsilon for both methods."""
    figure, axis = _figure()
    _set_titles(axis, "Certification rate comparison", "Certification rate (%)")
    _configure_x_axis(axis, series)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 20, 40, 60, 80, 100])

    definitions = (
        ("baseline_percent", "Baseline IBP", COLOR_BASELINE, "o", -6, -0.06),
        ("refined_percent", "Refined IBP + LP", COLOR_REFINED, "s", 6, 0.06),
    )
    for key, label, color, marker, label_dx, missing_dx in definitions:
        values = [entry[key] for entry in series]
        points = [
            (position + missing_dx / 2.0, float(value))
            for position, value in enumerate(values)
            if _is_present(value)
        ]
        if points:
            xs, ys = zip(*points)
            axis.plot(
                xs,
                ys,
                marker=marker,
                markersize=7,
                linewidth=2,
                label=label,
                color=color,
            )
            for x, y in points:
                _value_label(
                    axis,
                    x,
                    y,
                    _format_percentage(y),
                    dx=label_dx,
                    above=y < 95.0,
                    horizontal_alignment="right" if label_dx < 0 else "left",
                )
        for position, value in enumerate(values):
            if not _is_present(value):
                _missing_label(axis, position + missing_dx)

    if not _present([entry[key] for entry in series for key, _, _, _, _, _ in definitions]):
        _empty_message(axis, "No certification-rate measurements available")
    return _finish(figure, axis)


def plot_interval_widths(series: Sequence[dict[str, Any]]) -> Any:
    """Grouped bars of mean output interval widths, with relative reduction."""
    figure, axis = _figure()
    _set_titles(
        axis,
        "Output interval width comparison",
        "Mean output interval width (logit units)",
    )
    _configure_x_axis(axis, series)

    definitions = (
        ("baseline_width", "Baseline IBP", COLOR_BASELINE, -0.2),
        ("refined_width", "Refined IBP + LP", COLOR_REFINED, 0.2),
    )
    for key, label, color, offset in definitions:
        labeled = False
        for position, entry in enumerate(series):
            value = entry[key]
            x = position + offset
            if not _is_present(value):
                _missing_label(axis, x)
                continue
            numeric = float(value)
            axis.bar(
                [x],
                [numeric],
                width=0.38,
                label=label if not labeled else None,
                color=color,
            )
            labeled = True
            if numeric == 0.0:
                axis.scatter([x], [0.0], marker="_", s=110, linewidths=2, color=color, zorder=4)
            _value_label(axis, x, numeric, _format_number(numeric))

    for position, entry in enumerate(series):
        baseline = entry["baseline_width"]
        refined = entry["refined_width"]
        if not (_is_present(baseline) and _is_present(refined)):
            continue
        baseline_value = float(baseline)
        refined_value = float(refined)
        if baseline_value > 0.0:
            reduction = 100.0 * (baseline_value - refined_value) / baseline_value
            if reduction > 0.0:
                note = f"{reduction:.1f}% narrower"
            elif reduction < 0.0:
                note = f"{-reduction:.1f}% wider"
            else:
                note = "same width"
            axis.text(
                position,
                0.97,
                note,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=MUTED_TEXT_COLOR,
            )

    values = _present(
        [entry[key] for entry in series for key, _, _, _ in definitions]
    )
    maximum = max(values, default=0.0)
    axis.set_ylim(0, maximum * 1.3 if maximum > 0.0 else 1.0)
    if not values:
        _empty_message(axis, "No width measurements available")
    return _finish(figure, axis)


def plot_runtime(series: Sequence[dict[str, Any]]) -> Any:
    """Grouped bars of baseline, LP, and total refined runtimes in seconds."""
    figure, axis = _figure()
    _set_titles(axis, "Runtime comparison", "Mean runtime (s)")
    _configure_x_axis(axis, series)

    definitions = (
        ("baseline_time", "Baseline IBP", COLOR_BASELINE, -0.28),
        ("lp_refinement_time", "LP refinement", COLOR_LP, 0.0),
        ("refined_total_time", "Total refined verification", COLOR_REFINED, 0.28),
    )
    values = _present([entry[key] for entry in series for key, _, _, _ in definitions])
    positive = [value for value in values if value > 0.0]
    use_log_scale = (
        len(positive) == len(values)
        and bool(positive)
        and max(positive) / min(positive) >= 100.0
    )

    for key, label, color, offset in definitions:
        labeled = False
        for position, entry in enumerate(series):
            value = entry[key]
            x = position + offset
            if not _is_present(value):
                _missing_label(axis, x)
                continue
            numeric = float(value)
            axis.bar(
                [x],
                [numeric],
                width=0.25,
                label=label if not labeled else None,
                color=color,
            )
            labeled = True
            if numeric == 0.0:
                axis.scatter([x], [0.0], marker="_", s=100, linewidths=2, color=color, zorder=4)
            _value_label(axis, x, numeric, _format_number(numeric))

    if use_log_scale:
        axis.set_yscale("log")
        axis.set_ylim(min(positive) * 0.55, max(positive) * 3.0)
        axis.set_ylabel("Mean runtime (s, log scale)")
    else:
        maximum = max(values, default=0.0)
        axis.set_ylim(0, maximum * 1.35 if maximum > 0.0 else 1.0)
    if not values:
        _empty_message(axis, "No runtime measurements available")
    return _finish(figure, axis)


def plot_improvement(series: Sequence[dict[str, Any]]) -> Any:
    """Bars and markers of refined minus baseline certification, in pp."""
    figure, axis = _figure()
    _set_titles(
        axis,
        "Certification improvement (refined - baseline)",
        "Certification improvement (percentage points)",
    )
    _configure_x_axis(axis, series)

    values = [entry["improvement_pp"] for entry in series]
    for position, value in enumerate(values):
        if not _is_present(value):
            _missing_label(axis, position)
            continue
        numeric = float(value)
        color = (
            COLOR_IMPROVEMENT
            if numeric > 0.0
            else COLOR_NEGATIVE
            if numeric < 0.0
            else COLOR_ZERO
        )
        axis.bar([position], [numeric], width=0.55, color=color)
        axis.scatter(
            [position],
            [numeric],
            s=48,
            color=color,
            edgecolor=TEXT_COLOR,
            linewidth=0.6,
            zorder=4,
        )
        _value_label(
            axis,
            position,
            numeric,
            _format_improvement(numeric),
            above=numeric >= 0.0,
        )

    axis.axhline(0.0, color=TEXT_COLOR, linewidth=1.0, zorder=3)
    present = _present(values)
    minimum = min([0.0, *present])
    maximum = max([0.0, *present])
    margin = max(1.0, (maximum - minimum) * 0.3)
    axis.set_ylim(minimum - margin, maximum + margin)
    if not present:
        _empty_message(axis, "No improvement measurements available")
    return _finish(figure, axis, legend=False)


# ==============================================================================
# Multi-Experiment Comparison Plots
# ==============================================================================

EXP_PALETTE = ("#702B3A", "#627D68", "#A7793D", "#596F83", "#963E50", "#6C626A", "#8C765C", "#455D50")


def _comparison_figure_setup(
    comparison_data: dict[str, Any], title: str, ylabel: str
) -> tuple[Any, Any, list[float], list[dict[str, Any]], int, float]:
    figure, axis = _figure()
    _set_titles(axis, title, ylabel)
    matching_epsilons = [float(e) for e in comparison_data.get("matching_epsilons", [])]
    experiments = comparison_data.get("experiments", [])
    num_exps = max(1, len(experiments))

    dummy_entries = [{"epsilon": eps} for eps in matching_epsilons]
    _configure_x_axis(axis, dummy_entries)

    cluster_width = 0.72
    bar_width = cluster_width / num_exps

    return figure, axis, matching_epsilons, experiments, num_exps, bar_width


def plot_comparison_certification(comparison_data: dict[str, Any]) -> Any:
    """Grouped bar chart comparing refined certification rates across experiments."""
    figure, axis, matching_epsilons, experiments, num_exps, bar_width = _comparison_figure_setup(
        comparison_data,
        "Certification rate comparison (Refined IBP + LP)",
        "Refined certification rate (%)",
    )
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 20, 40, 60, 80, 100])

    if not matching_epsilons or not experiments:
        _empty_message(axis, "No matching epsilon values to compare")
        return _finish(figure, axis, legend=False)

    all_present_values: list[float] = []
    for exp_idx, exp in enumerate(experiments):
        color = EXP_PALETTE[exp_idx % len(EXP_PALETTE)]
        offset = (exp_idx - (num_exps - 1) / 2.0) * bar_width
        label = exp.get("label", exp.get("id", f"Exp {exp_idx + 1}"))
        labeled = False

        for pos, eps in enumerate(matching_epsilons):
            x = pos + offset
            metrics_dict = exp.get("per_epsilon", {}).get(eps)
            val = metrics_dict.get("refined_rate_pct") if metrics_dict else None
            if not _is_present(val):
                _missing_label(axis, x)
                continue
            numeric = float(val)
            all_present_values.append(numeric)
            axis.bar(
                [x],
                [numeric],
                width=bar_width * 0.88,
                label=label if not labeled else None,
                color=color,
            )
            labeled = True
            _value_label(axis, x, numeric, _format_percentage(numeric))

    if not all_present_values:
        _empty_message(axis, "No certification rate measurements available")
    return _finish(figure, axis, legend=True)


def plot_comparison_widths(comparison_data: dict[str, Any]) -> Any:
    """Grouped bar chart comparing refined mean output interval widths across experiments."""
    figure, axis, matching_epsilons, experiments, num_exps, bar_width = _comparison_figure_setup(
        comparison_data,
        "Refined output interval width comparison",
        "Mean output interval width (logit units)",
    )

    if not matching_epsilons or not experiments:
        _empty_message(axis, "No matching epsilon values to compare")
        return _finish(figure, axis, legend=False)

    all_present_values: list[float] = []
    for exp_idx, exp in enumerate(experiments):
        color = EXP_PALETTE[exp_idx % len(EXP_PALETTE)]
        offset = (exp_idx - (num_exps - 1) / 2.0) * bar_width
        label = exp.get("label", exp.get("id", f"Exp {exp_idx + 1}"))
        labeled = False

        for pos, eps in enumerate(matching_epsilons):
            x = pos + offset
            metrics_dict = exp.get("per_epsilon", {}).get(eps)
            val = metrics_dict.get("refined_width") if metrics_dict else None
            if not _is_present(val):
                _missing_label(axis, x)
                continue
            numeric = float(val)
            all_present_values.append(numeric)
            axis.bar(
                [x],
                [numeric],
                width=bar_width * 0.88,
                label=label if not labeled else None,
                color=color,
            )
            labeled = True
            _value_label(axis, x, numeric, _format_number(numeric))

    maximum = max(all_present_values, default=0.0)
    axis.set_ylim(0, maximum * 1.3 if maximum > 0.0 else 1.0)
    if not all_present_values:
        _empty_message(axis, "No width measurements available")
    return _finish(figure, axis, legend=True)


def plot_comparison_runtime(comparison_data: dict[str, Any]) -> Any:
    """Grouped bar chart comparing total refined verification runtimes across experiments."""
    figure, axis, matching_epsilons, experiments, num_exps, bar_width = _comparison_figure_setup(
        comparison_data,
        "Total refined verification runtime comparison",
        "Mean runtime (s)",
    )

    if not matching_epsilons or not experiments:
        _empty_message(axis, "No matching epsilon values to compare")
        return _finish(figure, axis, legend=False)

    all_present_values: list[float] = []
    for exp_idx, exp in enumerate(experiments):
        color = EXP_PALETTE[exp_idx % len(EXP_PALETTE)]
        offset = (exp_idx - (num_exps - 1) / 2.0) * bar_width
        label = exp.get("label", exp.get("id", f"Exp {exp_idx + 1}"))
        labeled = False

        for pos, eps in enumerate(matching_epsilons):
            x = pos + offset
            metrics_dict = exp.get("per_epsilon", {}).get(eps)
            val = metrics_dict.get("refined_total_time") if metrics_dict else None
            if not _is_present(val):
                _missing_label(axis, x)
                continue
            numeric = float(val)
            all_present_values.append(numeric)
            axis.bar(
                [x],
                [numeric],
                width=bar_width * 0.88,
                label=label if not labeled else None,
                color=color,
            )
            labeled = True
            _value_label(axis, x, numeric, _format_number(numeric))

    maximum = max(all_present_values, default=0.0)
    axis.set_ylim(0, maximum * 1.35 if maximum > 0.0 else 1.0)
    if not all_present_values:
        _empty_message(axis, "No runtime measurements available")
    return _finish(figure, axis, legend=True)


def plot_comparison_improvement(comparison_data: dict[str, Any]) -> Any:
    """Grouped bar chart comparing certification improvement (pp) across experiments."""
    figure, axis, matching_epsilons, experiments, num_exps, bar_width = _comparison_figure_setup(
        comparison_data,
        "Certification improvement comparison (Refined - Baseline)",
        "Improvement (percentage points)",
    )

    if not matching_epsilons or not experiments:
        _empty_message(axis, "No matching epsilon values to compare")
        return _finish(figure, axis, legend=False)

    all_present_values: list[float] = []
    for exp_idx, exp in enumerate(experiments):
        color = EXP_PALETTE[exp_idx % len(EXP_PALETTE)]
        offset = (exp_idx - (num_exps - 1) / 2.0) * bar_width
        label = exp.get("label", exp.get("id", f"Exp {exp_idx + 1}"))
        labeled = False

        for pos, eps in enumerate(matching_epsilons):
            x = pos + offset
            metrics_dict = exp.get("per_epsilon", {}).get(eps)
            val = metrics_dict.get("improvement_pp") if metrics_dict else None
            if not _is_present(val):
                _missing_label(axis, x)
                continue
            numeric = float(val)
            all_present_values.append(numeric)
            axis.bar(
                [x],
                [numeric],
                width=bar_width * 0.88,
                label=label if not labeled else None,
                color=color,
            )
            labeled = True
            _value_label(
                axis,
                x,
                numeric,
                _format_improvement(numeric),
                above=numeric >= 0.0,
            )

    axis.axhline(0.0, color=TEXT_COLOR, linewidth=1.0, zorder=3)
    present = _present(all_present_values)
    minimum = min([0.0, *present])
    maximum = max([0.0, *present])
    margin = max(1.0, (maximum - minimum) * 0.3)
    axis.set_ylim(minimum - margin, maximum + margin)
    if not all_present_values:
        _empty_message(axis, "No improvement measurements available")
    return _finish(figure, axis, legend=True)

