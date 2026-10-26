import altair as alt
import marimo as mo
import numpy as np
import pandas as pd

from blackwell_ita.evaluation import (
    METHOD_WELFARES,
    WELFARE_METHODS,
    WELFARES,
    bootstrap_summary,
)

MODEL_LABELS = {
    "gold": "Gold",
    "qwen3_0_6b_bradley_terry": "Qwen3 0.6B",
    "qwen3_1_7b_bradley_terry": "Qwen3 1.7B",
    "qwen3_4b_bradley_terry": "Qwen3 4B",
}
METHOD_LABELS = {
    "uniform": "Uniform pool",
    "max_mean_win_rate": "Max mean win rate",
    "best_of_n": "Best-of-N",
    "maximin": "Maximin",
    "scalarised_nash": "Best-of-Nash",
    "blackwell": "BoB (orthant)",
    "blackwell_nash": "BoB (Nash target)",
    "robust_nash": "Direct Nash",
    "budgeted_nash": "Robust Nash (error budget)",
}
# Fixed categorical order, so each method keeps its colour in every chart
METHOD_COLOURS = [
    "#8a8984",
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#6250d6",
    "#e34948",
    "#5b3a1e",
]
WELFARE_LABELS = {
    "egalitarian": "Egalitarian",
    "utilitarian": "Utilitarian",
    "nash": "Nash",
}
TARGET_LABELS = {"orthant": "Orthant", "half_space": "Half-space", "nash": "Nash"}
LEVEL_LABELS = {
    "whole_group": "Whole-group only",
    "mixed": "Six of each",
    "single_participant": "Single-participant only",
}
NEUTRAL_COLOUR = "#8a8984"


def method_colour_scale(method_names: list[str]) -> alt.Scale:
    """The labels of ``method_names`` mapped to their fixed colours."""
    return alt.Scale(
        domain=[METHOD_LABELS[method_name] for method_name in method_names],
        range=[
            colour
            for method_name, colour in zip(METHOD_LABELS, METHOD_COLOURS, strict=True)
            if method_name in method_names
        ],
    )


def panel_methods(welfare: str) -> list[str]:
    """The baselines and Best-of-Nash, then the method that targets ``welfare``."""
    shown = ["max_mean_win_rate", "best_of_n", "maximin", "scalarised_nash"]
    return shown + [WELFARE_METHODS[welfare]] * (WELFARE_METHODS[welfare] not in shown)


def present_methods(frame: pd.DataFrame) -> list[str]:
    """The methods in ``frame``, in label order."""
    return [name for name in METHOD_LABELS if name in set(frame["method"])]


def interval_layers(
    frame: pd.DataFrame, x_title: str, y: alt.Y, colour: alt.Color, tooltip: list
) -> alt.LayerChart:
    """A horizontal 95% interval rule with the mean as a point."""
    intervals = (
        alt.Chart()
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("interval_low:Q", title=x_title),
            x2="interval_high:Q",
            y=y,
            color=colour,
        )
    )
    means = (
        alt.Chart()
        .mark_point(filled=True, size=90, opacity=1)
        .encode(x="value:Q", y=y, color=colour, tooltip=tooltip)
    )
    return alt.layer(intervals, means, data=frame)  # pyright: ignore[reportReturnType]


def regret_chart(results: pd.DataFrame, title: str) -> alt.FacetChart:
    """Mean welfare regret of each method with a 95% bootstrap interval over instances.

    One column per welfare index and one row per selector. The uniform pool's
    regret is several times every other method's, so it is left out.
    """
    results = results[results["method"] != "uniform"]  # pyright: ignore[reportAssignmentType]
    long_results = results.melt(
        id_vars=["selector", "method", "question_id"],
        value_vars=[f"{welfare}_regret" for welfare in WELFARES],
        var_name="welfare",
        value_name="value",
    )
    long_results["welfare"] = long_results["welfare"].str.removesuffix("_regret")
    summary = bootstrap_summary(
        long_results, ["selector", "method", "welfare"], "value"
    )
    methods = present_methods(results)
    frame = summary.assign(
        selector_label=summary["selector"].map(MODEL_LABELS),  # pyright: ignore[reportArgumentType]
        method_label=summary["method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType]
        welfare_label=summary["welfare"].map(WELFARE_LABELS),  # pyright: ignore[reportArgumentType]
    )
    layers = interval_layers(
        frame,
        "Welfare regret",
        alt.Y(
            "method_label:N",
            sort=[METHOD_LABELS[name] for name in methods],
            title=None,
        ),
        alt.Color("method_label:N", scale=method_colour_scale(methods), legend=None),
        [
            alt.Tooltip("method_label:N", title="Method"),
            alt.Tooltip("welfare_label:N", title="Welfare"),
            alt.Tooltip("value:Q", title="Regret", format=".4f"),
            alt.Tooltip("interval_low:Q", title="Interval low", format=".4f"),
            alt.Tooltip("interval_high:Q", title="Interval high", format=".4f"),
            alt.Tooltip("instances:Q", title="Instances"),
        ],
    )
    return (
        layers.properties(width=260)
        .facet(
            row=alt.Row(
                "selector_label:N",
                sort=[
                    MODEL_LABELS[name]
                    for name in MODEL_LABELS
                    if name in set(results["selector"])
                ],
                title=None,
            ),
            column=alt.Column(
                "welfare_label:N",
                sort=[WELFARE_LABELS[welfare] for welfare in WELFARES],
                title=None,
            ),
        )
        .properties(title=title)
    )


def welfare_row_chart(
    results: pd.DataFrame, x_column: str, x_title: str, x_sort: list[str] | None = None
) -> alt.FacetChart:
    """Mean robust welfare of each method against ``x_column``, one column per welfare index.

    Each panel shows the baselines, Best-of-Nash and the method that targets
    that welfare. Points are means over groups with 95% bootstrap intervals.
    ``x_sort`` makes the axis an ordered category.
    """
    summary = pd.concat(
        [
            bootstrap_summary(
                results[results["method"].isin(panel_methods(welfare))],  # pyright: ignore[reportArgumentType]
                [x_column, "method"],
                f"{welfare}_welfare",
            ).assign(welfare=welfare)
            for welfare in WELFARES
        ],
        ignore_index=True,
    )
    methods = present_methods(summary)
    frame = summary.assign(
        method_label=summary["method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType]
        welfare_label=summary["welfare"].map(WELFARE_LABELS),  # pyright: ignore[reportArgumentType]
    )
    x = (
        alt.X(f"{x_column}:Q", title=x_title)
        if x_sort is None
        else alt.X(f"{x_column}:N", title=x_title, sort=x_sort)
    )
    colour = alt.Color(
        "method_label:N", scale=method_colour_scale(methods), title="Method"
    )
    tooltip = [
        alt.Tooltip("method_label:N", title="Method"),
        alt.Tooltip(f"{x_column}:{'Q' if x_sort is None else 'N'}", title=x_title),
        alt.Tooltip("value:Q", title="Welfare", format=".4f"),
        alt.Tooltip("interval_low:Q", title="Interval low", format=".4f"),
        alt.Tooltip("interval_high:Q", title="Interval high", format=".4f"),
    ]
    intervals = (
        alt.Chart()
        .mark_rule()
        .encode(x=x, y=alt.Y("interval_low:Q"), y2="interval_high:Q", color=colour)
    )
    lines = (
        alt.Chart()
        .mark_line()
        .encode(
            x=x,
            y=alt.Y("value:Q", title="Welfare", scale=alt.Scale(zero=False)),
            color=colour,
        )
    )
    points = (
        alt.Chart()
        .mark_point(filled=True, size=60)
        .encode(x=x, y="value:Q", color=colour, tooltip=tooltip)
    )
    return (
        alt.layer(intervals, lines, points, data=frame)
        .properties(width=240, height=220)
        .facet(
            column=alt.Column(
                "welfare_label:N",
                sort=[WELFARE_LABELS[welfare] for welfare in WELFARES],
                title=None,
            )
        )
        .resolve_scale(y="independent")
    )


def sampling_chart(sampling: pd.DataFrame, candidate_count: int) -> alt.LayerChart:
    """Mean Blackwell regret of the deployed policy against the pool size, per target.

    The dashed line is the bound's sampling term, twice (1 minus 1 over the
    candidate count) to the power of the pool size.
    """
    summary = bootstrap_summary(sampling, ["pool_size", "target"], "regret")
    frame = summary.assign(
        target_label=summary["target"].map(TARGET_LABELS),  # pyright: ignore[reportArgumentType]
        interval_low=summary["interval_low"].clip(lower=1e-5),
    )
    pool_sizes = np.array(sorted(sampling["pool_size"].unique()))
    bound = pd.DataFrame(
        {"pool_size": pool_sizes, "value": 2 * (1 - 1 / candidate_count) ** pool_sizes}
    )
    x = alt.X("pool_size:Q", scale=alt.Scale(type="log", base=2), title="Pool size N")
    y = alt.Y("value:Q", scale=alt.Scale(type="log"), title="Blackwell regret")
    colour = alt.Color(
        "target_label:N", sort=list(TARGET_LABELS.values()), title="Target"
    )
    tooltip = [
        alt.Tooltip("target_label:N", title="Target"),
        alt.Tooltip("pool_size:Q", title="Pool size"),
        alt.Tooltip("value:Q", title="Regret", format=".4f"),
        alt.Tooltip("interval_low:Q", title="Interval low", format=".4f"),
        alt.Tooltip("interval_high:Q", title="Interval high", format=".4f"),
    ]
    bound_line = (
        alt.Chart(bound)
        .mark_line(color=NEUTRAL_COLOUR, strokeDash=[6, 4])
        .encode(x=x, y=y)
    )
    intervals = (
        alt.Chart(frame)
        .mark_rule()
        .encode(x=x, y=alt.Y("interval_low:Q"), y2="interval_high:Q", color=colour)
    )
    lines = alt.Chart(frame).mark_line().encode(x=x, y=y, color=colour)
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=70)
        .encode(x=x, y=y, color=colour, tooltip=tooltip)
    )
    return alt.layer(bound_line, intervals, lines, points).properties(  # pyright: ignore[reportReturnType]
        width=420, height=280, title="Sampling term, the gold selecting"
    )


def error_chart(sweep: pd.DataFrame) -> alt.LayerChart:
    """Each target method's regret on its own welfare against the selector's model error."""
    own_regrets = pd.concat(
        [
            sweep[sweep["method"] == method_name].assign(
                value=lambda frame, welfare=welfare: frame[f"{welfare}_regret"]
            )
            for method_name, welfare in METHOD_WELFARES.items()
        ]
    )
    summary = bootstrap_summary(own_regrets, ["factor", "method"], "value")
    errors = sweep.groupby("factor")["model_error"].mean()
    methods = present_methods(summary)
    frame = summary.assign(
        model_error=summary["factor"].map(errors),  # pyright: ignore[reportArgumentType]
        method_label=summary["method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType]
    )
    x = alt.X("model_error:Q", title="Model error, worst participant")
    colour = alt.Color(
        "method_label:N", scale=method_colour_scale(methods), title="Method"
    )
    tooltip = [
        alt.Tooltip("method_label:N", title="Method"),
        alt.Tooltip("factor:Q", title="Error factor"),
        alt.Tooltip("model_error:Q", title="Model error", format=".4f"),
        alt.Tooltip("value:Q", title="Regret", format=".4f"),
        alt.Tooltip("interval_low:Q", title="Interval low", format=".4f"),
        alt.Tooltip("interval_high:Q", title="Interval high", format=".4f"),
    ]
    intervals = (
        alt.Chart(frame)
        .mark_rule()
        .encode(x=x, y=alt.Y("interval_low:Q"), y2="interval_high:Q", color=colour)
    )
    lines = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=x,
            y=alt.Y("value:Q", title="Regret on the method's own welfare"),
            color=colour,
        )
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=70)
        .encode(x=x, y="value:Q", color=colour, tooltip=tooltip)
    )
    return alt.layer(intervals, lines, points).properties(  # pyright: ignore[reportReturnType]
        width=420, height=280, title="Model-error term, whole group"
    )


def budget_chart(sweep: pd.DataFrame) -> alt.LayerChart:
    """Mean Nash regret of the budgeted algorithm against the budget as a multiple of the true error."""
    summary = bootstrap_summary(sweep, ["budget_factor"], "nash_regret")
    x = alt.X(
        "budget_factor:Q", title="Error budget, as a multiple of the true model error"
    )
    tooltip = [
        alt.Tooltip("budget_factor:Q", title="Budget factor"),
        alt.Tooltip("value:Q", title="Nash regret", format=".4f"),
        alt.Tooltip("interval_low:Q", title="Interval low", format=".4f"),
        alt.Tooltip("interval_high:Q", title="Interval high", format=".4f"),
    ]
    intervals = (
        alt.Chart(summary)
        .mark_rule(color=METHOD_COLOURS[-1])
        .encode(x=x, y=alt.Y("interval_low:Q"), y2="interval_high:Q")
    )
    lines = (
        alt.Chart(summary)
        .mark_line(color=METHOD_COLOURS[-1])
        .encode(x=x, y=alt.Y("value:Q", title="Nash regret"))
    )
    points = (
        alt.Chart(summary)
        .mark_point(filled=True, size=70, color=METHOD_COLOURS[-1])
        .encode(x=x, y="value:Q", tooltip=tooltip)
    )
    return alt.layer(intervals, lines, points).properties(  # pyright: ignore[reportReturnType]
        width=420,
        height=280,
        title="Robust Nash with a misspecified budget, whole group",
    )


def explorer_controls(
    groups: pd.DataFrame, selector_names: list[str]
) -> tuple[mo.ui.dropdown, mo.ui.dropdown]:
    """Question and selector dropdowns for ``question_view``."""
    return (
        mo.ui.dropdown(
            options=dict(zip(groups["question"], groups["question_id"], strict=True)),
            value=groups["question"].iloc[0],
            label="Question",
            searchable=True,
        ),
        mo.ui.dropdown(
            options={MODEL_LABELS[name]: name for name in selector_names},
            value=MODEL_LABELS[selector_names[0]],
            label="Selector",
        ),
    )


def mixture_table(
    policies: dict[str, np.ndarray], pool_statements: pd.DataFrame
) -> pd.DataFrame:
    """Answers any method puts weight on, with each method's weight as a column.

    ``pool_statements`` are the generated candidates in sample order. Zero
    weights are NaN.
    """
    weights = {
        method_label: policies[method_name]
        for method_name, method_label in METHOD_LABELS.items()
        if method_name in policies
    }
    support = np.flatnonzero(np.any(np.stack(list(weights.values())) > 0, axis=0))
    return pd.DataFrame(
        {
            "answer": pool_statements["response"].iloc[support].tolist(),
            "originating_kind": [
                f"{kind} ({', '.join(map(str, targets))})" if len(targets) else kind
                for kind, targets in zip(
                    pool_statements["candidate_kind"].iloc[support],
                    pool_statements["target_participants"].iloc[support],
                    strict=True,
                )
            ],
        }
        | {
            method_label: [
                round(float(weight), 3) if weight > 0 else np.nan
                for weight in method_weights[support]
            ]
            for method_label, method_weights in weights.items()
        }
    )


def question_view(
    groups: pd.DataFrame,
    candidates: pd.DataFrame,
    policies: dict[str, np.ndarray],
    question_id: str,
) -> mo.Html:
    """The question and its four participants, then every method's mixture."""
    group_row = groups.set_index("question_id").loc[question_id]
    pool_statements = candidates.loc[
        lambda frame: frame["question_id"] == question_id
    ].sort_values("sample_index")
    weights_table = mixture_table(policies, pool_statements)
    return mo.vstack(
        [
            mo.md(
                f"## {group_row['question']}\n\n"
                + "\n\n".join(
                    f"**Participant {position + 1}:** {opinion}"
                    for position, opinion in enumerate(group_row["opinions"])
                )
            ),
            mo.ui.table(
                weights_table,
                selection=None,
                pagination=False,
                wrapped_columns=["answer"],
                format_mapping={
                    column: lambda weight: "" if pd.isna(weight) else f"{weight:.3f}"
                    for column in weights_table.columns[2:]
                },
            ),
        ]
    )
