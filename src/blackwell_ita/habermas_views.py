import altair as alt
import marimo as mo
import numpy as np
import pandas as pd

from blackwell_ita.habermas import bootstrap_summary, selection_policies

MODEL_LABELS = {
    "qwen3_4b_pairwise": "Qwen PW",
    "qwen3_4b_bradley_terry": "Qwen BT",
}
METHOD_LABELS = {
    "uniform": "Uniform pool",
    "max_mean_win_rate": "Max mean win rate",
    "best_of_n": "Best-of-N",
    "maximin": "Maximin",
    "scalarised_nash": "Scalarised Nash",
    "blackwell": "BoB (orthant)",
    "blackwell_nash": "BoB (Nash target)",
    "reference_nash": "Reference BoB (Nash)",
}
# Fixed categorical order, so each method keeps its colour in every chart
METHOD_COLOURS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
NASH_METHODS = ["reference_nash", "best_of_n", "maximin", "scalarised_nash"]
AXIS_TITLES = {
    "average": "Average win rate",
    "best": "Best-served win rate",
    "worst": "Worst-served win rate",
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


def frontier_chart(
    frontier: pd.DataFrame, welfare: pd.DataFrame, axis: str, title: str
) -> alt.FacetChart:
    """The gold frontier in grey with each method's mean point, one panel per selector.

    The uniform pool sits at an average of 1/2, far from the frontier, so it is left out.

    ``axis`` is "average" or "best" and is plotted against the worst-served
    participant's win rate.
    """
    plotted_methods = [
        method_name
        for method_name in METHOD_LABELS
        if method_name != "uniform" and method_name in set(welfare["method"])
    ]
    method_points = (
        welfare[welfare["method"].isin(plotted_methods)]
        .groupby(["selector", "method"], sort=False)[[axis, "worst"]]
        .mean()
        .reset_index()  # pyright: ignore[reportAttributeAccessIssue]
        .assign(kind="method")
    )
    frontier_lines = pd.concat(
        [
            frontier.assign(selector=selector_name, kind="frontier")
            for selector_name in method_points["selector"].unique()
        ],
        ignore_index=True,
    )
    frame = pd.concat([frontier_lines, method_points], ignore_index=True)
    frame = frame.assign(
        selector_label=frame["selector"].map(MODEL_LABELS),  # pyright: ignore[reportArgumentType]
        method_label=frame["method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType]
    )
    x = alt.X(f"{axis}:Q", scale=alt.Scale(zero=False), title=AXIS_TITLES[axis])
    y = alt.Y("worst:Q", scale=alt.Scale(zero=False), title=AXIS_TITLES["worst"])
    line = (
        alt.Chart()
        .transform_filter(alt.datum.kind == "frontier")
        .mark_line(color=NEUTRAL_COLOUR, strokeWidth=2)
        .encode(x=x, y=y, order="worst_weight:Q")
    )
    points = (
        alt.Chart()
        .transform_filter(alt.datum.kind == "method")
        .mark_point(filled=True, size=90, opacity=1)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "method_label:N",
                scale=method_colour_scale(plotted_methods),
                title="Method",
            ),
            shape=alt.Shape(
                "method_label:N",
                scale=alt.Scale(
                    domain=[
                        METHOD_LABELS[method_name] for method_name in plotted_methods
                    ]
                ),
                title="Method",
            ),
            tooltip=[
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip(f"{axis}:Q", title=AXIS_TITLES[axis], format=".3f"),
                alt.Tooltip("worst:Q", title=AXIS_TITLES["worst"], format=".3f"),
            ],
        )
    )
    return (
        alt.layer(line, points, data=frame)
        .properties(width=320, height=280)
        .facet(column=alt.Column("selector_label:N", title=None))
        .properties(title=title)
    )


def nash_chart(welfare: pd.DataFrame) -> alt.FacetChart:
    """Mean Nash welfare with a 95% bootstrap interval, one row per selector."""
    summary = bootstrap_summary(
        welfare[welfare["method"].isin(NASH_METHODS)].rename(columns={"nash": "value"}),  # pyright: ignore[reportCallIssue]
        ["selector", "method"],
    )
    frame = summary.assign(
        selector_label=summary["selector"].map(MODEL_LABELS),  # pyright: ignore[reportArgumentType]
        method_label=summary["method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType]
    )
    y = alt.Y(
        "method_label:N",
        sort=[METHOD_LABELS[method_name] for method_name in NASH_METHODS],
        title=None,
    )
    intervals = (
        alt.Chart()
        .mark_rule(color=METHOD_COLOURS[0], strokeWidth=2)
        .encode(
            x=alt.X(
                "interval_low:Q", scale=alt.Scale(zero=False), title="Nash welfare"
            ),
            x2="interval_high:Q",
            y=y,
        )
    )
    means = (
        alt.Chart()
        .mark_point(filled=True, size=90, opacity=1, color=METHOD_COLOURS[0])
        .encode(
            x="value:Q",
            y=y,
            tooltip=[
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("value:Q", title="Nash welfare", format=".4f"),
                alt.Tooltip("interval_low:Q", title="Interval low", format=".4f"),
                alt.Tooltip("interval_high:Q", title="Interval high", format=".4f"),
                alt.Tooltip("instances:Q", title="Instances"),
            ],
        )
    )
    return (
        alt.layer(intervals, means, data=frame)
        .properties(width=360)
        .facet(row=alt.Row("selector_label:N", title=None))
        .properties(title="Nash welfare against the pool, whole group")
    )


def winner_chart(welfare: pd.DataFrame) -> alt.FacetChart:
    """Win rate against the round's winner with a 95% bootstrap interval.

    One column for the average participant and one for the worst served, one
    row per selector. The uniform pool is the baseline of a random candidate.
    """
    long_welfare = welfare.melt(
        id_vars=["selector", "method", "question_id"],
        value_vars=["winner_average", "winner_worst"],
        var_name="served",
    ).dropna(subset=["value"])
    summary = bootstrap_summary(long_welfare, ["selector", "method", "served"])
    frame = summary.assign(
        selector_label=summary["selector"].map(MODEL_LABELS),  # pyright: ignore[reportArgumentType]
        method_label=summary["method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType]
        served_label=summary["served"].map(  # pyright: ignore[reportArgumentType, reportCallIssue]
            {"winner_average": "Average", "winner_worst": "Worst served"}  # pyright: ignore[reportArgumentType]
        ),
    )
    y = alt.Y("method_label:N", sort=list(METHOD_LABELS.values()), title=None)
    x = alt.X(
        "interval_low:Q",
        scale=alt.Scale(zero=False),
        title="Win rate against the Habermas winner",
    )
    intervals = (
        alt.Chart()
        .mark_rule(color=METHOD_COLOURS[0], strokeWidth=2)
        .encode(x=x, x2="interval_high:Q", y=y)
    )
    means = (
        alt.Chart()
        .mark_point(filled=True, size=90, opacity=1, color=METHOD_COLOURS[0])
        .encode(
            x="value:Q",
            y=y,
            tooltip=[
                alt.Tooltip("method_label:N", title="Method"),
                alt.Tooltip("served_label:N", title="Participant"),
                alt.Tooltip("value:Q", title="Win rate", format=".3f"),
                alt.Tooltip("interval_low:Q", title="Interval low", format=".3f"),
                alt.Tooltip("interval_high:Q", title="Interval high", format=".3f"),
                alt.Tooltip("instances:Q", title="Instances"),
            ],
        )
    )
    return (
        alt.layer(intervals, means, data=frame)
        .properties(width=280)
        .facet(
            row=alt.Row("selector_label:N", title=None),
            column=alt.Column("served_label:N", title=None),
        )
        .properties(title="Against the deliberation's final statement, whole group")
    )


def beaten_histogram(results: pd.DataFrame, pool_size: int) -> alt.HConcatChart:
    """How many pool candidates each participant prefers less than the mixture.

    One column of panels per selector, since Vega-Lite misplaces sorted facet
    rows when a selector lacks a method.
    """
    beaten_bin = alt.Bin(extent=[0, pool_size + 1], step=2)

    def selector_column(selector_name: str) -> alt.FacetChart:
        selector_results = results[results["selector"] == selector_name]
        frame = selector_results.assign(
            method_label=selector_results["method"].map(METHOD_LABELS)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        )
        return (
            alt.Chart(frame[["method_label", "candidates_beaten"]])
            .mark_bar(color=METHOD_COLOURS[0], cornerRadiusEnd=4)
            .encode(
                x=alt.X(
                    "candidates_beaten:Q",
                    bin=beaten_bin,
                    title=f"Candidates beaten, out of {pool_size}",
                ),
                y=alt.Y("count():Q", title="Participants"),
                tooltip=[
                    alt.Tooltip("method_label:N", title="Method"),
                    alt.Tooltip("candidates_beaten:Q", bin=beaten_bin, title="Beaten"),
                    alt.Tooltip("count():Q", title="Participants"),
                ],
            )
            .properties(width=220, height=90)
            .facet(
                row=alt.Row(
                    "method_label:N",
                    sort=[
                        method_label
                        for method_name, method_label in METHOD_LABELS.items()
                        if method_name in set(selector_results["method"])
                    ],
                    title=None,
                ),
            )
            .properties(title=MODEL_LABELS[selector_name])
        )

    return alt.hconcat(
        *[
            selector_column(selector_name)
            for selector_name in results["selector"].unique()
        ],
        title="Candidates each participant prefers less than the mixture",
    )


def explorer_controls(
    groups: pd.DataFrame, tensors_by_selector: dict[str, dict]
) -> tuple[mo.ui.dropdown, mo.ui.dropdown]:
    """Question and selector dropdowns for ``question_view``."""
    selector_names = list(tensors_by_selector)
    return (
        mo.ui.dropdown(
            options=dict(zip(groups["question"], groups["question_id"], strict=True)),
            value=groups["question"].iloc[0],
            label="Question",
            searchable=True,
        ),
        mo.ui.dropdown(
            options={
                MODEL_LABELS[selector_name]: selector_name
                for selector_name in selector_names
            },
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
    tensors_by_selector: dict[str, dict],
    question_id: str,
    selector_name: str,
) -> mo.Html:
    """The question and its four participants, then every method's mixture."""
    selector_tensor = tensors_by_selector[selector_name].get(question_id)
    if selector_tensor is None:
        return mo.md("Not scored yet for this selector.")
    group_row = groups.set_index("question_id").loc[question_id]
    pool_statements = candidates.loc[
        lambda frame: frame["question_id"] == question_id
    ].sort_values("sample_index")
    weights_table = mixture_table(
        selection_policies(selector_tensor, selector_name), pool_statements
    )
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
