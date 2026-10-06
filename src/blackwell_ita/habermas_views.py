import altair as alt
import marimo as mo
import numpy as np
import pandas as pd

from blackwell_ita.habermas import GENERATION_BACKBONES, selection_policies

MODEL_LABELS = {
    "qwen3_4b_pairwise": "Qwen PW",
    "qwen3_4b_bradley_terry": "Qwen BT",
    "gemma4_12b_pairwise_lora": "Gemma",
}
BACKBONE_LABELS = {
    backbone_name: backbone_name.split("/")[-1].split("-")[0]
    for backbone_name in GENERATION_BACKBONES
}
METHOD_LABELS = {
    "blackwell": "Blackwell (orthant)",
    "max_mean_win_rate": "Max mean win rate",
    "maximin": "Maximin",
    "best_of_n": "Best-of-N",
    "scalarised_nash": "Scalarised Nash",
    "blackwell_nash": "Blackwell (Nash target)",
}
WELFARE_LABELS = {
    "egalitarian_welfare": "Egalitarian",
    "nash_welfare": "Nash",
    "utilitarian_welfare": "Utilitarian",
}


def method_matrix_heatmap(
    summary: pd.DataFrame, centre: float, scale: float, title: str, legend_title: str
) -> alt.HConcatChart:
    """Methods by methods, one grid per welfare function, coloured around ``centre``.

    Cells show ``scale`` times the value. Stars mark a 95% interval that
    excludes ``centre``.
    """
    method_order = list(METHOD_LABELS.values())
    plotted = summary[
        summary["row_method"].isin(METHOD_LABELS)
        & summary["column_method"].isin(METHOD_LABELS)
    ]
    frame = plotted.assign(
        row_label=plotted["row_method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        column_label=plotted["column_method"].map(METHOD_LABELS),  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        welfare_label=plotted["welfare"].map(WELFARE_LABELS),  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        shown=scale * plotted["value"],
        significant=(plotted["interval_low"] > centre + 1e-9)
        | (plotted["interval_high"] < centre - 1e-9),
    )
    signed = centre == 0
    frame["cell_text"] = [
        f"{shown:{'+' if signed else ''}.2f}{'*' if significant else ''}"
        for shown, significant in zip(frame["shown"], frame["significant"], strict=True)
    ]
    shown_centre = scale * centre
    colour_limit = float((frame["shown"] - shown_centre).abs().max())  # pyright: ignore[reportArgumentType]
    x = alt.X(
        "column_label:N",
        sort=method_order,
        title=None,
        axis=alt.Axis(orient="top", labelAngle=-40),
    )

    def grid(welfare_label: str, show_rows: bool) -> alt.LayerChart:
        y = alt.Y(
            "row_label:N",
            sort=method_order,
            title=None,
            axis=alt.Axis() if show_rows else None,
        )
        cells = (
            alt.Chart()
            .mark_rect()
            .encode(
                x=x,
                y=y,
                color=alt.Color(
                    "shown:Q",
                    scale=alt.Scale(
                        domain=[
                            shown_centre - colour_limit,
                            shown_centre,
                            shown_centre + colour_limit,
                        ],
                        range=["#e34948", "#f0efec", "#2a78d6"],
                        interpolate="lab",
                    ),
                    title=legend_title,
                ),
                tooltip=[
                    alt.Tooltip("row_label:N", title="Row"),
                    alt.Tooltip("column_label:N", title="Column"),
                    alt.Tooltip("welfare_label:N", title="Welfare"),
                    alt.Tooltip("shown:Q", title=legend_title, format=".3f"),
                    alt.Tooltip("instances:Q", title="Instances"),
                ],
            )
        )
        text = (
            alt.Chart()
            .mark_text(fontSize=10)
            .encode(
                x=x,
                y=y,
                text="cell_text:N",
                color=alt.condition(
                    f"abs(datum.shown - {shown_centre}) > {0.6 * colour_limit}",
                    alt.value("#ffffff"),
                    alt.value("#1a1a19"),
                ),
            )
        )
        return alt.layer(  # pyright: ignore[reportReturnType]
            cells,
            text,
            data=frame.loc[
                frame["welfare_label"] == welfare_label,
                [
                    "row_label",
                    "column_label",
                    "welfare_label",
                    "shown",
                    "cell_text",
                    "instances",
                ],
            ],
        ).properties(title=welfare_label, width=alt.Step(40), height=alt.Step(26))

    return alt.hconcat(
        *[
            grid(welfare_label, show_rows=position == 0)
            for position, welfare_label in enumerate(WELFARE_LABELS.values())
        ],
        title=title,
    )


def explorer_controls(
    groups: pd.DataFrame, tensors_by_backbone: dict[str, dict[str, dict]]
) -> tuple[mo.ui.dropdown, mo.ui.dropdown, mo.ui.dropdown]:
    """Backbone, question and selector dropdowns for ``question_view``."""
    model_names = list(next(iter(tensors_by_backbone.values())))
    return (
        mo.ui.dropdown(
            options={
                BACKBONE_LABELS[backbone_name]: backbone_name
                for backbone_name in tensors_by_backbone
            },
            value=BACKBONE_LABELS[next(iter(tensors_by_backbone))],
            label="Backbone",
        ),
        mo.ui.dropdown(
            options=dict(zip(groups["question"], groups["question_id"], strict=True)),
            value=groups["question"].iloc[0],
            label="Question",
            searchable=True,
        ),
        mo.ui.dropdown(
            options={
                MODEL_LABELS[model_name]: model_name for model_name in model_names
            },
            value=MODEL_LABELS[model_names[0]],
            label="Selector",
        ),
    )


def mixture_table(
    policies: dict[str, np.ndarray], pool_statements: pd.DataFrame
) -> pd.DataFrame:
    """Answers any method puts weight on, with each method's weight as a column.

    ``pool_statements`` are the generated candidates in sample order, without
    the anchor that ends every policy. Zero weights are NaN.
    """
    weights = {
        method_label: policies[method_name][: len(pool_statements)]
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
    candidates_by_backbone: dict[str, pd.DataFrame],
    tensors_by_backbone: dict[str, dict[str, dict]],
    backbone_name: str,
    question_id: str,
    selector_name: str,
) -> mo.Html:
    """The question and its four participants, then every method's mixture."""
    selector_tensor = tensors_by_backbone[backbone_name][selector_name].get(question_id)
    if selector_tensor is None:
        return mo.md("Not scored yet for this selector.")
    group_row = groups.set_index("question_id").loc[question_id]
    pool_statements = (
        candidates_by_backbone[backbone_name]
        .loc[lambda candidates: candidates["question_id"] == question_id]
        .sort_values("sample_index")
    )
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
