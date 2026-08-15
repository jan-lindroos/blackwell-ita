# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita",
#     "huggingface_hub",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from blackwell_ita.utils.artifacts import HEADLINE_METHODS, artifact_path

    return HEADLINE_METHODS, artifact_path


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    return pd, plt


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=["helpsteer2", "community-alignment"],
        value="helpsteer2",
        label="Dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _(artifact_path, dataset_picker, pd):
    judgements = pd.read_parquet(
        artifact_path(dataset_picker.value, "judgements.parquet")
    )
    criterion_scores = pd.read_parquet(
        artifact_path(dataset_picker.value, "criterion_scores.parquet")
    )
    return criterion_scores, judgements


@app.cell
def _(HEADLINE_METHODS):
    # Colour follows the method across every figure. The slot order is fixed,
    # so new methods append rather than reorder
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    method_colours = dict(zip(HEADLINE_METHODS, palette, strict=True))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    method_markers = dict(zip(HEADLINE_METHODS, markers, strict=True))
    method_labels = {
        "base": "Base policy",
        "bt_best_of_n": "BT best-of-N",
        "best_of_nash": "Best-of-Nash",
        "blackwell": "Blackwell",
        "mean_criterion_best_of_n": "Mean-criterion best-of-N",
        "worst_criterion_best_of_n": "Worst-criterion best-of-N",
        "bt_mean_criterion_best_of_n": "BT mean-criterion best-of-N",
        "bt_worst_criterion_best_of_n": "BT worst-criterion best-of-N",
    }
    return method_colours, method_labels, method_markers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Headline: worst group and overall cost

    The Blackwell winner optimises the worst per-criterion (HelpSteer2
    attributes) or per-group (Community Alignment country and age
    subpopulations) win rate.
    The overall expected win rate (EWR = win + draw/2) against the reference
    anchor shows what that costs.
    """)
    return


@app.cell
def _(HEADLINE_METHODS, criterion_scores, judgements, pd):
    max_n = judgements["n"].max()
    headline_judgements = judgements[
        judgements["method"].isin(HEADLINE_METHODS)
        & ((judgements["n"] == max_n) | (judgements["method"] == "base"))
    ]
    per_criterion = criterion_scores.pivot_table(
        index="method", columns="criterion", values="score", aggfunc="mean"
    )
    headline_table = pd.DataFrame(
        {
            "worst criterion": per_criterion.min(axis=1),
            "EWR vs reference": headline_judgements[
                headline_judgements["anchor"] == "reference"
            ]
            .groupby("method")["score"]
            .mean(),
        }
    ).reindex(HEADLINE_METHODS)
    headline_table.round(3)
    return (per_criterion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-criterion win rates at N = 64
    """)
    return


@app.cell
def _(HEADLINE_METHODS, method_colours, method_labels, per_criterion, plt):
    criterion_figure, criterion_axes = plt.subplots(figsize=(6.5, 3))
    criteria = list(per_criterion.columns)
    bar_width = 0.8 / len(HEADLINE_METHODS)
    for method_index, bar_method in enumerate(HEADLINE_METHODS):
        criterion_axes.bar(
            [
                position + method_index * bar_width
                for position in range(len(criteria))
            ],
            per_criterion.loc[bar_method, criteria],
            width=bar_width,
            color=method_colours[bar_method],
            edgecolor="black",
            linewidth=0.4,
            label=method_labels[bar_method],
        )
    criterion_axes.set_xticks(
        [position + 0.4 - bar_width / 2 for position in range(len(criteria))],
        criteria,
        rotation=20,
        ha="right",
    )
    criterion_axes.axhline(
        0.5, color="grey", linewidth=0.8, linestyle="--", zorder=0
    )
    criterion_axes.set_ylabel("Win rate vs anchor")
    criterion_axes.legend(frameon=False, ncols=2)
    criterion_axes.spines[["top", "right"]].set_visible(False)
    criterion_figure.tight_layout()
    criterion_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overall EWR against the reference anchor as N grows
    """)
    return


@app.cell
def _(
    HEADLINE_METHODS, judgements, method_colours, method_labels, method_markers, plt
):
    sweep_figure, sweep_axes = plt.subplots(figsize=(5.5, 3))
    n_sweep = judgements[
        (judgements["anchor"] == "reference") & (judgements["method"] != "base")
    ]
    for sweep_method in HEADLINE_METHODS:
        method_frame = n_sweep[n_sweep["method"] == sweep_method]
        if method_frame.empty:
            continue
        stats = method_frame.groupby("n")["score"].agg(["mean", "sem"])
        sweep_axes.plot(
            stats.index,
            stats["mean"],
            color=method_colours[sweep_method],
            linewidth=1.2,
            marker=method_markers[sweep_method],
            markersize=4,
            label=method_labels[sweep_method],
        )
        sweep_axes.fill_between(
            stats.index,
            stats["mean"] - stats["sem"],
            stats["mean"] + stats["sem"],
            color=method_colours[sweep_method],
            alpha=0.15,
            linewidth=0,
        )
    sweep_axes.set_xscale("log", base=2)
    sweep_axes.set_xlabel("N (number of samples)")
    sweep_axes.set_ylabel("EWR vs reference")
    sweep_axes.legend(frameon=False)
    sweep_axes.spines[["top", "right"]].set_visible(False)
    sweep_figure.tight_layout()
    sweep_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Future directions

    - Systematic target sets vs best-linear-scalarisation

    - Blackwell with overall as criterion

    - Uncertainty quantification

    - DPO baseline
    """)
    return


if __name__ == "__main__":
    app.run()
