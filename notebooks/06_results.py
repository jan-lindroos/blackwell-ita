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
    from blackwell_ita.artifacts import HEADLINE_METHODS, artifact_path

    return HEADLINE_METHODS, artifact_path


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

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
    # Categorical palette slots 1-8 in fixed order (see dataviz palette reference);
    # colour follows the method across every figure. The slot order is the
    # validated adjacency order, so new methods append rather than reorder
    palette = [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    ]
    method_colours = dict(zip(HEADLINE_METHODS, palette, strict=True))
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
    headline_methods = HEADLINE_METHODS
    return headline_methods, method_colours, method_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Headline: worst group and overall cost

    The Blackwell winner optimises the worst per-criterion (HelpSteer2 attributes)
    or per-group (Community Alignment country-age) win rate; the overall expected
    win rate (EWR = win + draw/2) against the reference anchor shows what that
    costs.
    """)
    return


@app.cell
def _(criterion_scores, headline_methods, judgements, pd):
    max_n = judgements["n"].max()
    headline_judgements = judgements[
        judgements["method"].isin(headline_methods)
        & ((judgements["n"] == max_n) | (judgements["method"] == "base"))
    ]
    per_criterion = criterion_scores.pivot_table(
        index="method", columns="criterion", values="score"
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
    ).reindex(headline_methods)
    headline_table.round(3)
    return (per_criterion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-criterion win rates at N = 64
    """)
    return


@app.cell
def _(headline_methods, method_colours, method_labels, per_criterion, plt):
    criterion_figure, criterion_axes = plt.subplots(figsize=(8, 3.5))
    criteria = list(per_criterion.columns)
    bar_width = 0.8 / len(headline_methods)
    for method_index, bar_method in enumerate(headline_methods):
        criterion_axes.bar(
            [
                position + method_index * bar_width
                for position in range(len(criteria))
            ],
            per_criterion.loc[bar_method, criteria],
            width=bar_width,
            color=method_colours[bar_method],
            edgecolor="white",
            linewidth=2,
            label=method_labels[bar_method],
        )
    criterion_axes.set_xticks(
        [position + 0.4 - bar_width / 2 for position in range(len(criteria))],
        criteria,
        rotation=20,
        ha="right",
    )
    criterion_axes.axhline(0.5, color="#c3c2b7", linewidth=1, zorder=0)
    criterion_axes.set_ylabel("Win rate vs anchor")
    criterion_axes.legend(frameon=False)
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
def _(judgements, method_colours, method_labels, plt):
    sweep_figure, sweep_axes = plt.subplots(figsize=(7, 3.5))
    n_sweep = judgements[
        (judgements["anchor"] == "reference") & (judgements["method"] != "base")
    ]
    for sweep_method, method_frame in n_sweep.groupby("method"):
        means = method_frame.groupby("n")["score"].mean()
        standard_errors = method_frame.groupby("n")["score"].sem()
        sweep_axes.plot(
            means.index,
            means.values,
            color=method_colours[sweep_method],
            linewidth=2,
            marker="o",
            markersize=6,
            label=method_labels[sweep_method],
        )
        sweep_axes.fill_between(
            means.index,
            means - standard_errors,
            means + standard_errors,
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
