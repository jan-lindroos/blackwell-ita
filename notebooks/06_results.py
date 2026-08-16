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
    import numpy as np
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
    return np, pd, plt


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
def _():
    # Colour follows the objective across every figure; hollow marks and
    # dashed lines mark the Bradley-Terry judge, filled and solid the
    # pairwise judge
    objective_colours = {
        "overall": "#2a78d6",
        "mean": "#eb6834",
        "worst": "#1baf7a",
        "blackwell": "#4a3aa7",
    }
    objective_labels = {
        "overall": "Overall",
        "mean": "Mean criterion",
        "worst": "Worst criterion",
        "blackwell": "Blackwell",
    }
    base_colour = "#7f7f7f"
    bt_methods = {
        "bt_best_of_n",
        "bt_mean_criterion_best_of_n",
        "bt_worst_criterion_best_of_n",
    }
    method_colours = {
        "base": base_colour,
        "bt_best_of_n": objective_colours["overall"],
        "best_of_nash": objective_colours["overall"],
        "mean_criterion_best_of_n": objective_colours["mean"],
        "bt_mean_criterion_best_of_n": objective_colours["mean"],
        "worst_criterion_best_of_n": objective_colours["worst"],
        "bt_worst_criterion_best_of_n": objective_colours["worst"],
        "blackwell": objective_colours["blackwell"],
    }
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
    short_labels = {
        "base": "Base",
        "bt_best_of_n": "BoN (BT)",
        "best_of_nash": "Best-of-Nash",
        "blackwell": "Blackwell",
        "mean_criterion_best_of_n": "Mean-crit",
        "worst_criterion_best_of_n": "Worst-crit",
        "bt_mean_criterion_best_of_n": "Mean-crit (BT)",
        "bt_worst_criterion_best_of_n": "Worst-crit (BT)",
    }
    return (
        base_colour,
        bt_methods,
        method_colours,
        method_labels,
        objective_colours,
        objective_labels,
        short_labels,
    )


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
def _(bt_methods, headline_table, method_colours, plt, short_labels):
    scatter_figure, scatter_axes = plt.subplots(figsize=(5, 3.2))
    scatter_table = headline_table.dropna()
    for scatter_method, scatter_row in scatter_table.iterrows():
        scatter_colour = method_colours[scatter_method]
        scatter_axes.plot(
            scatter_row["EWR vs reference"],
            scatter_row["worst criterion"],
            marker="o",
            markersize=7,
            color=scatter_colour,
            markerfacecolor="white"
            if scatter_method in bt_methods
            else scatter_colour,
            markeredgecolor=scatter_colour
            if scatter_method in bt_methods
            else "white",
        )
    scatter_axes.axhline(
        0.5, color="grey", linewidth=0.8, linestyle="--", zorder=0
    )
    scatter_axes.axvline(
        0.5, color="grey", linewidth=0.8, linestyle="--", zorder=0
    )
    scatter_axes.margins(x=0.25, y=0.3)
    scatter_axes.set_xlabel("EWR vs reference")
    scatter_axes.set_ylabel("Worst criterion win rate")
    scatter_axes.spines[["top", "right"]].set_visible(False)
    scatter_figure.tight_layout()
    # Labels take greedy vertical slots in display space so clustered
    # points stay individually readable
    scatter_figure.canvas.draw()
    scatter_mid = sum(scatter_axes.get_xlim()) / 2
    scatter_pixels = {
        pixel_method: scatter_axes.transData.transform(
            (pixel_row["EWR vs reference"], pixel_row["worst criterion"])
        )
        for pixel_method, pixel_row in scatter_table.iterrows()
    }
    scatter_slots = []
    for scatter_method in scatter_table["worst criterion"].sort_values().index:
        scatter_point = scatter_table.loc[scatter_method]
        scatter_x_px, scatter_y_px = scatter_pixels[scatter_method]
        scatter_slot = scatter_y_px
        while any(abs(scatter_slot - used) < 12 for used in scatter_slots):
            scatter_slot += 13
        scatter_slots.append(scatter_slot)
        # Flip the label to the other side when another point sits where
        # the text would go
        scatter_blocked = {
            side: any(
                abs(other_y - scatter_slot) < 9
                and (
                    -80 < other_x - scatter_x_px < 0
                    if side == "left"
                    else 0 < other_x - scatter_x_px < 80
                )
                for other_method, (other_x, other_y) in scatter_pixels.items()
                if other_method != scatter_method
            )
            for side in ("left", "right")
        }
        scatter_left = scatter_point["EWR vs reference"] > scatter_mid
        if (
            scatter_blocked["left" if scatter_left else "right"]
            and not scatter_blocked["right" if scatter_left else "left"]
        ):
            scatter_left = not scatter_left
        scatter_lifted = scatter_slot != scatter_y_px
        scatter_axes.annotate(
            short_labels[scatter_method],
            (
                scatter_point["EWR vs reference"],
                scatter_point["worst criterion"],
            ),
            xytext=(
                (-18 if scatter_left else 18) if scatter_lifted else
                (-7 if scatter_left else 7),
                (scatter_slot - scatter_y_px) * 72 / scatter_figure.dpi,
            ),
            textcoords="offset points",
            ha="right" if scatter_left else "left",
            va="center",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "color": "#b0b0b0",
                "linewidth": 0.6,
                "shrinkA": 2,
                "shrinkB": 4,
            }
            if scatter_lifted
            else None,
        )
    scatter_figure
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
    ).reindex(HEADLINE_METHODS)
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
    return headline_table, per_criterion


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-criterion win rates at N = 64
    """)
    return


@app.cell
def _(bt_methods, np, per_criterion, plt, short_labels):
    heatmap_order = [
        "base",
        "best_of_nash",
        "mean_criterion_best_of_n",
        "worst_criterion_best_of_n",
        "blackwell",
        "bt_best_of_n",
        "bt_mean_criterion_best_of_n",
        "bt_worst_criterion_best_of_n",
    ]
    heatmap_scores = per_criterion.reindex(heatmap_order).dropna(how="all")
    heatmap_methods = list(heatmap_scores.index)
    heatmap_values = heatmap_scores.to_numpy()

    heatmap_figure, heatmap_axes = plt.subplots(figsize=(6.5, 3.6))
    heatmap_image = heatmap_axes.imshow(
        heatmap_values, cmap="Blues", aspect="auto"
    )
    # Outline each row's minimum, the quantity Blackwell optimises
    for heatmap_row in range(len(heatmap_methods)):
        for heatmap_column in range(heatmap_values.shape[1]):
            heatmap_value = heatmap_values[heatmap_row, heatmap_column]
            heatmap_axes.text(
                heatmap_column,
                heatmap_row,
                f"{heatmap_value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white"
                if heatmap_image.norm(heatmap_value) > 0.6
                else "black",
            )
        heatmap_axes.add_patch(
            plt.Rectangle(
                (int(np.nanargmin(heatmap_values[heatmap_row])) - 0.5,
                 heatmap_row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
            )
        )
    heatmap_bt_start = next(
        (
            index
            for index, method in enumerate(heatmap_methods)
            if method in bt_methods
        ),
        len(heatmap_methods),
    )
    for heatmap_boundary in (0.5, heatmap_bt_start - 0.5):
        heatmap_axes.axhline(heatmap_boundary, color="white", linewidth=2)
    heatmap_axes.set_xticks(
        range(heatmap_values.shape[1]),
        list(heatmap_scores.columns),
        rotation=20,
        ha="right",
    )
    heatmap_axes.set_yticks(
        range(len(heatmap_methods)),
        [short_labels[method] for method in heatmap_methods],
    )
    heatmap_axes.spines[:].set_visible(False)
    heatmap_axes.tick_params(length=0)
    heatmap_figure.colorbar(
        heatmap_image, ax=heatmap_axes, shrink=0.8, label="Win rate vs anchor"
    )
    heatmap_figure.tight_layout()
    heatmap_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overall EWR against the reference anchor as N grows
    """)
    return


@app.cell
def _(
    base_colour,
    judgements,
    method_colours,
    objective_colours,
    objective_labels,
    plt,
):
    sweep_figure, (sweep_left_axes, sweep_right_axes) = plt.subplots(
        1, 2, figsize=(7, 3.2), sharey=True, layout="constrained"
    )
    n_sweep = judgements[judgements["anchor"] == "reference"]
    base_ewr = n_sweep[n_sweep["method"] == "base"]["score"].mean()
    sweep_panels = (
        (
            sweep_left_axes,
            "Pairwise judge",
            "-",
            [
                "best_of_nash",
                "mean_criterion_best_of_n",
                "worst_criterion_best_of_n",
                "blackwell",
            ],
        ),
        (
            sweep_right_axes,
            "Bradley-Terry judge",
            "--",
            [
                "bt_best_of_n",
                "bt_mean_criterion_best_of_n",
                "bt_worst_criterion_best_of_n",
            ],
        ),
    )
    for sweep_axes, sweep_title, sweep_style, sweep_methods in sweep_panels:
        sweep_axes.axhline(
            base_ewr, color=base_colour, linewidth=1, linestyle=":"
        )
        for sweep_method in sweep_methods:
            stats = (
                n_sweep[n_sweep["method"] == sweep_method]
                .groupby("n")["score"]
                .mean()
            )
            if stats.empty:
                continue
            sweep_axes.plot(
                stats.index,
                stats.to_numpy(),
                color=method_colours[sweep_method],
                linewidth=1.2,
                linestyle=sweep_style,
                marker="o",
                markersize=4,
                markerfacecolor="white"
                if sweep_style == "--"
                else method_colours[sweep_method],
            )
        sweep_axes.set_xscale("log", base=2)
        sweep_axes.set_xlabel("N (number of samples)")
        sweep_axes.set_title(sweep_title, fontsize=9)
        sweep_axes.spines[["top", "right"]].set_visible(False)
    sweep_left_axes.set_ylabel("EWR vs reference")
    sweep_handles = [
        plt.Line2D(
            [],
            [],
            color=objective_colours[objective],
            linewidth=1.2,
            label=objective_labels[objective],
        )
        for objective in objective_colours
    ] + [
        plt.Line2D(
            [],
            [],
            color=base_colour,
            linewidth=1,
            linestyle=":",
            label="Base policy",
        )
    ]
    sweep_figure.legend(
        handles=sweep_handles,
        loc="outside lower center",
        ncols=5,
        frameon=False,
    )
    sweep_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Head-to-head paired differences

    Each cell is the mean per-prompt EWR gap (row minus column) at the
    chosen N. Prompt difficulty cancels in the subtraction, so the
    bootstrap interval is driven only by prompts where the two policies
    differ; an asterisk marks gaps whose 95% interval excludes zero.
    """)
    return


@app.cell
def _(judgements, mo):
    head_to_head_n_picker = mo.ui.dropdown(
        options={str(n): int(n) for n in sorted(judgements["n"].unique())},
        value=str(int(judgements["n"].max())),
        label="N",
    )
    head_to_head_n_picker
    return (head_to_head_n_picker,)


@app.cell
def _(
    bt_methods,
    head_to_head_n_picker,
    judgements,
    np,
    plt,
    short_labels,
):
    paired_wide = (
        judgements[
            (judgements["anchor"] == "reference")
            & (judgements["n"] == head_to_head_n_picker.value)
        ]
        .pivot_table(index="prompt", columns="method", values="score")
        .dropna()
    )
    # Pairwise-judge block first, then the Bradley-Terry block, so the
    # judge-family comparisons sit in contiguous sub-blocks
    grid_order = [
        "base",
        "best_of_nash",
        "mean_criterion_best_of_n",
        "worst_criterion_best_of_n",
        "blackwell",
        "bt_best_of_n",
        "bt_mean_criterion_best_of_n",
        "bt_worst_criterion_best_of_n",
    ]
    grid_methods = [
        method for method in grid_order if method in paired_wide.columns
    ]
    paired_scores = paired_wide[grid_methods].to_numpy()
    prompt_count = len(paired_scores)
    paired_diffs = paired_scores[:, :, None] - paired_scores[:, None, :]
    mean_gaps = paired_diffs.mean(axis=0)
    # Multinomial weights are the prompt bootstrap without materialising the
    # (resample, prompt, method, method) array
    boot_counts = np.random.default_rng(0).multinomial(
        prompt_count, np.full(prompt_count, 1 / prompt_count), size=2000
    )
    boot_gaps = (
        boot_counts @ paired_diffs.reshape(prompt_count, -1)
    ) / prompt_count
    gap_low, gap_high = np.percentile(boot_gaps, [2.5, 97.5], axis=0)
    significant = ((gap_low > 0.0) | (gap_high < 0.0)).reshape(mean_gaps.shape)

    grid_figure, grid_axes = plt.subplots(figsize=(7.2, 5))
    gap_limit = max(np.abs(mean_gaps).max(), 1e-3)
    shown_gaps = mean_gaps.copy()
    np.fill_diagonal(shown_gaps, np.nan)
    grid_cmap = plt.get_cmap("RdBu_r").copy()
    grid_cmap.set_bad("#f0f0f0")
    grid_image = grid_axes.imshow(
        shown_gaps, cmap=grid_cmap, vmin=-gap_limit, vmax=gap_limit
    )
    for grid_row in range(len(grid_methods)):
        for grid_column in range(len(grid_methods)):
            if grid_row == grid_column:
                continue
            gap = mean_gaps[grid_row, grid_column]
            grid_axes.text(
                grid_column,
                grid_row,
                f"{gap:+.2f}" + ("*" if significant[grid_row, grid_column] else ""),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(gap) > 0.6 * gap_limit else "black",
            )
    grid_bt_start = next(
        (
            index
            for index, method in enumerate(grid_methods)
            if method in bt_methods
        ),
        len(grid_methods),
    )
    if 0 < grid_bt_start < len(grid_methods):
        grid_axes.axhline(grid_bt_start - 0.5, color="white", linewidth=2)
        grid_axes.axvline(grid_bt_start - 0.5, color="white", linewidth=2)
    grid_axes.set_xticks(
        range(len(grid_methods)),
        [short_labels[method] for method in grid_methods],
        rotation=30,
        ha="right",
    )
    grid_axes.set_yticks(
        range(len(grid_methods)),
        [short_labels[method] for method in grid_methods],
    )
    grid_axes.spines[:].set_visible(False)
    grid_axes.tick_params(length=0)
    grid_figure.colorbar(
        grid_image, ax=grid_axes, shrink=0.7, label="Row EWR - column EWR"
    )
    grid_figure.tight_layout()
    grid_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sweeping the overall threshold

    The overall preference joins the per-criterion ones as an extra
    criterion with its own threshold tau, swept from 0.50 to 0.95 at the
    largest N. The grey segment joins plain Blackwell to Best-of-Nash and
    bounds every random mixture of the two. The companion panel reads the
    same sweep against tau, with dashed lines at the endpoint methods'
    levels.
    """)
    return


@app.cell
def _(criterion_scores, judgements, mo):
    frontier_ewr = (
        judgements[
            (judgements["anchor"] == "reference")
            & (judgements["n"] == judgements["n"].max())
        ]
        .groupby("method")["score"]
        .mean()
    )
    frontier_worst = criterion_scores.pivot_table(
        index="method", columns="criterion", values="score", aggfunc="mean"
    ).min(axis=1)
    frontier_methods = [
        method
        for method in sorted(frontier_ewr.index)
        if method.startswith("blackwell_tau_") and method in frontier_worst.index
    ]
    mo.stop(not frontier_methods, mo.md("No tau sweep results yet."))
    frontier_taus = [
        int(method.removeprefix("blackwell_tau_")) / 100
        for method in frontier_methods
    ]
    return frontier_ewr, frontier_methods, frontier_taus, frontier_worst


@app.cell
def _(
    frontier_ewr,
    frontier_methods,
    frontier_taus,
    frontier_worst,
    method_colours,
    plt,
    short_labels,
):
    frontier_figure, frontier_axes = plt.subplots(figsize=(5, 3.2))
    frontier_axes.plot(
        [frontier_worst[method] for method in ("blackwell", "best_of_nash")],
        [frontier_ewr[method] for method in ("blackwell", "best_of_nash")],
        color="grey",
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )
    frontier_axes.plot(
        [frontier_worst[method] for method in frontier_methods],
        [frontier_ewr[method] for method in frontier_methods],
        color=method_colours["blackwell"],
        linewidth=1.2,
        marker="o",
        markersize=4,
    )
    for frontier_index, frontier_method in enumerate(frontier_methods):
        if frontier_index % 2 and frontier_index != len(frontier_methods) - 1:
            continue
        frontier_axes.annotate(
            f"{frontier_taus[frontier_index]:.2f}",
            (frontier_worst[frontier_method], frontier_ewr[frontier_method]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#606060",
        )
    for frontier_anchor in ("blackwell", "best_of_nash"):
        frontier_axes.plot(
            frontier_worst[frontier_anchor],
            frontier_ewr[frontier_anchor],
            marker="o",
            markersize=7,
            color=method_colours[frontier_anchor],
            markeredgecolor="white",
        )
        frontier_axes.annotate(
            short_labels[frontier_anchor],
            (frontier_worst[frontier_anchor], frontier_ewr[frontier_anchor]),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=method_colours[frontier_anchor],
        )
    frontier_axes.margins(x=0.15, y=0.15)
    frontier_axes.set_xlabel("Worst criterion win rate")
    frontier_axes.set_ylabel("EWR vs reference")
    frontier_axes.spines[["top", "right"]].set_visible(False)
    frontier_figure.tight_layout()
    frontier_figure
    return


@app.cell
def _(
    frontier_ewr,
    frontier_methods,
    frontier_taus,
    frontier_worst,
    method_colours,
    plt,
):
    tau_figure, tau_axes = plt.subplots(figsize=(5, 3.2))
    for reference_level, reference_method, reference_label in [
        (frontier_ewr["best_of_nash"], "best_of_nash", "Best-of-Nash EWR"),
        (frontier_worst["blackwell"], "blackwell", "Blackwell worst criterion"),
    ]:
        tau_axes.axhline(
            reference_level,
            color=method_colours[reference_method],
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
        tau_axes.annotate(
            reference_label,
            (frontier_taus[0], reference_level),
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=7,
            color=method_colours[reference_method],
        )
    tau_axes.plot(
        frontier_taus,
        [frontier_ewr[method] for method in frontier_methods],
        color="#303030",
        linewidth=1.2,
        marker="o",
        markersize=4,
        label="EWR vs reference",
    )
    tau_axes.plot(
        frontier_taus,
        [frontier_worst[method] for method in frontier_methods],
        color="#909090",
        linewidth=1.2,
        marker="s",
        markersize=4,
        label="Worst criterion win rate",
    )
    tau_axes.set_xlabel("Overall threshold tau")
    tau_axes.set_ylabel("Win rate")
    tau_axes.legend(frameon=False)
    tau_axes.spines[["top", "right"]].set_visible(False)
    tau_figure.tight_layout()
    tau_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Future directions

    - Systematic target sets vs best-linear-scalarisation

    - Uncertainty quantification

    - DPO (how? ov prefs? that would make the worst criterion performance result obvious?) and PROSPER (use their checkpoint)
    """)
    return


if __name__ == "__main__":
    app.run()
