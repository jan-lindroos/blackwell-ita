# /// script
# requires-python = ">=3.13"
# dependencies = ["cvxpy", "huggingface-hub", "marimo", "matplotlib", "numpy", "pandas", "pyarrow"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/conflict_mvp.slides.json",
)

with app.setup:
    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    N = 128
    N_VALUES = [32, 128]
    HEADS = [
        "helpfulness",
        "correctness",
        "coherence",
        "complexity",
        "verbosity",
        "overall",
    ]
    CRITERIA = ["helpfulness", "correctness", "coherence", "simplicity"]
    SCENARIOS = {
        "helpful_incorrect": ([0, 1], 1, ["helpfulness", "incorrectness"]),
        "simplicity": ([0, 1, 2, 3], 3, CRITERIA),
        "flipped_correctness": (
            [0, 1, 2],
            1,
            ["helpfulness", "incorrectness", "coherence"],
        ),
    }
    METRICS = ["minimum", "geometric", "arithmetic", "tokens"]


@app.cell
def _():
    return


@app.function
def objectives(scores, scenario="simplicity"):
    """Apply the same selected heads and reversal to selection and evaluation."""
    indices, flipped, _ = SCENARIOS[scenario]
    result = np.array(scores[indices], dtype=float, copy=True)
    result[flipped] = 1.0 - result[flipped]
    return result


@app.function
def winner(tensor, norm="inf"):
    """Minimize the worst opponent's Lp distance to the 0.5 orthant."""
    count = tensor.shape[1]
    policy = cp.Variable(count, nonneg=True)
    shortfall = cp.Variable(nonneg=True)
    assert norm in ("inf", "1", "2")
    constraints = [cp.sum(policy) == 1]
    if norm == "inf":
        constraints += [head.T @ policy + shortfall >= 0.5 for head in tensor]
    else:
        # Columns are opponents: take the norm across criteria BEFORE max over opponents.
        deficits = cp.Variable((len(tensor), count), nonneg=True)
        constraints += [
            deficits[g] >= 0.5 - head.T @ policy for g, head in enumerate(tensor)
        ]
        distances = (
            cp.sum(deficits, axis=0) if norm == "1" else cp.norm(deficits, 2, axis=0)
        )
        constraints += [distances <= shortfall]
    problem = cp.Problem(cp.Minimize(shortfall), constraints)
    problem.solve(solver=cp.CLARABEL)
    if norm != "inf" and problem.status == cp.OPTIMAL_INACCURATE:
        # Degenerate cones can stall Clarabel; require a converged SCS solve.
        problem.solve(solver=cp.SCS, eps=1e-6, max_iters=100000)
    if problem.status != cp.OPTIMAL or policy.value is None:
        raise RuntimeError(f"Policy solve failed: {problem.status}")
    weights = np.clip(policy.value, 0.0, None)
    return weights / weights.sum()


@app.function
def select_policies(tensor, scenario="simplicity", norm="inf"):
    quality = objectives(tensor, scenario)
    scalar = quality.mean(axis=0)
    # Includes self-comparisons (all 0.5), which do not change the ranking.
    best = int(np.argmax(scalar.mean(axis=1)))
    return {
        "blackwell": winner(quality, norm),
        "scalarized_nash": winner(scalar[None]),
        "best_of_n": np.eye(len(scalar))[best],
    }


@app.function
def evaluate_policy(policy, anchor_scores, tokens, scenario="simplicity"):
    """Aggregate objectives after taking the policy expectation, within a prompt."""
    rates = objectives(anchor_scores, scenario) @ policy
    geometric = 0.0 if np.any(rates == 0) else float(np.exp(np.log(rates).mean()))
    return {
        **dict(zip(SCENARIOS[scenario][2], rates, strict=True)),
        "minimum": float(rates.min()),
        "geometric": geometric,
        "arithmetic": float(rates.mean()),
        "tokens": float(np.asarray(tokens) @ policy),
    }


@app.function
def run_experiment(
    selector, evaluator, candidates, n=N, scenario="simplicity", norm="inf"
):
    prompts = selector["prompts"].tolist()
    assert len(prompts) == len(set(prompts))
    assert evaluator["prompts"].tolist() == prompts
    for tensors in (selector, evaluator):
        assert tensors["criteria"].tolist() == HEADS
    rows = []
    for index, prompt in enumerate(prompts):
        selection = selector[f"tensor_{index}"]
        evaluation = evaluator[f"tensor_{index}"]
        assert selection.shape == evaluation.shape
        assert selection.shape[0] == len(HEADS)
        assert selection.shape[1] == selection.shape[2] and selection.shape[1] > n
        pool = (
            candidates[candidates["prompt"] == prompt]
            .sort_values("sample_index")
            .iloc[:n]
        )
        assert pool["sample_index"].tolist() == list(range(n))
        assert np.isfinite(pool["tokens"]).all() and (pool["tokens"] >= 0).all()
        for tensor in (selection, evaluation):
            assert np.isfinite(tensor).all() and tensor.min() >= 0 and tensor.max() <= 1
        for method, policy in select_policies(
            selection[:, :n, :n], scenario, norm
        ).items():
            for grader, scores in (
                ("Phi (independent)", evaluation),
                ("Qwen (self-evaluation)", selection),
            ):
                rows.append(
                    {
                        "prompt": prompt,
                        "method": method,
                        "grader": grader,
                        "scenario": scenario,
                        "n": n,
                        **evaluate_policy(
                            policy,
                            scores[:, :n, -1],
                            pool["tokens"].to_numpy(),
                            scenario,
                        ),
                    }
                )
    return pd.DataFrame(rows)


@app.function
def comparison_statistics(results, draws=5000, seed=1810):
    """Means plus percentile CIs for paired Blackwell-minus-baseline differences."""
    assert not results.duplicated(["prompt", "method"]).any(), (
        "Pass one scenario, N, and grader at a time"
    )
    prompts = sorted(results["prompt"].unique())
    weights = np.random.default_rng(seed).multinomial(
        len(prompts), np.full(len(prompts), 1 / len(prompts)), size=draws
    ) / len(prompts)
    values = {
        method: group.set_index("prompt").reindex(prompts)[METRICS].to_numpy()
        for method, group in results.groupby("method")
    }
    assert all(np.isfinite(value).all() for value in values.values())
    table = pd.DataFrame(
        {method: value.mean(axis=0) for method, value in values.items()}, index=METRICS
    )
    differences = []
    for baseline in ("scalarized_nash", "best_of_n"):
        difference = values["blackwell"] - values[baseline]
        low, high = np.quantile(weights @ difference, [0.025, 0.975], axis=0)
        differences.extend(
            {
                "baseline": baseline,
                "metric": metric,
                "difference": mean,
                "low": lo,
                "high": hi,
            }
            for metric, mean, lo, hi in zip(
                METRICS, difference.mean(axis=0), low, high, strict=True
            )
        )
    return table.reset_index(names="metric"), pd.DataFrame(differences)


@app.function
def comparison_table(results, draws=5000, seed=1810):
    table, differences = comparison_statistics(results, draws, seed)
    for baseline, group in differences.groupby("baseline", sort=False):
        ordered = group.set_index("metric").loc[table["metric"]]
        table[f"Blackwell − {baseline} [95% CI]"] = [
            f"{row.difference:+.4f} [{row.low:+.4f}, {row.high:+.4f}]"
            for row in ordered.itertuples()
        ]
    return table


@app.function
def heatmap_differences(results):
    frames = []
    for (scenario, grader, n), group in results.groupby(["scenario", "grader", "n"]):
        _, differences = comparison_statistics(group)
        frames.append(differences.assign(scenario=scenario, grader=grader, n=n))
    return pd.concat(frames, ignore_index=True)


@app.function
def chart_ablations(differences):
    """Shared symmetric scale; colors and labels are percentage-point differences."""
    scenarios = ["simplicity", "flipped_correctness", "helpful_incorrect"]
    labels = [
        "Helpfulness, correctness,\ncoherence, simplicity",
        "Helpfulness, incorrectness,\ncoherence",
        "Helpfulness, incorrectness",
    ]
    graders = ["Phi (independent)", "Qwen (self-evaluation)"]
    baselines = ["scalarized_nash", "best_of_n"]
    metrics = METRICS[:3]
    data = differences[differences["metric"].isin(metrics)]
    limit = max(float(data["difference"].abs().max()) * 100, 0.1)
    figure, axes = plt.subplots(2, 2, figsize=(14, 7), layout="constrained")
    figure.get_layout_engine().set(wspace=0.08, hspace=0.12)
    for row, grader in enumerate(graders):
        for col, n in enumerate(N_VALUES):
            ax = axes[row, col]
            panel = data[(data["grader"] == grader) & (data["n"] == n)].set_index(
                ["scenario", "baseline", "metric"]
            )
            values = np.full((3, 7), np.nan)
            for i, scenario in enumerate(scenarios):
                for b, baseline in enumerate(baselines):
                    for j, metric in enumerate(metrics):
                        stat = panel.loc[(scenario, baseline, metric)]
                        x = b * 4 + j
                        value = float(stat["difference"]) * 100
                        values[i, x] = value
                        marker = "*" if stat["low"] > 0 or stat["high"] < 0 else ""
                        ax.text(
                            x,
                            i,
                            f"{value:+.2f}{marker}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white" if abs(value) > 0.65 * limit else "#222222",
                        )
            raster = ax.imshow(
                np.ma.masked_invalid(values),
                cmap="RdBu",
                vmin=-limit,
                vmax=limit,
                aspect="equal",
            )
            ax.set_title(
                f"{'Phi' if row == 0 else 'Self-evaluation'} N={n}",
                loc="left",
                pad=40,
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                1,
                1.05,
                "vs scalarized Nash",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=10,
            )
            ax.text(
                5,
                1.05,
                "vs Best-of-N",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=10,
            )
            ax.set_xticks([0, 1, 2, 4, 5, 6], ["Min", "Geom", "Arith"] * 2)
            ax.set_yticks(range(3), labels if col == 0 else ["", "", ""])
            ax.tick_params(length=0, pad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
    figure.colorbar(
        raster, ax=axes, shrink=0.8, label="Blackwell − baseline (percentage points)"
    )
    return figure


@app.cell
def _():
    def artifact(filename):
        return hf_hub_download(
            "blackwell-ita/artifacts", f"helpsteer2/{filename}", repo_type="dataset"
        )

    selector = np.load(artifact("preference_tensors.npz"))
    evaluator = np.load(artifact("preference_tensors_eval.npz"))
    candidates = pd.read_parquet(artifact("candidates.parquet"))
    return candidates, evaluator, selector


@app.cell
def _(candidates, evaluator, selector):
    results = pd.concat(
        [
            run_experiment(selector, evaluator, candidates, n=n, scenario=scenario)
            for scenario in SCENARIOS
            for n in N_VALUES
        ],
        ignore_index=True,
    )
    return (results,)


@app.cell
def _():
    return


@app.cell
def _(results):
    differences = heatmap_differences(results)
    figure = chart_ablations(differences)
    figure
    return (differences,)


@app.cell
def _(differences, results):
    mo.accordion(
        {
            "Absolute scores and expected tokens": mo.ui.table(
                results.groupby(["scenario", "grader", "n", "method"])[METRICS]
                .mean()
                .reset_index(),
                selection=None,
            ),
            "Paired differences and exact 95% intervals": mo.ui.table(
                differences, selection=None
            ),
        }
    )
    return


@app.cell
def _(results):
    mo.accordion(
        {
            "Per-criterion means": results.groupby(
                ["scenario", "grader", "n", "method"]
            )[[*CRITERIA, "incorrectness"]].mean(),
            "Per-prompt results": mo.ui.table(results, selection=None),
        }
    )
    return


@app.cell
def _():
    norm_button = mo.ui.run_button(label="Run L1 and L2 at N=128 (self-evaluation)")
    norm_button
    return (norm_button,)


@app.cell
def _(candidates, norm_button, results, selector):
    mo.stop(not norm_button.value)
    norm_runs = []
    for norm_scenario in SCENARIOS:
        for norm_value in ("1", "2"):
            norm_run = run_experiment(
                selector,
                selector,
                candidates,
                n=128,
                scenario=norm_scenario,
                norm=norm_value,
            )
            norm_runs.append(
                norm_run[norm_run["grader"] == "Qwen (self-evaluation)"].assign(
                    norm=norm_value
                )
            )
    infinity_results = results[
        (results["n"] == 128) & (results["grader"] == "Qwen (self-evaluation)")
    ].assign(norm="inf")
    norm_results = pd.concat([infinity_results, *norm_runs], ignore_index=True)
    return (norm_results,)


@app.function
def chart_norms(norm_results):
    """Both baseline comparisons for every norm, with paired prompt intervals."""
    scenarios = ["simplicity", "flipped_correctness", "helpful_incorrect"]
    titles = [
        "Helpfulness, correctness,\ncoherence, simplicity",
        "Helpfulness, incorrectness,\ncoherence",
        "Helpfulness, incorrectness",
    ]
    norms = ["inf", "1", "2"]
    baselines = ["scalarized_nash", "best_of_n"]
    panels = {}
    intervals = {}
    for index, scenario in enumerate(scenarios):
        for norm_index, norm in enumerate(norms):
            group = norm_results[
                (norm_results["scenario"] == scenario) & (norm_results["norm"] == norm)
            ]
            _, stats = comparison_statistics(group)
            for row, baseline in enumerate(baselines):
                selected = (
                    stats[stats["baseline"] == baseline]
                    .set_index("metric")
                    .loc[METRICS[:3]]
                )
                panels.setdefault((row, index), np.zeros((3, 3)))[norm_index] = (
                    selected["difference"].to_numpy() * 100
                )
                intervals.setdefault((row, index), np.zeros((3, 3), dtype=bool))[
                    norm_index
                ] = ((selected["low"] > 0) | (selected["high"] < 0)).to_numpy()
    limit = max(max(float(np.abs(panel).max()) for panel in panels.values()), 0.1)
    figure, axes = plt.subplots(2, 3, figsize=(12, 9), layout="constrained")
    figure.get_layout_engine().set(wspace=0.08, hspace=0.12)
    for row, baseline in enumerate(baselines):
        for index, ax in enumerate(axes[row]):
            values = panels[row, index]
            raster = ax.imshow(
                values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="equal"
            )
            for i in range(3):
                for j in range(3):
                    value = values[i, j]
                    star = "*" if intervals[row, index][i, j] else ""
                    ax.text(
                        j,
                        i,
                        f"{value:+.2f}{star}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="white" if abs(value) > 0.65 * limit else "#222222",
                    )
            ax.set_title(titles[index], fontsize=11, pad=14)
            ax.set_xticks(range(3), ["Min", "Geom", "Arith"])
            ax.set_yticks(range(3), ["L∞", "L₁", "L₂"] if index == 0 else ["", "", ""])
            if index == 0:
                ax.set_ylabel(
                    "vs scalarized Nash" if row == 0 else "vs Best-of-N",
                    fontsize=12,
                    labelpad=15,
                )
            ax.tick_params(length=0, pad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
    figure.colorbar(
        raster, ax=axes, shrink=0.8, label="Blackwell − baseline (percentage points)"
    )
    figure.supxlabel(
        "Self-evaluation N=128 · * paired 95% CI excludes zero (unadjusted)",
        fontsize=10,
    )
    return figure


@app.cell
def _(norm_results):
    norm_figure = chart_norms(norm_results)
    norm_figure
    return


if __name__ == "__main__":
    app.run()
