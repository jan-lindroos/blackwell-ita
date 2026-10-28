# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita@main",
#     "marimo>=0.23.16",
#     # molab's base image ships a torchvision built against a mismatched
#     # torch. Install a matching one so transformers doesn't import the
#     # broken system copy
#     "torchvision",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd

    from blackwell_ita.evaluation import (
        MAIN_METHODS,
        WELFARES,
        accuracy,
        bootstrap_summary,
        method_results,
        model_error,
        paired_gaps,
        pool_size_results,
        selection_policies,
        targeting_results,
        welfare_optima,
    )
    from blackwell_ita.habermas import (
        GOLD_CONFIGURATION,
        HABERMAS_PREFIX,
        PROXY_CONFIGURATIONS,
        WHOLE_GROUP,
        complete_groups,
    )
    from blackwell_ita.habermas_views import (
        LEVEL_LABELS,
        MODEL_LABELS,
        explorer_controls,
        question_view,
        welfare_row_chart,
    )
    from blackwell_ita.hub import (
        ARTIFACTS_REPOSITORY,
        REWARD_MODELS_REPOSITORY,
        ensure_hub_dataframe,
        hub_file_exists,
        read_hub_dataframe,
    )
    from blackwell_ita.scoring import download_tensors

    # The test split during development, the paper split once the analysis is frozen
    EVALUATION_SPLIT = "test"
    MAIN_PROXY = "qwen3_4b_bradley_terry"
    POOL_SIZES = [4, 8, 12, 16, 24]
    TARGETING_LEVELS = {
        "whole_group": {"consensus": 12},
        "mixed": {"consensus": 6, "participant": 6},
        "single_participant": {"participant": 12},
    }
    SUBSET_COUNT = 50


@app.cell(hide_code=True)
def _():
    mo.md(f"# Results on the {EVALUATION_SPLIT} split")
    return


@app.cell
def _():
    groups = read_hub_dataframe(ARTIFACTS_REPOSITORY, "groups.parquet", HABERMAS_PREFIX)
    candidates = read_hub_dataframe(
        ARTIFACTS_REPOSITORY, "candidates.parquet", HABERMAS_PREFIX
    )
    gold_tensors = download_tensors(
        ARTIFACTS_REPOSITORY, f"gold_{EVALUATION_SPLIT}.npz", HABERMAS_PREFIX
    )
    mo.stop(not gold_tensors, mo.md("The gold model has not scored these pools yet."))
    proxy_tensors = {
        configuration.name: download_tensors(
            ARTIFACTS_REPOSITORY,
            f"proxy_{EVALUATION_SPLIT}_{configuration.name}.npz",
            HABERMAS_PREFIX,
        )
        for configuration in PROXY_CONFIGURATIONS
    }
    proxy_tensors = {
        name: tensors for name, tensors in proxy_tensors.items() if tensors
    }
    scored = set(gold_tensors).intersection(*proxy_tensors.values())
    # Groups with a participant who wrote no opinion stay out of every result
    evaluation_groups = complete_groups(groups[groups["split"] == EVALUATION_SPLIT])  # pyright: ignore[reportArgumentType]
    evaluation_groups = evaluation_groups[
        evaluation_groups["question_id"].isin(list(scored))
    ]
    selector_tensors = {"gold": gold_tensors} | proxy_tensors
    bradley_terry = {"gold": False} | {
        configuration.name: configuration.model_type == "bradley_terry"
        for configuration in PROXY_CONFIGURATIONS
        if configuration.name in proxy_tensors
    }
    candidate_kinds = {
        question_id: pool["candidate_kind"].to_numpy()
        for question_id, pool in candidates.sort_values("sample_index").groupby(
            "question_id"
        )
    }
    # The grid selects with the main proxy, or with the gold until the proxies exist
    grid_selector = MAIN_PROXY if MAIN_PROXY in proxy_tensors else "gold"
    pd.DataFrame(
        {
            "scored_pools": {
                name: len(tensors) for name, tensors in selector_tensors.items()
            },
            "evaluated_groups": len(evaluation_groups),
        }
    )
    return (
        bradley_terry,
        candidate_kinds,
        candidates,
        evaluation_groups,
        gold_tensors,
        grid_selector,
        proxy_tensors,
        selector_tensors,
    )


@app.cell
def _(evaluation_groups, gold_tensors, proxy_tensors):
    gold_metrics = read_hub_dataframe(
        REWARD_MODELS_REPOSITORY, GOLD_CONFIGURATION.metrics_filename, HABERMAS_PREFIX
    )
    if hub_file_exists(
        ARTIFACTS_REPOSITORY, "gold_paper_metrics.parquet", HABERMAS_PREFIX
    ):
        gold_metrics = pd.concat(
            [
                gold_metrics,
                read_hub_dataframe(
                    ARTIFACTS_REPOSITORY, "gold_paper_metrics.parquet", HABERMAS_PREFIX
                ),
            ],
            ignore_index=True,
        )
    proxy_quality = pd.DataFrame(
        [
            {
                "proxy": MODEL_LABELS[proxy_name],
                "question_id": question_id,
                "model_error": model_error(
                    tensors[question_id], gold_tensors[question_id]
                ).max(),
                "accuracy": accuracy(tensors[question_id], gold_tensors[question_id]),
            }
            for proxy_name, tensors in proxy_tensors.items()
            for question_id in evaluation_groups["question_id"]
        ],
        columns=["proxy", "question_id", "model_error", "accuracy"],
    )
    mo.vstack(
        [
            mo.md("## The gold model against held-out human preferences"),
            # Human labels are 1, 0 or a tie, so decisive pairs are every pair with a preference
            gold_metrics[
                ["split", "loss", "decisive_accuracy", "decisive_count"]
            ].rename(  # pyright: ignore[reportCallIssue]
                columns={
                    "decisive_accuracy": "accuracy",
                    "decisive_count": "pairs_with_a_preference",
                }
            ),
            mo.md("## Each proxy against the gold model"),
            proxy_quality.groupby("proxy", sort=False)[
                ["model_error", "accuracy"]
            ].mean(),
            mo.md(
                "Accuracy is the share of pairs where the model predicts the right preference, giving the preferred statement a probability above one half. Pairs with no preference are left out."
            ),
        ]
    )
    return


@app.cell
def _(bradley_terry, evaluation_groups, gold_tensors, selector_tensors):
    results = pd.concat(
        [
            method_results(
                evaluation_groups,
                tensors,
                gold_tensors,
                selector_name,
                WHOLE_GROUP,
                bradley_terry[selector_name],
                MAIN_METHODS,
            )
            for selector_name, tensors in selector_tensors.items()
        ],
        ignore_index=True,
    )
    regret_summary = pd.concat(
        [
            bootstrap_summary(
                results, ["selector", "method"], f"{welfare}_regret"
            ).assign(welfare=welfare)
            for welfare in WELFARES
        ],
        ignore_index=True,
    )
    mo.vstack(
        [
            mo.md("## Welfare regret on the full pool"),
            regret_summary.pivot_table(
                index=["selector", "method"],
                columns="welfare",
                values=["value", "interval_low", "interval_high"],
                sort=False,
            ),
            mo.md("## Each welfare's own method minus each baseline"),
            paired_gaps(results, ["selector"]),
        ]
    )
    return (results,)


@app.cell
def _(
    bradley_terry,
    candidate_kinds,
    evaluation_groups,
    gold_tensors,
    grid_selector,
    selector_tensors,
):
    pool_sizes = ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        f"pool_sizes_{EVALUATION_SPLIT}_{grid_selector}.parquet",
        HABERMAS_PREFIX,
        lambda: pool_size_results(
            evaluation_groups,
            selector_tensors[grid_selector],
            gold_tensors,
            candidate_kinds,
            POOL_SIZES,
            SUBSET_COUNT,
            bradley_terry[grid_selector],
        ),
    )
    mo.vstack(
        [
            mo.md(f"## Pool size, {MODEL_LABELS[grid_selector]} selecting"),
            welfare_row_chart(pool_sizes, "pool_size", "Pool size"),
        ]
    )
    return


@app.cell
def _(
    bradley_terry,
    candidate_kinds,
    evaluation_groups,
    gold_tensors,
    grid_selector,
    selector_tensors,
):
    targeting = ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        f"targeting_{EVALUATION_SPLIT}_{grid_selector}.parquet",
        HABERMAS_PREFIX,
        lambda: targeting_results(
            evaluation_groups,
            selector_tensors[grid_selector],
            gold_tensors,
            candidate_kinds,
            TARGETING_LEVELS,
            SUBSET_COUNT,
            bradley_terry[grid_selector],
        ),
    )
    targeting = targeting.assign(level_label=targeting["level"].map(LEVEL_LABELS))  # pyright: ignore[reportArgumentType]
    mo.vstack(
        [
            mo.md(f"## Candidate targeting, {MODEL_LABELS[grid_selector]} selecting"),
            welfare_row_chart(
                targeting,
                "level_label",
                "Candidate targeting",
                [LEVEL_LABELS[level] for level in TARGETING_LEVELS],
            ),
        ]
    )
    return


@app.cell
def _(evaluation_groups, gold_tensors, proxy_tensors, results):
    mo.stop(not proxy_tensors, mo.md("No proxy has scored these pools yet."))
    proxy_errors = {
        proxy_name: float(
            pd.Series(
                [
                    model_error(tensors[question_id], gold_tensors[question_id]).max()
                    for question_id in evaluation_groups["question_id"]
                ]
            ).mean()
        )
        for proxy_name, tensors in proxy_tensors.items()
    }
    proxy_results = results[results["selector"].isin(list(proxy_errors))]
    proxy_results = proxy_results.assign(
        model_error=proxy_results["selector"].map(proxy_errors)
    )
    mo.vstack(
        [
            mo.md("## Proxy quality"),
            welfare_row_chart(
                proxy_results, "model_error", "Proxy model error, worst participant"
            ),
        ]
    )
    return


@app.cell
def _(evaluation_groups, selector_tensors):
    explorer_question, explorer_selector = explorer_controls(
        evaluation_groups, list(selector_tensors)
    )
    mo.vstack([explorer_question, explorer_selector])
    return explorer_question, explorer_selector


@app.cell
def _(
    bradley_terry,
    candidates,
    evaluation_groups,
    explorer_question,
    explorer_selector,
    gold_tensors,
    selector_tensors,
):
    question_view(
        evaluation_groups,
        candidates,
        selection_policies(
            selector_tensors[explorer_selector.value][explorer_question.value],
            bradley_terry[explorer_selector.value],
            welfare_optima(gold_tensors[explorer_question.value])["nash"],
            float(
                model_error(
                    selector_tensors[explorer_selector.value][explorer_question.value],
                    gold_tensors[explorer_question.value],
                ).max()
            ),
        ),
        explorer_question.value,
    )
    return


if __name__ == "__main__":
    app.run()
