# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita@jan",
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
    import functools

    import marimo as mo
    import pandas as pd
    import torch

    from blackwell_ita.data import split_summary
    from blackwell_ita.habermas import (
        CANDIDATE_SLOTS,
        DRAWS_PER_SLOT,
        GOLD_CONFIGURATION,
        HABERMAS_CRITERIA,
        HABERMAS_PREFIX,
        PARTICIPANT_PAIRS,
        PROXY_CONFIGURATIONS,
        SELECTOR_PREFIX,
        WHOLE_GROUP,
        complete_groups,
        ensure_groups,
        ensure_rankings,
        ensure_winners,
        evaluation_results,
        gold_frontier,
        gold_pairs,
        habermas_pairs,
        instance_welfare,
        participant_preference_tensor,
        participant_winner_rates,
        pool_statements,
    )
    from blackwell_ita.habermas_views import (
        beaten_histogram,
        explorer_controls,
        frontier_chart,
        nash_chart,
        question_view,
        winner_chart,
    )
    from blackwell_ita.hub import (
        ARTIFACTS_REPOSITORY,
        REWARD_MODELS_REPOSITORY,
        hub_file_exists,
        read_hub_dataframe,
    )
    from blackwell_ita.scoring import download_tensors, score_with_resume
    from blackwell_ita.training import ensure_trained, load_trained

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEST_POOL_SIZE = DRAWS_PER_SLOT["test"] * len(CANDIDATE_SLOTS)


@app.function
def score_participant_tensors(
    configuration,
    model_prefix: str,
    filename: str,
    keyed_inputs: dict,
    score_one=participant_preference_tensor,
) -> dict:
    """Predicted participant tensors per key, loading the model only if needed.

    On CPU nothing is scored, only the keys already on the hub come back.
    """
    if DEVICE == "cpu":
        cached_tensors = download_tensors(
            ARTIFACTS_REPOSITORY, filename, HABERMAS_PREFIX
        )
        return {
            key: cached_tensors[key] for key in keyed_inputs if key in cached_tensors
        }
    load_model = functools.cache(
        lambda: load_trained(configuration, model_prefix, DEVICE)
    )
    predicted_tensors = score_with_resume(
        ARTIFACTS_REPOSITORY,
        filename,
        HABERMAS_PREFIX,
        keyed_inputs,
        lambda inputs: score_one(load_model(), device=DEVICE, **inputs),
    )
    load_model.cache_clear()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return predicted_tensors


@app.function
def available_on_hub(repository_id: str, filename: str, prefix: str) -> bool:
    """Whether a step can run: always on a GPU, on CPU only if the hub has its output."""
    return DEVICE != "cpu" or hub_file_exists(repository_id, filename, prefix)


@app.cell
def _():
    rankings = ensure_rankings()
    human_pairs = habermas_pairs(rankings[rankings["split"] != "paper"])
    split_summary(human_pairs, "question_id")
    return human_pairs, rankings


@app.cell
def _(human_pairs):
    mo.stop(
        not available_on_hub(
            REWARD_MODELS_REPOSITORY,
            GOLD_CONFIGURATION.checkpoint_filename,
            HABERMAS_PREFIX,
        ),
        mo.md("The gold model is not on the hub yet."),
    )
    gold_metrics = ensure_trained(
        GOLD_CONFIGURATION, human_pairs, HABERMAS_CRITERIA, HABERMAS_PREFIX, DEVICE
    )
    gold_metrics[["split", "decisive_accuracy", "decisive_count", "pooled_loss"]]
    return


@app.cell
def _(rankings):
    groups = ensure_groups(rankings)
    groups
    return (groups,)


@app.cell
def _():
    mo.stop(
        not hub_file_exists(
            ARTIFACTS_REPOSITORY, "candidates.parquet", HABERMAS_PREFIX
        ),
        mo.md(
            "The candidate pools are not on the hub yet. Run `uv run --env-file .env python -m blackwell_ita.habermas` locally."
        ),
    )
    candidates = read_hub_dataframe(
        ARTIFACTS_REPOSITORY, "candidates.parquet", HABERMAS_PREFIX
    )
    candidates.groupby("candidate_kind")["response"].apply(
        lambda responses: responses.str.split().str.len().describe()
    ).unstack()
    return (candidates,)


@app.cell
def _(candidates, groups):
    pool_inputs = {
        split_name: {
            group_row["question_id"]: {
                "question": group_row["question"],
                "opinions": list(group_row["opinions"]),
                "statements": pool_statements(candidates, group_row["question_id"]),
            }
            for group_row in split_groups.to_dict("records")
        }
        for split_name, split_groups in groups.groupby("split", sort=False)
    }
    gold_tensors = {
        split_name: score_participant_tensors(
            GOLD_CONFIGURATION, HABERMAS_PREFIX, f"gold_{split_name}.npz", split_inputs
        )
        for split_name, split_inputs in pool_inputs.items()
    }
    pd.DataFrame(
        {
            "groups": {name: len(inputs) for name, inputs in pool_inputs.items()},
            "gold_scored": {
                name: len(tensors) for name, tensors in gold_tensors.items()
            },
        }
    )
    return gold_tensors, pool_inputs


@app.cell
def _(groups, pool_inputs, rankings):
    # Only the gold model sees the winners, as opponents in grading
    winners = ensure_winners(groups, rankings)
    gold_winner_rates = score_participant_tensors(
        GOLD_CONFIGURATION,
        HABERMAS_PREFIX,
        "gold_test_winner.npz",
        {
            winner_row["question_id"]: pool_inputs["test"][winner_row["question_id"]]
            | {"winner": winner_row["winner"]}
            for winner_row in winners.to_dict("records")
        },
        score_one=participant_winner_rates,
    )
    return (gold_winner_rates,)


@app.cell
def _(candidates, gold_tensors, groups, pool_inputs):
    mo.stop(
        any(
            len(gold_tensors[split_name]) < len(split_inputs)
            for split_name, split_inputs in pool_inputs.items()
        ),
        mo.md("The gold model has not scored every pool yet."),
    )
    # The proxies only ever see gold labels on generated pools
    proxy_pairs = gold_pairs(
        groups,
        candidates,
        {
            question_id: tensor
            for split_tensors in gold_tensors.values()
            for question_id, tensor in split_tensors.items()
        },
    )
    trained_proxies = [
        configuration
        for configuration in PROXY_CONFIGURATIONS
        if available_on_hub(
            REWARD_MODELS_REPOSITORY, configuration.checkpoint_filename, SELECTOR_PREFIX
        )
    ]
    mo.stop(not trained_proxies, mo.md("No proxy model is on the hub yet."))
    proxy_metrics = pd.concat(
        [
            ensure_trained(
                configuration,
                proxy_pairs,
                HABERMAS_CRITERIA,
                SELECTOR_PREFIX,
                DEVICE,
            )
            for configuration in trained_proxies
        ],
        ignore_index=True,
    )
    proxy_metrics.pivot_table(
        index="split", columns="name", values=["decisive_accuracy", "pooled_loss"]
    )
    return (trained_proxies,)


@app.cell
def _(gold_tensors, gold_winner_rates, groups, pool_inputs, trained_proxies):
    proxy_tensors = {
        configuration.name: score_participant_tensors(
            configuration,
            SELECTOR_PREFIX,
            f"proxy_test_{configuration.name}.npz",
            pool_inputs["test"],
        )
        for configuration in trained_proxies
    }
    mo.stop(
        not all(proxy_tensors.values()),
        mo.md("The proxies have not scored the test pools yet."),
    )
    # Groups with a participant who wrote no opinion stay out of every result
    test_groups = complete_groups(groups[groups["split"] == "test"])
    results = pd.concat(
        [
            evaluation_results(
                test_groups[test_groups["question_id"].isin(list(selector_tensors))],  # pyright: ignore[reportArgumentType]
                selector_tensors,
                gold_tensors["test"],
                selector_name,
                WHOLE_GROUP + PARTICIPANT_PAIRS,
                gold_winner_rates,
            )
            for selector_name, selector_tensors in proxy_tensors.items()
        ],
        ignore_index=True,
    )
    welfare = instance_welfare(results)
    welfare.groupby(["group_size", "selector", "method"], sort=False)[
        ["average", "best", "worst", "nash"]
    ].mean()
    return proxy_tensors, results, test_groups, welfare


@app.cell
def _(gold_tensors, test_groups, welfare):
    frontier_chart(
        gold_frontier(test_groups, gold_tensors["test"], WHOLE_GROUP, "average"),
        welfare[welfare["group_size"] == 4],
        "average",
        "Average against worst-served participant, whole group",
    )
    return


@app.cell
def _(gold_tensors, test_groups, welfare):
    frontier_chart(
        gold_frontier(test_groups, gold_tensors["test"], WHOLE_GROUP, "best"),
        welfare[welfare["group_size"] == 4],
        "best",
        "Best-served against worst-served participant, whole group",
    )
    return


@app.cell
def _(gold_tensors, test_groups, welfare):
    frontier_chart(
        gold_frontier(test_groups, gold_tensors["test"], PARTICIPANT_PAIRS, "best"),
        welfare[welfare["group_size"] == 2],
        "best",
        "Best-served against worst-served participant, every pair of participants",
    )
    return


@app.cell
def _(welfare):
    nash_chart(welfare[welfare["group_size"] == 4])
    return


@app.cell
def _(welfare):
    winner_chart(welfare[welfare["group_size"] == 4])
    return


@app.cell
def _(results):
    beaten_histogram(results[results["group_size"] == 4], TEST_POOL_SIZE)
    return


@app.cell
def _(proxy_tensors, test_groups):
    explorer_question, explorer_selector = explorer_controls(test_groups, proxy_tensors)
    mo.vstack([explorer_question, explorer_selector])
    return explorer_question, explorer_selector


@app.cell
def _(candidates, explorer_question, explorer_selector, proxy_tensors, test_groups):
    question_view(
        test_groups,
        candidates,
        proxy_tensors,
        explorer_question.value,
        explorer_selector.value,
    )
    return


if __name__ == "__main__":
    app.run()
