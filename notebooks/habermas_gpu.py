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
    import functools

    import marimo as mo
    import pandas as pd
    import torch

    from blackwell_ita.data import (
        PreferencePairDataset,
        evaluation_loader,
        split_summary,
    )
    from blackwell_ita.habermas import (
        GOLD_CONFIGURATION,
        HABERMAS_CRITERIA,
        HABERMAS_PREFIX,
        PROXY_CONFIGURATIONS,
        SELECTOR_PREFIX,
        gold_pairs,
        habermas_pairs,
        participant_preference_tensor,
        scoring_inputs,
    )
    from blackwell_ita.hub import (
        ARTIFACTS_REPOSITORY,
        REWARD_MODELS_REPOSITORY,
        ensure_hub_dataframe,
        hub_file_exists,
        read_hub_dataframe,
    )
    from blackwell_ita.scoring import download_tensors, score_with_resume
    from blackwell_ita.training import ensure_trained, evaluate, load_trained

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    mo.stop(
        not hub_file_exists(ARTIFACTS_REPOSITORY, "rankings.parquet", HABERMAS_PREFIX),
        mo.md(
            "The splits are not on the hub yet. Run `notebooks/habermas_local.py` locally first."
        ),
    )
    rankings = read_hub_dataframe(
        ARTIFACTS_REPOSITORY, "rankings.parquet", HABERMAS_PREFIX
    )
    human_pairs = habermas_pairs(rankings[rankings["split"] != "paper"])  # pyright: ignore[reportArgumentType]
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
    return (gold_metrics,)


@app.cell
def _(gold_metrics, rankings):
    mo.stop(
        not available_on_hub(
            ARTIFACTS_REPOSITORY, "gold_paper_metrics.parquet", HABERMAS_PREFIX
        ),
        mo.md("The gold model is not evaluated on the paper split yet."),
    )

    def paper_metrics() -> pd.DataFrame:
        gold_model = load_trained(GOLD_CONFIGURATION, HABERMAS_PREFIX, DEVICE)
        paper_pairs = PreferencePairDataset(
            habermas_pairs(rankings[rankings["split"] == "paper"]),  # pyright: ignore[reportArgumentType]
            HABERMAS_CRITERIA,
            augment_presentation_order=False,
        )
        pooled_loss, criterion_metrics = evaluate(
            gold_model,
            evaluation_loader(paper_pairs, GOLD_CONFIGURATION.batch_size),
            DEVICE,
        )
        return criterion_metrics.assign(split="paper", pooled_loss=pooled_loss)

    # The paper questions stayed out of the gold's training and early stopping
    gold_paper_metrics = ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "gold_paper_metrics.parquet",
        HABERMAS_PREFIX,
        paper_metrics,
    )
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    pd.concat([gold_metrics, gold_paper_metrics], ignore_index=True)[
        ["split", "decisive_accuracy", "decisive_count", "pooled_loss"]
    ]
    return


@app.cell
def _():
    mo.stop(
        not all(
            hub_file_exists(ARTIFACTS_REPOSITORY, filename, HABERMAS_PREFIX)
            for filename in ["groups.parquet", "candidates.parquet"]
        ),
        mo.md(
            "The candidate pools are not on the hub yet. Run `notebooks/habermas_local.py` locally first."
        ),
    )
    groups, candidates = (
        read_hub_dataframe(ARTIFACTS_REPOSITORY, filename, HABERMAS_PREFIX)
        for filename in ["groups.parquet", "candidates.parquet"]
    )
    return candidates, groups


@app.cell
def _(candidates, groups):
    pool_inputs = scoring_inputs(groups, candidates)
    # Test first, so its results can be checked while the other splits score
    gold_tensors = {
        split_name: score_participant_tensors(
            GOLD_CONFIGURATION,
            HABERMAS_PREFIX,
            f"gold_{split_name}.npz",
            pool_inputs[split_name],
        )
        for split_name in ["test", "validation", "train", "paper"]
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
def _(candidates, gold_tensors, groups, pool_inputs):
    mo.stop(
        any(
            len(gold_tensors[split_name]) < len(split_inputs)
            for split_name, split_inputs in pool_inputs.items()
        ),
        mo.md("The gold model has not scored every pool yet."),
    )
    # The proxies only ever see gold labels on generated pools, never the paper split's
    proxy_pairs = gold_pairs(
        groups[groups["split"] != "paper"],  # pyright: ignore[reportArgumentType]
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
def _(pool_inputs, trained_proxies):
    proxy_tensors = {
        (configuration.name, split_name): score_participant_tensors(
            configuration,
            SELECTOR_PREFIX,
            f"proxy_{split_name}_{configuration.name}.npz",
            pool_inputs[split_name],
        )
        for configuration in trained_proxies
        for split_name in ["test", "paper"]
    }
    pd.Series(
        {key: len(tensors) for key, tensors in proxy_tensors.items()},
        name="scored_pools",
    )
    return


if __name__ == "__main__":
    app.run()
