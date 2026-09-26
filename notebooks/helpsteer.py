# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita",
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
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import torch
    import wandb

    from blackwell_ita.data import (
        assign_prompt_splits,
        load_helpsteer2_pairs,
        split_summary,
    )
    from blackwell_ita.hub import (
        HUB_PREFIX,
        PAIRS_FILENAME,
        REWARD_MODELS_REPOSITORY,
        SPLITS_REPOSITORY,
        download_hub_file,
        hub_file_exists,
        upload_hub_file,
    )
    from blackwell_ita.models import save_reward_model
    from blackwell_ita.training import REWARD_MODEL_CONFIGURATIONS, train_reward_model

    WANDB_PROJECT = "blackwell-ita-reward-models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Run on a GPU with `HF_TOKEN` and `WANDB_API_KEY` set.
    """)
    return


@app.cell
def _():
    if not hub_file_exists(SPLITS_REPOSITORY, PAIRS_FILENAME):
        with tempfile.TemporaryDirectory() as _temporary_directory:
            local_pairs_path = Path(_temporary_directory) / PAIRS_FILENAME
            assign_prompt_splits(load_helpsteer2_pairs()).to_parquet(local_pairs_path)
            upload_hub_file(SPLITS_REPOSITORY, local_pairs_path)
    pairs = pd.read_parquet(download_hub_file(SPLITS_REPOSITORY, PAIRS_FILENAME))
    split_summary(pairs)
    return (pairs,)


@app.cell
def _(pairs):
    for configuration in REWARD_MODEL_CONFIGURATIONS:
        if hub_file_exists(REWARD_MODELS_REPOSITORY, configuration.checkpoint_filename):
            print(f"{configuration.checkpoint_filename} is on the hub, skipping")
            continue
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=configuration.name,
            config=asdict(configuration) | {"hub_prefix": HUB_PREFIX},
        )
        wandb_run.define_metric("*", step_metric="step")
        trained_model, trained_metrics = train_reward_model(
            configuration, pairs, DEVICE, wandb_run.log
        )
        wandb_run.finish()
        with tempfile.TemporaryDirectory() as _temporary_directory:
            local_metrics_path = (
                Path(_temporary_directory) / configuration.metrics_filename
            )
            trained_metrics.to_parquet(local_metrics_path)
            upload_hub_file(REWARD_MODELS_REPOSITORY, local_metrics_path)
            local_checkpoint_path = (
                Path(_temporary_directory) / configuration.checkpoint_filename
            )
            save_reward_model(trained_model, local_checkpoint_path)
            upload_hub_file(REWARD_MODELS_REPOSITORY, local_checkpoint_path)
        del trained_model
    reward_model_metrics = pd.concat(
        [
            pd.read_parquet(
                download_hub_file(
                    REWARD_MODELS_REPOSITORY, configuration.metrics_filename
                )
            )
            for configuration in REWARD_MODEL_CONFIGURATIONS
        ],
        ignore_index=True,
    )
    reward_model_metrics.pivot_table(
        index=["split", "criterion"], columns="name", values="decisive_accuracy"
    )
    return


if __name__ == "__main__":
    app.run()
