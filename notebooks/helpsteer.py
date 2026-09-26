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
    import marimo as mo
    import pandas as pd
    import torch

    from blackwell_ita.data import (
        CRITERIA,
        assign_prompt_splits,
        load_helpsteer2_pairs,
        split_summary,
    )
    from blackwell_ita.hub import (
        HUB_PREFIX,
        PAIRS_FILENAME,
        SPLITS_REPOSITORY,
        hub_file_exists,
        read_hub_dataframe,
        upload_dataframe,
    )
    from blackwell_ita.training import REWARD_MODEL_CONFIGURATIONS, ensure_trained

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
        upload_dataframe(
            SPLITS_REPOSITORY,
            PAIRS_FILENAME,
            assign_prompt_splits(load_helpsteer2_pairs()),
            HUB_PREFIX,
        )
    pairs = read_hub_dataframe(SPLITS_REPOSITORY, PAIRS_FILENAME, HUB_PREFIX)
    split_summary(pairs)
    return (pairs,)


@app.cell
def _(pairs):
    reward_model_metrics = pd.concat(
        [
            ensure_trained(configuration, pairs, CRITERIA, HUB_PREFIX, DEVICE)
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
