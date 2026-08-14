# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita",
#     "datasets",
#     "pandas",
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
    from blackwell_ita.human_prefs import (
        community_alignment_pairs,
        helpsteer2_pairs,
    )

    return community_alignment_pairs, helpsteer2_pairs


@app.cell
def _():
    import datasets
    import pandas as pd

    return datasets, pd


@app.cell
def _(mo):
    mo.md("""
    ## HelpSteer2
    """)
    return


@app.cell
def _():
    helpsteer_repo_id = "jan-lindroos/helpsteer2-human-prefs"
    return (helpsteer_repo_id,)


@app.cell
def _(datasets):
    responses_dataframe = datasets.load_dataset(
        "nvidia/HelpSteer2", split="train"
    ).to_pandas()
    responses_dataframe
    return (responses_dataframe,)


@app.cell
def _(pd):
    preferences_dataframe = pd.read_json(
        "hf://datasets/nvidia/HelpSteer2/preference/preference.jsonl.gz", lines=True
    )
    preferences_dataframe = preferences_dataframe[
        preferences_dataframe["split"] == "train"
    ]
    preferences_dataframe
    return (preferences_dataframe,)


@app.cell
def _(helpsteer2_pairs, preferences_dataframe, responses_dataframe):
    helpsteer_pairs_dataframe = helpsteer2_pairs(
        responses_dataframe, preferences_dataframe
    )
    helpsteer_pairs_dataframe
    return (helpsteer_pairs_dataframe,)


@app.cell
def _(mo):
    mo.md("""
    Next step requires running `hf auth login`.
    """)
    return


@app.cell
def _(mo):
    helpsteer_push_button = mo.ui.run_button(label="Push to hub")
    helpsteer_push_button
    return (helpsteer_push_button,)


@app.cell
def _(
    datasets,
    helpsteer_pairs_dataframe,
    helpsteer_push_button,
    helpsteer_repo_id,
    mo,
):
    mo.stop(not helpsteer_push_button.value)
    datasets.Dataset.from_pandas(helpsteer_pairs_dataframe).push_to_hub(
        helpsteer_repo_id
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Community Alignment
    """)
    return


@app.cell
def _():
    language = "en"
    community_repo_id = "jan-lindroos/community-alignment-human-prefs"
    return community_repo_id, language


@app.cell
def _(pd):
    from huggingface_hub import hf_hub_download

    conversations_dataframe = pd.read_csv(
        hf_hub_download(
            "facebook/community-alignment-dataset",
            "community_alignment.csv",
            repo_type="dataset",
        )
    )
    conversations_dataframe
    return (conversations_dataframe,)


@app.cell
def _(community_alignment_pairs, conversations_dataframe, language):
    community_pairs_dataframe = community_alignment_pairs(
        conversations_dataframe, language
    )
    community_pairs_dataframe
    return (community_pairs_dataframe,)


@app.cell
def _(community_pairs_dataframe, mo):
    group_columns = [
        column
        for column in community_pairs_dataframe.columns
        if column not in ("prompt", "response_a", "response_b", "overall")
    ]
    mo.md(
        "Labelled pairs per group: "
        + ", ".join(
            f"{group}: {community_pairs_dataframe[group].notna().sum()}"
            for group in group_columns
        )
    )
    return


@app.cell
def _(mo):
    community_push_button = mo.ui.run_button(label="Push to hub")
    community_push_button
    return (community_push_button,)


@app.cell
def _(
    community_pairs_dataframe,
    community_push_button,
    community_repo_id,
    datasets,
    mo,
):
    mo.stop(not community_push_button.value)
    datasets.Dataset.from_pandas(community_pairs_dataframe).push_to_hub(
        community_repo_id
    )
    return


if __name__ == "__main__":
    app.run()
