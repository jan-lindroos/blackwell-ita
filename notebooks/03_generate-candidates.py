# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita",
#     "datasets",
#     "huggingface_hub",
#     "pandas",
#     "pyarrow",
#     "torch",
#     # molab's base image leaks a torchvision built against a different
#     # torch; install a matching one so transformers doesn't import the
#     # broken system copy
#     "torchvision",
#     "transformers",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Before running, switch to GPU. Then sign in with `hf auth login` (open terminal available via command menu) and add `blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita` and `torchvision` to dependencies via uv (in the sidebar, under packages).
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from blackwell_ita.artifacts import model_path, upload_artifact
    from blackwell_ita.generate import generate_candidates
    from blackwell_ita.human_prefs import (
        community_alignment_anchors,
        helpsteer2_anchors,
        prompt_splits,
    )

    return (
        community_alignment_anchors,
        generate_candidates,
        helpsteer2_anchors,
        model_path,
        prompt_splits,
        upload_artifact,
    )


@app.cell
def _():
    import json
    import tempfile
    from pathlib import Path

    import datasets
    import pandas as pd
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    return (
        EntryNotFoundError,
        Path,
        RepositoryNotFoundError,
        datasets,
        json,
        pd,
        tempfile,
    )


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options={
            "helpsteer2": "jan-lindroos/helpsteer2-human-prefs",
            "community-alignment": "jan-lindroos/community-alignment-human-prefs",
        },
        value="helpsteer2",
        label="Dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _():
    base_model_name = "RLHFlow/LLaMA3-SFT-v2"
    evaluation_prompt_count = 100
    samples_per_prompt = 64
    seed = 1810
    return base_model_name, evaluation_prompt_count, samples_per_prompt, seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation prompts and reference anchors

    ### HelpSteer2
    Prompts from the validation split; the
    anchor is the pair response preferred overall in HelpSteer2-Preference. Prompts whose pair lacks a preference annotation or ties on overall
    preference have no strict human winner and are skipped.

    ### Community Alignment
    The 100 held-out prompts from `prompt_splits` (excluded
    from both training halves); the anchor is the response with the highest
    pooled human win fraction.
    """)
    return


@app.cell
def _(
    EntryNotFoundError,
    RepositoryNotFoundError,
    community_alignment_anchors,
    dataset_picker,
    datasets,
    evaluation_prompt_count,
    helpsteer2_anchors,
    json,
    model_path,
    pd,
    prompt_splits,
    seed,
):
    if dataset_picker.selected_key == "helpsteer2":
        validation_dataframe = datasets.load_dataset(
            "nvidia/HelpSteer2", split="validation"
        ).to_pandas()
        # The preference file labels this split "val", not "validation"
        preferences_dataframe = pd.read_json(
            "hf://datasets/nvidia/HelpSteer2/preference/preference.jsonl.gz",
            lines=True,
        )
        preferences_dataframe = preferences_dataframe[
            preferences_dataframe["split"] == "val"
        ]
        anchors = helpsteer2_anchors(
            validation_dataframe, preferences_dataframe, evaluation_prompt_count
        )
    else:
        pairs_dataframe = datasets.load_dataset(
            dataset_picker.value, split="train"
        ).to_pandas()
        held_out_prompts, _, _ = prompt_splits(
            pairs_dataframe["prompt"], evaluation_prompt_count, seed
        )
        # The split is a function of hub-dataset row order; if the pairs dataset
        # were re-pushed, checkpoints would silently be evaluated on prompts
        # from their own training half. 02_train-rms.py records the lists; the
        # split is deterministic, so before that upload exists the local one
        # is the same one 02 will record
        try:
            prompt_split = json.loads(
                model_path(dataset_picker.selected_key, "prompt_split.json").read_text()
            )
            assert held_out_prompts == prompt_split["held_out"]
        except (EntryNotFoundError, RepositoryNotFoundError):
            pass
        anchors = community_alignment_anchors(pairs_dataframe, held_out_prompts)
    evaluation_prompts = list(anchors)
    len(evaluation_prompts)
    return anchors, evaluation_prompts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 64 + 1 samples per prompt

    Each evaluation prompt gets 65 base-policy samples: indices 0-63 form the
    candidate pool the methods select from, and index 64 is reserved as the
    base-policy anchor for judging — it never enters the pool.
    """)
    return


@app.cell
def _(mo):
    generate_button = mo.ui.run_button(label="Generate candidates")
    generate_button
    return (generate_button,)


@app.cell
def _(
    Path,
    anchors,
    base_model_name,
    dataset_picker,
    evaluation_prompts,
    generate_button,
    generate_candidates,
    mo,
    pd,
    samples_per_prompt,
    seed,
    tempfile,
    upload_artifact,
):
    mo.stop(not generate_button.value)
    # One extra sample per prompt is reserved as the base-policy anchor
    candidates_dataframe = generate_candidates(
        base_model_name,
        evaluation_prompts,
        samples_per_prompt + 1,
        seed=seed,
    )
    candidates_dataframe["sample_index"] = candidates_dataframe.groupby(
        "prompt", sort=False
    ).cumcount()
    candidates_dataframe = candidates_dataframe[["prompt", "sample_index", "response"]]
    # Row order is the canonical evaluation-prompt order
    anchors_dataframe = pd.DataFrame(
        {
            "prompt": evaluation_prompts,
            "anchor": [anchors[prompt] for prompt in evaluation_prompts],
        }
    )
    with tempfile.TemporaryDirectory() as temp_name:
        temp_directory = Path(temp_name)
        candidates_dataframe.to_parquet(temp_directory / "candidates.parquet")
        anchors_dataframe.to_parquet(temp_directory / "anchors.parquet")
        upload_artifact(dataset_picker.selected_key, temp_directory / "candidates.parquet")
        upload_artifact(dataset_picker.selected_key, temp_directory / "anchors.parquet")
    candidates_dataframe
    return


if __name__ == "__main__":
    app.run()
