# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita",
#     "huggingface_hub",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scipy",
#     "torch",
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
        Before running, switch to GPU. Then sign in with `hf auth login` (open terminal available via command menu) and add `blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita` to dependencies via uv (in the sidebar, in dependencies).
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from blackwell_ita.artifacts import artifact_path, model_path, upload_artifact
    from blackwell_ita.train_rms import (
        default_device,
        load_reward_model,
        pairwise_text,
    )
    from blackwell_ita.winners import preference_tensor, reward_scores

    return (
        artifact_path,
        default_device,
        load_reward_model,
        model_path,
        pairwise_text,
        preference_tensor,
        reward_scores,
        upload_artifact,
    )


@app.cell
def _():
    import json
    import tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch

    return Path, json, np, pd, tempfile, torch


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
    samples_per_prompt = 64
    return (samples_per_prompt,)


@app.cell
def _(artifact_path, dataset_picker, pd):
    candidates_dataframe = pd.read_parquet(
        artifact_path(dataset_picker.selected_key, "candidates.parquet")
    )
    anchors_dataframe = pd.read_parquet(
        artifact_path(dataset_picker.selected_key, "anchors.parquet")
    )
    anchors_dataframe
    return anchors_dataframe, candidates_dataframe


@app.cell
def _(dataset_picker, default_device, load_reward_model, model_path):
    device = default_device()
    bradley_terry_model, criterion_columns = load_reward_model(
        model_path(dataset_picker.selected_key, "bradley_terry.pt"), device
    )
    pairwise_model, pairwise_columns = load_reward_model(
        model_path(dataset_picker.selected_key, "pairwise.pt"), device
    )
    # The cached tensors' head order is criterion_columns; if the two
    # checkpoints were ever retrained separately with different column orders,
    # "overall" would silently become a different criterion
    assert criterion_columns == pairwise_columns
    evaluation_model, evaluation_columns = load_reward_model(
        model_path(dataset_picker.selected_key, "evaluation_pairwise.pt"), device
    )
    evaluation_criteria = [
        column for column in evaluation_columns if column != "overall"
    ]
    evaluation_criteria
    return (
        bradley_terry_model,
        criterion_columns,
        device,
        evaluation_columns,
        evaluation_model,
        pairwise_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cached reward-model inference

    This notebook runs every reward-model forward pass the experiments need and
    caches the results on the hub, so the local experiments need no GPU.
    """)
    return


@app.cell
def _(mo):
    score_button = mo.ui.run_button(label="Score with reward models")
    score_button
    return (score_button,)


@app.cell
def _(
    Path,
    anchors_dataframe,
    bradley_terry_model,
    candidates_dataframe,
    criterion_columns,
    dataset_picker,
    device,
    evaluation_columns,
    evaluation_model,
    json,
    mo,
    np,
    pairwise_model,
    pairwise_text,
    pd,
    preference_tensor,
    reward_scores,
    samples_per_prompt,
    score_button,
    tempfile,
    torch,
    upload_artifact,
):
    mo.stop(not score_button.value)
    prompts = anchors_dataframe["prompt"].tolist()
    score_arrays = {}
    criterion_rows = []
    for prompt_index, prompt in enumerate(mo.status.progress_bar(prompts)):
        prompt_candidates = candidates_dataframe[
            candidates_dataframe["prompt"] == prompt
        ].sort_values("sample_index")
        pool = prompt_candidates[
            prompt_candidates["sample_index"] < samples_per_prompt
        ]["response"].tolist()
        anchor = anchors_dataframe["anchor"].iloc[prompt_index]
        score_arrays[f"tensor_{prompt_index}"] = preference_tensor(
            pairwise_model, prompt, pool, device
        )
        score_arrays[f"rewards_{prompt_index}"] = reward_scores(
            bradley_terry_model, prompt, pool, device
        )
        # Evaluation-model P(candidate beats reference anchor | criterion),
        # averaged over both presentation orders
        for sample_index, response in enumerate(pool):
            forward_text = pairwise_text(prompt, response, anchor)
            backward_text = pairwise_text(prompt, anchor, response)
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    evaluation_model.score([forward_text, backward_text], device)
                )
            anchor_wins = (
                (probabilities[0] + 1.0 - probabilities[1]).cpu().numpy() / 2.0
            )
            criterion_rows.extend(
                {
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "response": response,
                    "criterion": criterion,
                    "score": float(anchor_wins[column_index]),
                }
                for column_index, criterion in enumerate(evaluation_columns)
                if criterion != "overall"
            )
    criterion_scores_dataframe = pd.DataFrame(criterion_rows)
    score_files = [
        "model_scores.npz",
        "model_scores_meta.json",
        "criterion_anchor_scores.parquet",
    ]
    with tempfile.TemporaryDirectory() as temp_name:
        temp_directory = Path(temp_name)
        np.savez(temp_directory / "model_scores.npz", **score_arrays)
        (temp_directory / "model_scores_meta.json").write_text(
            json.dumps({"prompts": prompts, "criterion_columns": criterion_columns})
        )
        criterion_scores_dataframe.to_parquet(
            temp_directory / "criterion_anchor_scores.parquet"
        )
        for score_file in score_files:
            upload_artifact(dataset_picker.selected_key, temp_directory / score_file)
    criterion_scores_dataframe
    return


if __name__ == "__main__":
    app.run()
