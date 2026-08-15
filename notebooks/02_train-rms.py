# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita",
#     "datasets",
#     "huggingface_hub",
#     "pandas",
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
    from blackwell_ita.artifacts import upload_model
    from blackwell_ita.human_prefs import prompt_splits
    from blackwell_ita.train_rms import (
        BradleyTerryModel,
        PairwisePreferenceModel,
        save_reward_model,
        train_reward_model,
    )
    from blackwell_ita.utils.tokens import max_response_tokens

    return (
        BradleyTerryModel,
        PairwisePreferenceModel,
        max_response_tokens,
        prompt_splits,
        save_reward_model,
        train_reward_model,
        upload_model,
    )


@app.cell
def _():
    import json
    import tempfile
    from pathlib import Path

    import datasets

    return Path, datasets, json, tempfile


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
def _(dataset_picker, datasets):
    preferences_dataframe = datasets.load_dataset(
        dataset_picker.value, split="train"
    ).to_pandas()
    preferences_dataframe
    return (preferences_dataframe,)


@app.cell
def _(preferences_dataframe):
    criterion_columns = sorted(
        column
        for column in preferences_dataframe.columns
        if column not in ("prompt", "response_a", "response_b")
    )
    criterion_columns
    return (criterion_columns,)


@app.cell
def _(dataset_picker):
    encoder_name = "Qwen/Qwen3-4B-Instruct-2507"
    learning_rate = 1e-5
    batch_size = 12
    bradley_terry_max_tokens = 2000
    pairwise_max_tokens = 4000
    warmup_steps = 100
    seed = 1810
    # Community Alignment holds out 100 prompts as its evaluation set;
    # HelpSteer2's evaluation prompts come from the validation split instead
    evaluation_count = 100 if dataset_picker.selected_key == "community-alignment" else 0
    return (
        batch_size,
        bradley_terry_max_tokens,
        encoder_name,
        evaluation_count,
        learning_rate,
        pairwise_max_tokens,
        seed,
        warmup_steps,
    )


@app.cell
def _(
    bradley_terry_max_tokens,
    encoder_name,
    max_response_tokens,
    mo,
    preferences_dataframe,
):
    from transformers import AutoTokenizer

    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    longest_response_tokens = max(
        max_response_tokens(preferences_dataframe["response_a"], encoder_tokenizer),
        max_response_tokens(preferences_dataframe["response_b"], encoder_tokenizer),
    )
    response_token_check = (
        mo.callout(
            f"Longest response is {longest_response_tokens} tokens, above "
            f"bradley_terry_max_tokens={bradley_terry_max_tokens}; "
            "model inputs will be truncated.",
            kind="warn",
        )
        if longest_response_tokens > bradley_terry_max_tokens
        else mo.md(f"Longest response: {longest_response_tokens} tokens.")
    )
    response_token_check
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split-half training

    The training prompts are halved: the optimisation models
    (`bradley_terry.pt`, `pairwise.pt`) train on the first half's pairs and the
    evaluation model (`evaluation_pairwise.pt`) on the second half's. The
    evaluator never sees a prompt the optimisation models trained on, so it can
    act as a fair referee for the responses they later select.
    """)
    return


@app.cell
def _(evaluation_count, preferences_dataframe, prompt_splits, seed):
    held_out_prompts, optimisation_prompts, evaluation_prompts = prompt_splits(
        preferences_dataframe["prompt"], evaluation_count, seed
    )
    optimisation_frame = preferences_dataframe[
        preferences_dataframe["prompt"].isin(set(optimisation_prompts))
    ]
    evaluation_frame = preferences_dataframe[
        preferences_dataframe["prompt"].isin(set(evaluation_prompts))
    ]
    return (
        evaluation_frame,
        evaluation_prompts,
        held_out_prompts,
        optimisation_frame,
        optimisation_prompts,
    )


@app.cell
def _(mo):
    bradley_terry_button = mo.ui.run_button(label="Train bradley_terry")
    pairwise_button = mo.ui.run_button(label="Train pairwise")
    evaluation_pairwise_button = mo.ui.run_button(label="Train evaluation_pairwise")
    mo.hstack(
        [bradley_terry_button, pairwise_button, evaluation_pairwise_button],
        justify="start",
    )
    return (bradley_terry_button, evaluation_pairwise_button, pairwise_button)


@app.cell
def _(
    Path,
    criterion_columns,
    dataset_picker,
    encoder_name,
    save_reward_model,
    tempfile,
    upload_model,
):
    def save_and_upload(model, checkpoint_name):
        with tempfile.TemporaryDirectory() as temp_name:
            checkpoint_path = Path(temp_name) / checkpoint_name
            save_reward_model(
                model, criterion_columns, encoder_name, checkpoint_path
            )
            upload_model(dataset_picker.selected_key, checkpoint_path)

    return (save_and_upload,)


@app.cell
def _(
    BradleyTerryModel,
    batch_size,
    bradley_terry_button,
    bradley_terry_max_tokens,
    criterion_columns,
    encoder_name,
    learning_rate,
    mo,
    optimisation_frame,
    save_and_upload,
    train_reward_model,
    warmup_steps,
):
    mo.stop(not bradley_terry_button.value)
    bradley_terry_model, bradley_terry_loss = train_reward_model(
        optimisation_frame,
        criterion_columns,
        BradleyTerryModel,
        bradley_terry_max_tokens,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
    )
    save_and_upload(bradley_terry_model, "bradley_terry.pt")
    mo.md(f"Uploaded bradley_terry.pt (validation loss {bradley_terry_loss:.4f}).")
    return (bradley_terry_loss,)


@app.cell
def _(
    PairwisePreferenceModel,
    batch_size,
    criterion_columns,
    encoder_name,
    learning_rate,
    mo,
    optimisation_frame,
    pairwise_button,
    pairwise_max_tokens,
    save_and_upload,
    train_reward_model,
    warmup_steps,
):
    mo.stop(not pairwise_button.value)
    pairwise_model, pairwise_loss = train_reward_model(
        optimisation_frame,
        criterion_columns,
        PairwisePreferenceModel,
        pairwise_max_tokens,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
    )
    save_and_upload(pairwise_model, "pairwise.pt")
    mo.md(f"Uploaded pairwise.pt (validation loss {pairwise_loss:.4f}).")
    return (pairwise_loss,)


@app.cell
def _(
    PairwisePreferenceModel,
    batch_size,
    criterion_columns,
    encoder_name,
    evaluation_frame,
    evaluation_pairwise_button,
    learning_rate,
    mo,
    pairwise_max_tokens,
    save_and_upload,
    train_reward_model,
    warmup_steps,
):
    mo.stop(not evaluation_pairwise_button.value)
    evaluation_pairwise_model, evaluation_pairwise_loss = train_reward_model(
        evaluation_frame,
        criterion_columns,
        PairwisePreferenceModel,
        pairwise_max_tokens,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
    )
    save_and_upload(evaluation_pairwise_model, "evaluation_pairwise.pt")
    mo.md(
        f"Uploaded evaluation_pairwise.pt "
        f"(validation loss {evaluation_pairwise_loss:.4f})."
    )
    return (evaluation_pairwise_loss,)


@app.cell
def _(
    Path,
    bradley_terry_loss,
    dataset_picker,
    evaluation_pairwise_loss,
    evaluation_prompts,
    held_out_prompts,
    json,
    optimisation_prompts,
    pairwise_loss,
    tempfile,
    upload_model,
):
    # Recorded so 03_generate-candidates.py can verify the held-out evaluation
    # prompts still match what these checkpoints were trained without; uploaded
    # last so its presence marks a complete, consistent checkpoint set
    with tempfile.TemporaryDirectory() as split_temp_name:
        split_path = Path(split_temp_name) / "prompt_split.json"
        split_path.write_text(
            json.dumps(
                {
                    "held_out": held_out_prompts,
                    "optimisation": optimisation_prompts,
                    "evaluation": evaluation_prompts,
                }
            )
        )
        upload_model(dataset_picker.selected_key, split_path)
    {
        "bradley_terry": bradley_terry_loss,
        "pairwise": pairwise_loss,
        "evaluation_pairwise": evaluation_pairwise_loss,
    }
    return


if __name__ == "__main__":
    app.run()
