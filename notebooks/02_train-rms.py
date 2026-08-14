# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita",
#     "datasets",
#     "huggingface_hub",
#     "pandas",
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
    from blackwell_ita.artifacts import upload_model
    from blackwell_ita.human_prefs import prompt_splits
    from blackwell_ita.train_rms import (
        save_reward_model,
        train_pairwise_model,
        train_reward_models,
    )
    from blackwell_ita.utils.tokens import max_response_tokens

    return (
        max_response_tokens,
        prompt_splits,
        save_reward_model,
        train_pairwise_model,
        train_reward_models,
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
    encoder_name = "Qwen/Qwen2.5-3B"
    learning_rate = 1e-5
    batch_size = 16
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
    train_button = mo.ui.run_button(label="Train reward models")
    train_button
    return (train_button,)


@app.cell
def _(
    batch_size,
    bradley_terry_max_tokens,
    criterion_columns,
    encoder_name,
    evaluation_frame,
    learning_rate,
    mo,
    optimisation_frame,
    pairwise_max_tokens,
    train_button,
    train_pairwise_model,
    train_reward_models,
    warmup_steps,
):
    mo.stop(not train_button.value)
    bradley_terry_model, pairwise_model, validation_losses = train_reward_models(
        optimisation_frame,
        criterion_columns,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        bradley_terry_max_tokens=bradley_terry_max_tokens,
        pairwise_max_tokens=pairwise_max_tokens,
        warmup_steps=warmup_steps,
    )
    evaluation_pairwise_model, evaluation_validation_loss = train_pairwise_model(
        evaluation_frame,
        criterion_columns,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        pairwise_max_tokens=pairwise_max_tokens,
        warmup_steps=warmup_steps,
    )
    {**validation_losses, "evaluation_pairwise": evaluation_validation_loss}
    return bradley_terry_model, evaluation_pairwise_model, pairwise_model


@app.cell
def _(
    Path,
    bradley_terry_model,
    criterion_columns,
    dataset_picker,
    encoder_name,
    evaluation_pairwise_model,
    evaluation_prompts,
    held_out_prompts,
    json,
    mo,
    optimisation_prompts,
    pairwise_model,
    save_reward_model,
    tempfile,
    upload_model,
):
    checkpoint_files = [
        "bradley_terry.pt",
        "pairwise.pt",
        "evaluation_pairwise.pt",
        "prompt_split.json",
    ]
    with tempfile.TemporaryDirectory() as temp_name:
        temp_directory = Path(temp_name)
        save_reward_model(
            bradley_terry_model,
            criterion_columns,
            encoder_name,
            temp_directory / "bradley_terry.pt",
        )
        save_reward_model(
            pairwise_model,
            criterion_columns,
            encoder_name,
            temp_directory / "pairwise.pt",
        )
        save_reward_model(
            evaluation_pairwise_model,
            criterion_columns,
            encoder_name,
            temp_directory / "evaluation_pairwise.pt",
        )
        # Recorded so 03_generate-candidates.py can verify the held-out evaluation
        # prompts still match what these checkpoints were trained without
        (temp_directory / "prompt_split.json").write_text(
            json.dumps(
                {
                    "held_out": held_out_prompts,
                    "optimisation": optimisation_prompts,
                    "evaluation": evaluation_prompts,
                }
            )
        )
        for checkpoint_file in checkpoint_files:
            upload_model(dataset_picker.selected_key, temp_directory / checkpoint_file)
    mo.md(f"Uploaded {', '.join(checkpoint_files)} for {dataset_picker.selected_key}.")
    return


if __name__ == "__main__":
    app.run()
