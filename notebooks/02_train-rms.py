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
#     # torch. Install a matching one so transformers doesn't import the
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
        bt_text,
        pairwise_text,
        save_reward_model,
        train_reward_model,
    )
    from blackwell_ita.utils.tokens import token_lengths

    return (
        BradleyTerryModel,
        PairwisePreferenceModel,
        bt_text,
        pairwise_text,
        prompt_splits,
        save_reward_model,
        token_lengths,
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
    # Community Alignment holds out 100 prompts as its evaluation set.
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
    bt_text,
    encoder_name,
    mo,
    pairwise_max_tokens,
    pairwise_text,
    preferences_dataframe,
    token_lengths,
):
    from transformers import AutoTokenizer

    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    prompt_response_triples = list(
        zip(
            preferences_dataframe["prompt"],
            preferences_dataframe["response_a"],
            preferences_dataframe["response_b"],
            strict=True,
        )
    )
    bt_truncation_rate = sum(
        length > bradley_terry_max_tokens
        for length in token_lengths(
            [
                bt_text(prompt, response)
                for prompt, first, second in prompt_response_triples
                for response in (first, second)
            ],
            encoder_tokenizer,
        )
    ) / (2 * len(prompt_response_triples))
    pairwise_truncation_rate = sum(
        length > pairwise_max_tokens
        for length in token_lengths(
            [
                pairwise_text(prompt, first, second)
                for prompt, first, second in prompt_response_triples
            ],
            encoder_tokenizer,
        )
    ) / len(prompt_response_triples)
    response_token_check = (
        mo.callout(
            f"{bt_truncation_rate:.1%} of Bradley-Terry inputs and "
            f"{pairwise_truncation_rate:.1%} of pairwise inputs exceed their "
            "token limits and will be truncated.",
            kind="warn",
        )
        if bt_truncation_rate > 0 or pairwise_truncation_rate > 0
        else mo.md("No inputs exceed their token limits.")
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
def _():
    def metrics_markdown(checkpoint_name, metrics):
        criterion_lines = "\n".join(
            f"- {criterion}: loss {values['loss']:.4f}, decisive accuracy "
            f"{values['decisive_accuracy']:.1%} over "
            f"{values['decisive_count']} entries"
            for criterion, values in metrics["criteria"].items()
        )
        return (
            f"Uploaded {checkpoint_name} "
            f"(validation loss {metrics['validation_loss']:.4f}).\n\n"
            f"{criterion_lines}"
        )

    return (metrics_markdown,)


@app.cell
def _(
    BradleyTerryModel,
    batch_size,
    bradley_terry_button,
    bradley_terry_max_tokens,
    criterion_columns,
    encoder_name,
    learning_rate,
    metrics_markdown,
    mo,
    optimisation_frame,
    save_and_upload,
    train_reward_model,
    warmup_steps,
):
    mo.stop(not bradley_terry_button.value)
    bradley_terry_model, bradley_terry_metrics = train_reward_model(
        optimisation_frame,
        criterion_columns,
        BradleyTerryModel,
        bradley_terry_max_tokens,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        # Order augmentation is a no-op for Bradley-Terry: BCE(-x, 1-t) = BCE(x, t)
        augment_presentation_order=False,
    )
    save_and_upload(bradley_terry_model, "bradley_terry.pt")
    mo.md(metrics_markdown("bradley_terry.pt", bradley_terry_metrics))
    return (bradley_terry_metrics,)


@app.cell
def _(
    PairwisePreferenceModel,
    batch_size,
    criterion_columns,
    encoder_name,
    learning_rate,
    metrics_markdown,
    mo,
    optimisation_frame,
    pairwise_button,
    pairwise_max_tokens,
    save_and_upload,
    train_reward_model,
    warmup_steps,
):
    mo.stop(not pairwise_button.value)
    pairwise_model, pairwise_metrics = train_reward_model(
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
    mo.md(metrics_markdown("pairwise.pt", pairwise_metrics))
    return (pairwise_metrics,)


@app.cell
def _(
    PairwisePreferenceModel,
    batch_size,
    criterion_columns,
    encoder_name,
    evaluation_frame,
    evaluation_pairwise_button,
    learning_rate,
    metrics_markdown,
    mo,
    pairwise_max_tokens,
    save_and_upload,
    train_reward_model,
    warmup_steps,
):
    mo.stop(not evaluation_pairwise_button.value)
    evaluation_pairwise_model, evaluation_pairwise_metrics = train_reward_model(
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
    mo.md(metrics_markdown("evaluation_pairwise.pt", evaluation_pairwise_metrics))
    return (evaluation_pairwise_metrics,)


@app.cell
def _(
    Path,
    bradley_terry_metrics,
    dataset_picker,
    evaluation_pairwise_metrics,
    evaluation_prompts,
    held_out_prompts,
    json,
    optimisation_prompts,
    pairwise_metrics,
    tempfile,
    upload_model,
):
    # Recorded so 03_generate-candidates.py can verify the held-out evaluation
    # prompts still match what these checkpoints were trained without. Uploaded
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
        "bradley_terry": bradley_terry_metrics,
        "pairwise": pairwise_metrics,
        "evaluation_pairwise": evaluation_pairwise_metrics,
    }
    return


if __name__ == "__main__":
    app.run()
