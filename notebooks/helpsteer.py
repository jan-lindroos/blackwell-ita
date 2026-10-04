# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "blackwell-ita @ git+https://github.com/jan-lindroos/blackwell-ita@ablation",
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
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import torch

    from blackwell_ita.data import (
        assign_prompt_splits,
        load_helpsteer2_pairs,
        split_summary,
    )
    from blackwell_ita.helpsteer import (
        GENERATION_BACKBONES,
        GENERATION_TEMPERATURE,
        HELPSTEER_CRITERIA,
        HELPSTEER_POOL_SIZE,
        HELPSTEER_PREFIX,
        generate_helpsteer_candidates,
        helpsteer_prompts,
        pool_prefix,
        score_helpsteer_pools,
        validate_prompts,
    )
    from blackwell_ita.helpsteer_cli import (
        compare_best_of_n,
        judge_prefix,
    )
    from blackwell_ita.helpsteer_cli import (
        run as run_claude_stage,
    )
    from blackwell_ita.hub import (
        ARTIFACTS_REPOSITORY,
        PAIRS_FILENAME,
        SPLITS_REPOSITORY,
        ensure_hub_dataframe,
        hub_file_exists,
        read_hub_dataframe,
    )
    from blackwell_ita.training import REWARD_MODEL_CONFIGURATIONS, ensure_trained

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## HelpSteer: Claude-supervised target sets

    Train/load reward models, generate 64 responses per prompt, then score on the
    notebook GPU. There are 50 tuning prompts and 200 evaluation prompts, all
    drawn from the reward-model test partition. Use the local Claude section at
    the bottom for judging, tuning and evaluation. Set `HF_TOKEN` and `WANDB_API_KEY`
    for GPU work. The new experiment prefix isolates these splits and checkpoints
    from previous pilots.
    """)
    return


@app.cell
def _():
    train_checkboxes = mo.ui.dictionary(
        {
            configuration.name: mo.ui.checkbox(label=f"Train/load {configuration.name}")
            for configuration in REWARD_MODEL_CONFIGURATIONS
        }
    )
    generate_checkboxes = mo.ui.dictionary(
        {
            backbone: mo.ui.checkbox(label=f"Generate/load {backbone.split('/')[-1]}")
            for backbone in GENERATION_BACKBONES
        }
    )
    score_button = mo.ui.run_button(label="Score selected pools")
    mo.vstack(
        [
            mo.hstack(list(train_checkboxes.values()), justify="start"),
            mo.hstack(list(generate_checkboxes.values()), justify="start"),
            score_button,
        ]
    )
    return generate_checkboxes, score_button, train_checkboxes


@app.cell
def _():
    pairs = ensure_hub_dataframe(
        SPLITS_REPOSITORY,
        PAIRS_FILENAME,
        HELPSTEER_PREFIX,
        lambda: assign_prompt_splits(load_helpsteer2_pairs()),
    )
    split_summary(pairs)
    return (pairs,)


@app.cell
def _(pairs, train_checkboxes):
    mo.stop(not any(train_checkboxes.value.values()))
    trained_configurations = [
        configuration
        for configuration in REWARD_MODEL_CONFIGURATIONS
        if train_checkboxes.value[configuration.name]
    ]
    reward_model_metrics = pd.concat(
        [
            ensure_trained(
                configuration, pairs, HELPSTEER_CRITERIA, HELPSTEER_PREFIX, DEVICE
            )
            for configuration in trained_configurations
        ],
        ignore_index=True,
    )
    reward_model_metrics.pivot_table(
        index=["split", "criterion"], columns="name", values="decisive_accuracy"
    )
    return (trained_configurations,)


@app.cell
def _(pairs):
    prompts = ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "prompts.parquet",
        HELPSTEER_PREFIX,
        lambda: helpsteer_prompts(pairs),
    )
    validate_prompts(prompts, pairs)
    prompts.groupby("split").size().rename("prompts")
    return (prompts,)


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Generate and score

    Each pool contains exactly **{HELPSTEER_POOL_SIZE} generated responses** at
    **temperature {GENERATION_TEMPERATURE}**, with a 1024-token generation cap.
    No lexical filtering or larger reservoir is used. Higher temperature is an
    experimental diversity setting, not a guarantee of higher quality.

    The human-preferred anchor is appended for scoring and excluded from selection.
    Pairwise models store preference tensors; BT stores raw per-criterion scalar
    rewards, from which both genuine best-of-N and Blackwell policies are computed.
    Completed artifacts are cached on Hugging Face; scoring resumes per prompt.
    """)
    return


@app.cell
def _(generate_checkboxes, prompts):
    mo.stop(not any(generate_checkboxes.value.values()))
    candidates_by_backbone = {
        backbone: ensure_hub_dataframe(
            ARTIFACTS_REPOSITORY,
            "candidates.parquet",
            pool_prefix(backbone),
            lambda backbone=backbone: generate_helpsteer_candidates(
                backbone, prompts, DEVICE
            ),
        )
        for backbone, selected in generate_checkboxes.value.items()
        if selected
    }
    pd.DataFrame(
        [
            {"backbone": name, "responses": len(frame)}
            for name, frame in candidates_by_backbone.items()
        ]
    )
    return (candidates_by_backbone,)


@app.cell
def _(candidates_by_backbone, prompts, score_button, trained_configurations):
    mo.stop(not score_button.value)
    scoring_summary = []
    for _backbone, candidates in candidates_by_backbone.items():
        for _configuration in trained_configurations:
            scores = score_helpsteer_pools(
                _configuration, _backbone, prompts, candidates, DEVICE
            )
            scoring_summary.append(
                {
                    "backbone": _backbone,
                    "selector": _configuration.name,
                    "scored_prompts": len(scores),
                }
            )
    pd.DataFrame(scoring_summary)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Local Claude judging and tuning

    After GPU scoring finishes, open this notebook on the machine authenticated
    with the Claude CLI, with `HF_TOKEN` set. Leave the GPU checkboxes above off.
    Choose the already-scored pools and selectors below, enter an explicit Claude
    model ID, then click **Tune and evaluate with local Claude**.

    The button downloads the saved artifacts, tunes all selected configurations,
    then evaluates them sequentially. It does not load reward models or generate
    responses. Claude runs on the machine hosting the notebook kernel.

    Tuning judges every candidate on the 50 tuning prompts, then freezes the
    target/weights. Target weights and thresholds start with a coarse search,
    then refine in 0.125 increments within [0.25, 0.75] until a full sweep finds
    no improvement. Refinement reuses the same Claude judgments.
    Evaluation judges only the union of selected supports on the
    other 200 prompts. Judgments are shared across selectors and both response
    orders are retained. Interrupted runs resume; failures never count as ties.
    Claude judges all four criteria, so welfare also uses Claude rather than a
    model proxy. Results appear below when the run finishes. To view an existing
    run, enter its model ID and click **Load results**, without running Claude.
    """)
    return


@app.cell
def _():
    local_backbones = mo.ui.multiselect(
        options=GENERATION_BACKBONES,
        value=["RLHFlow/LLaMA3-SFT-v2"],
        label="Scored generators",
    )
    local_selectors = mo.ui.multiselect(
        options=[configuration.name for configuration in REWARD_MODEL_CONFIGURATIONS],
        value=["qwen3_4b_pairwise", "qwen3_4b_bradley_terry"],
        label="Scored selectors",
    )
    judge_model = mo.ui.text(label="Explicit Claude model ID")
    claude_button = mo.ui.run_button(label="Tune and evaluate with local Claude")
    results_button = mo.ui.run_button(label="Load results")
    mo.vstack(
        [local_backbones, local_selectors, judge_model, claude_button, results_button]
    )
    return claude_button, judge_model, local_backbones, local_selectors, results_button


@app.cell
def _(claude_button, judge_model, local_backbones, local_selectors):
    local_run_completed = False
    if claude_button.value:
        mo.stop(not judge_model.value.strip(), mo.md("Enter a Claude model ID."))
        mo.stop(
            not local_backbones.value or not local_selectors.value,
            mo.md("Select at least one scored generator and selector."),
        )
        for _stage in ["tune", "evaluate"]:
            for _backbone in local_backbones.value:
                for _selector in local_selectors.value:
                    run_claude_stage(
                        _stage,
                        _backbone,
                        _selector,
                        judge_model.value.strip(),
                        Path(".helpsteer"),
                    )
        local_run_completed = True
    return (local_run_completed,)


@app.cell
def _(
    judge_model, local_backbones, local_run_completed, local_selectors, results_button
):
    mo.stop(not (local_run_completed or results_button.value))
    mo.stop(not judge_model.value.strip())
    summaries = []
    prompt_results = []
    for _backbone in local_backbones.value:
        for _configuration in REWARD_MODEL_CONFIGURATIONS:
            if _configuration.name not in local_selectors.value:
                continue
            filename = f"{_configuration.name}_summary.parquet"
            prefix = judge_prefix(_backbone, judge_model.value.strip())
            if hub_file_exists(ARTIFACTS_REPOSITORY, filename, prefix):
                summaries.append(
                    read_hub_dataframe(ARTIFACTS_REPOSITORY, filename, prefix).assign(
                        backbone=_backbone, selector=_configuration.name
                    )
                )
                prompt_results.append(
                    read_hub_dataframe(
                        ARTIFACTS_REPOSITORY,
                        f"{_configuration.name}_results.parquet",
                        prefix,
                    ).assign(backbone=_backbone, selector=_configuration.name)
                )
    mo.stop(
        not summaries,
        mo.md("No completed local evaluation uploaded for these selections."),
    )
    results_summary = pd.concat(summaries, ignore_index=True)
    results_by_prompt = pd.concat(prompt_results, ignore_index=True)
    results_summary.pivot_table(
        index=["backbone", "selector", "metric"], columns="method", values="mean"
    )
    return results_by_prompt, results_summary


@app.cell
def _(results_summary):
    results_summary[
        [
            "backbone",
            "selector",
            "metric",
            "method",
            "prompts",
            "learned_minus_method",
            "interval_low",
            "interval_high",
        ]
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Direct comparisons with scalar best-of-N

    Select both pairwise and BT configurations in the local controls. These paired differences
    compare every Blackwell variant against each scalar BT baseline on the same
    evaluation prompts. Positive values favour Blackwell; intervals resample
    prompts. No cross-scorer comparison appears until both results are available.
    """)
    return


@app.cell
def _(results_by_prompt):
    compare_best_of_n(results_by_prompt)
    return


if __name__ == "__main__":
    app.run()
