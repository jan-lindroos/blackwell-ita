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
    import functools
    from itertools import permutations

    import marimo as mo
    import pandas as pd
    import torch

    from blackwell_ita.data import assign_prompt_splits, split_summary
    from blackwell_ita.generation import generate_responses
    from blackwell_ita.habermas import (
        GENERATION_BACKBONES,
        HABERMAS_CONFIGURATIONS,
        HABERMAS_CRITERIA,
        HABERMAS_PREFIX,
        candidate_sets,
        consensus_prompt,
        deliberation_groups,
        habermas_pairs,
        human_preference_tensor,
        human_track_results,
        load_habermas_rankings,
        model_track_results,
        participant_preference_tensor,
    )
    from blackwell_ita.hub import ARTIFACTS_REPOSITORY, ensure_hub_dataframe
    from blackwell_ita.scoring import score_with_resume
    from blackwell_ita.selection import summarise_methods
    from blackwell_ita.training import ensure_trained, load_trained

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLES_PER_GROUP = 64
    WELFARE_NAMES = ["rawlsian_welfare", "nash_welfare", "utilitarian_welfare"]


@app.function
def score_participant_tensors(configuration, filename: str, keyed_inputs: dict) -> dict:
    """Predicted participant tensors per key, loading the model only if needed."""
    load_model = functools.cache(
        lambda: load_trained(configuration, HABERMAS_PREFIX, DEVICE)
    )
    predicted_tensors = score_with_resume(
        ARTIFACTS_REPOSITORY,
        filename,
        HABERMAS_PREFIX,
        keyed_inputs,
        lambda inputs: participant_preference_tensor(
            load_model(), device=DEVICE, **inputs
        ),
    )
    load_model.cache_clear()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return predicted_tensors


@app.cell
def _():
    train_checkboxes = mo.ui.dictionary(
        {
            configuration.name: mo.ui.checkbox(label=f"Train {configuration.name}")
            for configuration in HABERMAS_CONFIGURATIONS
        }
    )
    generate_checkboxes = mo.ui.dictionary(
        {
            backbone_name: mo.ui.checkbox(
                label=f"Generate {backbone_name.split('/')[-1]}"
            )
            for backbone_name in GENERATION_BACKBONES
        }
    )
    pool_size_slider = mo.ui.slider(steps=[16, 64], label="Pool size", show_value=True)
    mo.hstack(
        [
            *train_checkboxes.values(),
            *generate_checkboxes.values(),
            pool_size_slider,
        ],
        justify="start",
    )
    return generate_checkboxes, pool_size_slider, train_checkboxes


@app.cell
def _():
    rankings = ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "rankings.parquet",
        HABERMAS_PREFIX,
        lambda: assign_prompt_splits(load_habermas_rankings(), "question_id"),
    )
    pairs = habermas_pairs(rankings)
    split_summary(pairs, "question_id")
    return pairs, rankings


@app.cell
def _(pairs, train_checkboxes):
    mo.stop(not any(train_checkboxes.value.values()))
    trained_configurations = [
        configuration
        for configuration in HABERMAS_CONFIGURATIONS
        if train_checkboxes.value[configuration.name]
    ]
    preference_model_metrics = pd.concat(
        [
            ensure_trained(
                configuration, pairs, HABERMAS_CRITERIA, HABERMAS_PREFIX, DEVICE
            )
            for configuration in trained_configurations
        ],
        ignore_index=True,
    )
    preference_model_metrics.pivot_table(
        index=["split", "criterion"], columns="name", values="decisive_accuracy"
    )
    return (trained_configurations,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selection from human preferences
    """)
    return


@app.cell
def _(rankings):
    human_track_sets = candidate_sets(rankings, "test")
    human_track_results_frame = human_track_results(
        human_track_sets,
        {
            set_row["set_id"]: human_preference_tensor(set_row["participant_ranks"])
            for set_row in human_track_sets.to_dict("records")
        },
        "human preferences",
    )
    summarise_methods(
        human_track_results_frame,
        ["evidence", "selector"],
        "set_id",
        [
            f"{measure}_{welfare_name}"
            for measure in ("win_rate", "agreement")
            for welfare_name in WELFARE_NAMES
        ],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reward model selection (larger sample)

    Generated statements have no human ratings, so every number here is a
    model-based proxy graded by the preference model that did not select.
    """)
    return


@app.cell
def _(rankings):
    deliberation_groups_frame = ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "groups.parquet",
        HABERMAS_PREFIX,
        lambda: deliberation_groups(rankings, "test"),
    )
    deliberation_groups_frame
    return (deliberation_groups_frame,)


@app.cell
def _(deliberation_groups_frame, generate_checkboxes):
    mo.stop(not any(generate_checkboxes.value.values()))
    generation_prompts = [
        consensus_prompt(group_row["question"], list(group_row["opinions"]))
        for group_row in deliberation_groups_frame.to_dict("records")
    ]
    candidates_by_backbone = {
        backbone_name: ensure_hub_dataframe(
            ARTIFACTS_REPOSITORY,
            f"candidates_{backbone_name.split('/')[-1].lower()}.parquet",
            HABERMAS_PREFIX,
            lambda backbone_name=backbone_name: generate_responses(
                backbone_name, generation_prompts, SAMPLES_PER_GROUP, DEVICE
            ).assign(
                question_id=lambda responses: deliberation_groups_frame[
                    "question_id"
                ].to_numpy()[responses["prompt_index"]]
            ),
        )
        for backbone_name, selected in generate_checkboxes.value.items()
        if selected
    }
    return (candidates_by_backbone,)


@app.cell
def _(
    candidates_by_backbone,
    deliberation_groups_frame,
    pool_size_slider,
    trained_configurations,
):
    # Each preference model grades the other's selections
    mo.stop(len(trained_configurations) < 2)
    model_track_frames = []
    for backbone_name, backbone_candidates in candidates_by_backbone.items():
        backbone_slug = backbone_name.split("/")[-1].lower()
        # The anchor rides at the last index of every pool
        pool_inputs = {
            group_row["question_id"]: {
                "question": group_row["question"],
                "opinions": list(group_row["opinions"]),
                "statements": backbone_candidates[
                    (backbone_candidates["question_id"] == group_row["question_id"])
                    & (backbone_candidates["sample_index"] < pool_size_slider.value)
                ]
                .sort_values("sample_index")["response"]
                .tolist()
                + [group_row["anchor"]],
            }
            for group_row in deliberation_groups_frame.to_dict("records")
        }
        tensors_by_model = {
            configuration.name: score_participant_tensors(
                configuration,
                f"model_track_{backbone_slug}_{configuration.name}_pool{pool_size_slider.value}.npz",
                pool_inputs,
            )
            for configuration in trained_configurations
        }
        for selector_name, grader_name in permutations(tensors_by_model, 2):
            model_track_frames.append(
                model_track_results(
                    deliberation_groups_frame,
                    tensors_by_model[selector_name],
                    tensors_by_model[grader_name],
                    backbone_name,
                    selector_name,
                    grader_name,
                )
            )
    model_track_results_frame = pd.concat(model_track_frames, ignore_index=True)
    summarise_methods(
        model_track_results_frame,
        ["evidence", "backbone", "selector"],
        "question_id",
        [
            f"{opponent}_{welfare_name}"
            for opponent in ("anchor", "pool")
            for welfare_name in WELFARE_NAMES
        ],
    )
    return


if __name__ == "__main__":
    app.run()
