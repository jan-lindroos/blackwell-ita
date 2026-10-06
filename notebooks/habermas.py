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
    from blackwell_ita.habermas import (
        GENERATION_BACKBONES,
        HABERMAS_CONFIGURATIONS,
        HABERMAS_CRITERIA,
        HABERMAS_POOL_SIZE,
        HABERMAS_PREFIX,
        deliberation_groups,
        generate_habermas_candidates,
        habermas_pairs,
        load_habermas_rankings,
        model_track_results,
        participant_preference_tensor,
        pool_difference_summary,
    )
    from blackwell_ita.habermas_views import (
        explorer_controls,
        method_matrix_heatmap,
        question_view,
    )
    from blackwell_ita.hub import (
        ARTIFACTS_REPOSITORY,
        REWARD_MODELS_REPOSITORY,
        ensure_hub_dataframe,
        hub_file_exists,
    )
    from blackwell_ita.scoring import download_tensors, score_with_resume
    from blackwell_ita.selection import summarise_methods
    from blackwell_ita.training import ensure_trained, load_trained

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WELFARE_NAMES = ["egalitarian_welfare", "nash_welfare", "utilitarian_welfare"]
    METRIC_NAMES = [f"pool_{welfare_name}" for welfare_name in WELFARE_NAMES]
    STRATEGY = "diverse_v2"


@app.function
def score_participant_tensors(configuration, filename: str, keyed_inputs: dict) -> dict:
    """Predicted participant tensors per key, loading the model only if needed.

    On CPU nothing is scored, only the keys already on the hub come back.
    """
    if DEVICE == "cpu":
        cached_tensors = download_tensors(
            ARTIFACTS_REPOSITORY, filename, HABERMAS_PREFIX
        )
        return {
            key: cached_tensors[key] for key in keyed_inputs if key in cached_tensors
        }
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
def _(pairs):
    trained_configurations = [
        configuration
        for configuration in HABERMAS_CONFIGURATIONS
        if DEVICE != "cpu"
        or hub_file_exists(
            REWARD_MODELS_REPOSITORY, configuration.checkpoint_filename, HABERMAS_PREFIX
        )
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
def _(deliberation_groups_frame):
    candidate_filenames = {
        backbone_name: f"candidates_local_v2_{backbone_name.split('/')[-1].lower()}_{STRATEGY}.parquet"
        for backbone_name in GENERATION_BACKBONES
    }
    candidates_by_backbone = {
        backbone_name: ensure_hub_dataframe(
            ARTIFACTS_REPOSITORY,
            candidate_filename,
            HABERMAS_PREFIX,
            lambda backbone_name=backbone_name: generate_habermas_candidates(
                backbone_name, deliberation_groups_frame, DEVICE, strategy=STRATEGY
            ),
        )
        for backbone_name, candidate_filename in candidate_filenames.items()
        if DEVICE != "cpu"
        or hub_file_exists(ARTIFACTS_REPOSITORY, candidate_filename, HABERMAS_PREFIX)
    }
    return (candidates_by_backbone,)


@app.cell
def _(
    candidates_by_backbone,
    deliberation_groups_frame,
    trained_configurations,
):
    # Each preference model grades the other's selections
    model_track_frames = []
    tensors_by_backbone = {}
    for backbone_name, backbone_candidates in candidates_by_backbone.items():
        backbone_slug = backbone_name.split("/")[-1].lower()
        # The anchor rides at the last index of every pool
        pool_inputs = {
            group_row["question_id"]: {
                "question": group_row["question"],
                "opinions": list(group_row["opinions"]),
                "statements": backbone_candidates[
                    (backbone_candidates["question_id"] == group_row["question_id"])
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
                f"model_track_local_v2_{backbone_slug}_{STRATEGY}_{configuration.name}_pool{HABERMAS_POOL_SIZE}.npz",
                pool_inputs,
            )
            for configuration in trained_configurations
        }
        tensors_by_backbone[backbone_name] = tensors_by_model
        for selector_name, grader_name in permutations(tensors_by_model, 2):
            scored_question_ids = (
                tensors_by_model[selector_name].keys()
                & tensors_by_model[grader_name].keys()
            )
            model_track_frames.append(
                model_track_results(
                    deliberation_groups_frame[
                        deliberation_groups_frame["question_id"].isin(
                            scored_question_ids
                        )
                    ],
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
        METRIC_NAMES,
    )
    return model_track_results_frame, tensors_by_backbone


@app.cell
def _(model_track_results_frame):
    method_matrix_heatmap(
        pool_difference_summary(model_track_results_frame, WELFARE_NAMES),
        centre=0.0,
        scale=100.0,
        title="Column minus row, win rate against the pool, all backbones and selector-grader pairs",
        legend_title="Points",
    )
    return


@app.cell
def _(deliberation_groups_frame, tensors_by_backbone):
    explorer_backbone, explorer_question, explorer_selector = explorer_controls(
        deliberation_groups_frame, tensors_by_backbone
    )
    mo.vstack([explorer_backbone, explorer_question, explorer_selector])
    return explorer_backbone, explorer_question, explorer_selector


@app.cell
def _(
    candidates_by_backbone,
    deliberation_groups_frame,
    explorer_backbone,
    explorer_question,
    explorer_selector,
    tensors_by_backbone,
):
    question_view(
        deliberation_groups_frame,
        candidates_by_backbone,
        tensors_by_backbone,
        explorer_backbone.value,
        explorer_question.value,
        explorer_selector.value,
    )
    return


if __name__ == "__main__":
    app.run()
