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
        human_track_results,
        load_habermas_rankings,
        model_track_results,
        participant_preference_tensor,
    )
    from blackwell_ita.hub import (
        ARTIFACTS_REPOSITORY,
        hub_file_exists,
        read_hub_dataframe,
        upload_dataframe,
    )
    from blackwell_ita.scoring import score_with_resume
    from blackwell_ita.selection import summarise_methods
    from blackwell_ita.training import ensure_trained, load_trained

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLES_PER_GROUP = 64
    WELFARE_NAMES = ["rawlsian_welfare", "nash_welfare", "utilitarian_welfare"]


@app.function
def ensure_hub_dataframe(filename: str, build) -> pd.DataFrame:
    """Build and upload ``filename`` unless the hub has it, then read it back."""
    if not hub_file_exists(ARTIFACTS_REPOSITORY, filename, HABERMAS_PREFIX):
        upload_dataframe(ARTIFACTS_REPOSITORY, filename, build(), HABERMAS_PREFIX)
    return read_hub_dataframe(ARTIFACTS_REPOSITORY, filename, HABERMAS_PREFIX)


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Run on a GPU with `HF_TOKEN` and `WANDB_API_KEY` set.
    """)
    return


@app.cell
def _():
    rankings = ensure_hub_dataframe(
        "rankings.parquet",
        lambda: assign_prompt_splits(load_habermas_rankings(), "question_id"),
    )
    pairs = habermas_pairs(rankings)
    split_summary(pairs, "question_id")
    return pairs, rankings


@app.cell
def _(pairs):
    preference_model_metrics = pd.concat(
        [
            ensure_trained(
                configuration, pairs, HABERMAS_CRITERIA, HABERMAS_PREFIX, DEVICE
            )
            for configuration in HABERMAS_CONFIGURATIONS
        ],
        ignore_index=True,
    )
    preference_model_metrics.pivot_table(
        index=["split", "criterion"], columns="name", values="decisive_accuracy"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Human-rated candidates
    """)
    return


@app.cell
def _(rankings):
    human_track_sets = candidate_sets(rankings, "test")
    human_track_inputs = {
        set_row["set_id"]: {
            "question": set_row["question"],
            "opinions": list(set_row["opinions"]),
            "statements": list(set_row["statements"]),
        }
        for set_row in human_track_sets.to_dict("records")
    }
    human_track_results_frame = pd.concat(
        [
            human_track_results(
                human_track_sets,
                score_participant_tensors(
                    configuration,
                    f"human_track_{configuration.name}.npz",
                    human_track_inputs,
                ),
                configuration.name,
            )
            for configuration in HABERMAS_CONFIGURATIONS
        ],
        ignore_index=True,
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
    ## Model-based candidates

    Generated statements have no human ratings, so every number here is a
    model-based proxy graded by the preference model that did not select.
    """)
    return


@app.cell
def _(rankings):
    deliberation_groups_frame = ensure_hub_dataframe(
        "groups.parquet", lambda: deliberation_groups(rankings, "test")
    )
    deliberation_groups_frame
    return (deliberation_groups_frame,)


@app.cell
def _(deliberation_groups_frame):
    generation_prompts = [
        consensus_prompt(group_row["question"], list(group_row["opinions"]))
        for group_row in deliberation_groups_frame.to_dict("records")
    ]
    candidates_by_backbone = {
        backbone_name: ensure_hub_dataframe(
            f"candidates_{backbone_name.split('/')[-1].lower()}.parquet",
            lambda backbone_name=backbone_name: generate_responses(
                backbone_name, generation_prompts, SAMPLES_PER_GROUP, DEVICE
            ).assign(
                question_id=lambda responses: deliberation_groups_frame[
                    "question_id"
                ].to_numpy()[responses["prompt_index"]]
            ),
        )
        for backbone_name in GENERATION_BACKBONES
    }
    return (candidates_by_backbone,)


@app.cell
def _(candidates_by_backbone, deliberation_groups_frame):
    model_track_frames = []
    for backbone_name, backbone_candidates in candidates_by_backbone.items():
        backbone_slug = backbone_name.split("/")[-1].lower()
        # The anchor rides at the last index of every pool
        pool_inputs = {
            group_row["question_id"]: {
                "question": group_row["question"],
                "opinions": list(group_row["opinions"]),
                "statements": backbone_candidates[
                    backbone_candidates["question_id"] == group_row["question_id"]
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
                f"model_track_{backbone_slug}_{configuration.name}.npz",
                pool_inputs,
            )
            for configuration in HABERMAS_CONFIGURATIONS
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
