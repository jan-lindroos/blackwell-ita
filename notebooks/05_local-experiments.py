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
#     # torch and transformers run no model here: winners imports them
#     # transitively via train_rms
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


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from blackwell_ita.judge import JUDGE_MODEL, outcomes
    from blackwell_ita.utils.artifacts import (
        HEADLINE_METHODS,
        artifact_path,
        upload_dataframe,
        upsert_dataframe,
    )
    from blackwell_ita.winners import (
        best_of_nash,
        blackwell_winner,
        bt_best_of_n,
        bt_preference_tensor,
        expected_scores,
        mean_criterion_best_of_n,
        policy_support,
        worst_criterion_best_of_n,
    )

    return (
        HEADLINE_METHODS,
        JUDGE_MODEL,
        artifact_path,
        best_of_nash,
        blackwell_winner,
        bt_best_of_n,
        bt_preference_tensor,
        expected_scores,
        mean_criterion_best_of_n,
        outcomes,
        policy_support,
        upload_dataframe,
        upsert_dataframe,
        worst_criterion_best_of_n,
    )


@app.cell
def _():
    import json

    import numpy as np
    import pandas as pd

    return json, np, pd


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=["helpsteer2", "community-alignment"],
        value="helpsteer2",
        label="Dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _():
    evaluation_prompt_count = 100
    samples_per_prompt = 64
    n_values = [1, 2, 4, 8, 16, 32, 64]
    judge_n_values = [1, 4, 16, 64]
    return (
        evaluation_prompt_count,
        judge_n_values,
        n_values,
        samples_per_prompt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inputs from the hub

    The GPU notebooks cache everything model-side on the artifacts repo:
    candidate responses, reference anchors, preference tensors,
    Bradley-Terry rewards and per-criterion anchor scores. This notebook is
    CPU-only: linear programmes, Claude judging and aggregation.
    """)
    return


@app.cell
def _(artifact_path, dataset_picker, evaluation_prompt_count, json, np, pd):
    dataset = dataset_picker.value
    candidates_dataframe = pd.read_parquet(artifact_path(dataset, "candidates.parquet"))
    anchors_dataframe = pd.read_parquet(artifact_path(dataset, "anchors.parquet"))
    model_scores = np.load(artifact_path(dataset, "model_scores.npz"))
    model_scores_meta = json.loads(
        artifact_path(dataset, "model_scores_meta.json").read_text()
    )
    criterion_anchor_scores = pd.read_parquet(
        artifact_path(dataset, "criterion_anchor_scores.parquet")
    )
    evaluation_prompts = model_scores_meta["prompts"]
    criterion_columns = model_scores_meta["criterion_columns"]
    # tensor_{i}/rewards_{i} index rows of anchors.parquet. A mismatch would
    # silently score every prompt against another prompt's cached tensors
    assert evaluation_prompts == anchors_dataframe["prompt"].tolist()
    assert len(evaluation_prompts) == evaluation_prompt_count
    anchors = dict(
        zip(anchors_dataframe["prompt"], anchors_dataframe["anchor"], strict=True)
    )
    overall_index = criterion_columns.index("overall")
    criteria_indices = [
        index for index, column in enumerate(criterion_columns) if column != "overall"
    ]
    len(evaluation_prompts)
    return (
        anchors,
        candidates_dataframe,
        criteria_indices,
        criterion_anchor_scores,
        dataset,
        evaluation_prompts,
        model_scores,
        overall_index,
    )


@app.cell
def _(mo):
    solve_button = mo.ui.run_button(label="Compute policies and select responses")
    solve_button
    return (solve_button,)


@app.cell
def _(
    best_of_nash,
    blackwell_winner,
    bt_best_of_n,
    bt_preference_tensor,
    candidates_dataframe,
    criteria_indices,
    dataset,
    evaluation_prompts,
    mean_criterion_best_of_n,
    mo,
    model_scores,
    n_values,
    overall_index,
    pd,
    policy_support,
    samples_per_prompt,
    solve_button,
    upload_dataframe,
    worst_criterion_best_of_n,
):
    mo.stop(not solve_button.value)
    selection_rows = []
    for prompt_index, prompt in enumerate(mo.status.progress_bar(evaluation_prompts)):
        responses = (
            candidates_dataframe[candidates_dataframe["prompt"] == prompt]
            .sort_values("sample_index")["response"]
            .tolist()
        )
        base_anchor = responses[samples_per_prompt]
        responses = responses[:samples_per_prompt]
        tensor = model_scores[f"tensor_{prompt_index}"]
        full_rewards = model_scores[f"rewards_{prompt_index}"]
        rewards = full_rewards[:, overall_index]
        rows = [
            {"method": "base", "n": 1, "weight": 1.0, "response": responses[0]},
            {"method": "base_anchor", "n": 0, "weight": 1.0, "response": base_anchor},
        ]
        for n in n_values:
            criteria_tensor = tensor[criteria_indices][:, :n, :n]
            bt_criteria_tensor = bt_preference_tensor(
                full_rewards[:n][:, criteria_indices]
            )
            policies = {
                "bt_best_of_n": bt_best_of_n(rewards[:n]),
                "best_of_nash": best_of_nash(tensor[overall_index, :n, :n]),
                "mean_criterion_best_of_n": mean_criterion_best_of_n(criteria_tensor),
                "worst_criterion_best_of_n": worst_criterion_best_of_n(
                    criteria_tensor
                ),
                "blackwell": blackwell_winner(criteria_tensor),
                "bt_mean_criterion_best_of_n": mean_criterion_best_of_n(
                    bt_criteria_tensor
                ),
                "bt_worst_criterion_best_of_n": worst_criterion_best_of_n(
                    bt_criteria_tensor
                ),
            }
            rows.extend(
                {
                    "method": method,
                    "n": n,
                    "weight": weight,
                    "response": responses[index],
                }
                for method, policy in policies.items()
                for index, weight in policy_support(policy)
            )
        selection_rows.extend({"prompt": prompt, **row} for row in rows)
    selections_dataframe = pd.DataFrame(
        selection_rows, columns=["prompt", "method", "n", "weight", "response"]
    )
    upload_dataframe(dataset, "selections.parquet", selections_dataframe)
    selections_dataframe
    return (selections_dataframe,)


@app.cell
def _(HEADLINE_METHODS, mo):
    judge_method_picker = mo.ui.multiselect(
        options=HEADLINE_METHODS,
        value=HEADLINE_METHODS,
        label="Methods",
    )
    judge_button = mo.ui.run_button(label="Judge with Claude")
    mo.hstack([judge_method_picker, judge_button], justify="start")
    return judge_button, judge_method_picker


@app.cell
def _(
    JUDGE_MODEL,
    anchors,
    dataset,
    expected_scores,
    judge_button,
    judge_method_picker,
    judge_n_values,
    mo,
    outcomes,
    pd,
    selections_dataframe,
    upsert_dataframe,
):
    mo.stop(not judge_button.value)
    judged_selections = selections_dataframe[
        selections_dataframe["method"].isin(judge_method_picker.value)
        & selections_dataframe["n"].isin(judge_n_values)
    ]
    # Expectation scoring: judge each distinct support atom once against the
    # reference anchor, then average atom scores under each policy's
    # weights. Shared atom scores pin all methods to the same value at
    # N = 1 and make sweep curves move only where the policies actually
    # differ
    atoms = judged_selections[["prompt", "response"]].drop_duplicates()
    comparisons = [
        {
            "instruction": atom["prompt"],
            "response": atom["response"],
            "anchor": anchors[atom["prompt"]],
        }
        for _, atom in atoms.iterrows()
    ]
    atom_scores = {}
    atom_rows = []
    for (_, judged_atom), judgement in zip(
        atoms.iterrows(), outcomes(comparisons), strict=True
    ):
        atom_scores[(judged_atom["prompt"], judged_atom["response"])] = judgement[
            "score"
        ]
        atom_rows.append(
            {
                "prompt": judged_atom["prompt"],
                "response": judged_atom["response"],
                "anchor": "reference",
                "judge_model": JUDGE_MODEL,
                **judgement,
            }
        )
    judgements_dataframe = expected_scores(judged_selections, atom_scores)
    judgements_dataframe["anchor"] = "reference"
    judgements_dataframe["criterion"] = "overall"
    judgements_dataframe["judge_model"] = JUDGE_MODEL
    judgements_dataframe = judgements_dataframe[
        ["prompt", "method", "n", "anchor", "criterion", "judge_model", "score"]
    ]
    # Judging a subset of methods replaces only their hub rows, so partial
    # runs compose into the full artifact notebook 06 reads
    upsert_dataframe(
        dataset, "judgements.parquet", judgements_dataframe, ["method", "n"]
    )
    upsert_dataframe(
        dataset,
        "atom_judgements.parquet",
        pd.DataFrame(atom_rows),
        ["prompt", "response"],
    )
    judgements_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-criterion measurement

    Per-criterion and per-group win probabilities come from the held-out
    evaluation pairwise model. Claude judges overall quality only.
    """)
    return


@app.cell
def _(
    HEADLINE_METHODS,
    criterion_anchor_scores,
    dataset,
    expected_scores,
    pd,
    samples_per_prompt,
    selections_dataframe,
    upload_dataframe,
):
    headline_selections = selections_dataframe[
        selections_dataframe["method"].isin(HEADLINE_METHODS)
        & (
            (selections_dataframe["n"] == samples_per_prompt)
            | (selections_dataframe["method"] == "base")
        )
    ]
    criterion_frames = []
    for criterion in criterion_anchor_scores["criterion"].unique():
        criterion_subset = criterion_anchor_scores[
            criterion_anchor_scores["criterion"] == criterion
        ]
        criterion_atom_scores = {
            (atom_prompt, atom_response): atom_score
            for atom_prompt, atom_response, atom_score in zip(
                criterion_subset["prompt"],
                criterion_subset["response"],
                criterion_subset["score"],
                strict=True,
            )
        }
        criterion_frame = expected_scores(headline_selections, criterion_atom_scores)
        criterion_frame["criterion"] = criterion
        criterion_frames.append(criterion_frame)
    criterion_scores_dataframe = pd.concat(criterion_frames, ignore_index=True)[
        ["prompt", "method", "criterion", "score"]
    ]
    upload_dataframe(dataset, "criterion_scores.parquet", criterion_scores_dataframe)
    criterion_scores_dataframe
    return


if __name__ == "__main__":
    app.run()
