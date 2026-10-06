from itertools import permutations

import numpy as np
import pandas as pd

from blackwell_ita.habermas import model_track_results, pool_difference_summary
from blackwell_ita.habermas_views import (
    explorer_controls,
    method_matrix_heatmap,
    mixture_table,
    question_view,
)
from blackwell_ita.scoring import rewards_to_preferences

BACKBONE = "mistralai/Mistral-7B-Instruct-v0.3"
MODELS = ["qwen3_4b_bradley_terry", "qwen3_4b_pairwise"]


def groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1", "q2"],
            "question": ["Should we?", "Must we?"],
            "opinions": [[f"opinion {i}" for i in range(4)]] * 2,
            "anchor": ["anchor", "anchor"],
        }
    )


def tensors_by_backbone() -> dict:
    # Four participants over a pool of three plus the anchor
    random_generator = np.random.default_rng(0)
    return {
        BACKBONE: {
            model_name: {
                question_id: rewards_to_preferences(
                    random_generator.normal(size=(4, 4))
                )
                for question_id in ["q1", "q2"]
            }
            for model_name in MODELS
        }
    }


def candidates() -> dict:
    return {
        BACKBONE: pd.DataFrame(
            {
                "question_id": ["q1"] * 3 + ["q2"] * 3,
                "sample_index": [2, 0, 1] * 2,
                "candidate_kind": ["pair", "consensus", "participant"] * 2,
                "target_participants": [[1, 3], [], [2]] * 2,
                "response": ["third", "first", "second"] * 2,
            }
        )
    }


def test_method_matrix_heatmap_draws_one_grid_per_welfare_function_without_uniform():
    tensors = tensors_by_backbone()[BACKBONE]
    results = pd.concat(
        [
            model_track_results(
                groups(), tensors[selector], tensors[grader], BACKBONE, selector, grader
            )
            for selector, grader in permutations(MODELS, 2)
        ],
        ignore_index=True,
    )
    summary = pool_difference_summary(
        results, ["egalitarian_welfare", "nash_welfare", "utilitarian_welfare"]
    )
    chart = method_matrix_heatmap(summary, 0.0, 100.0, "Title", "Points").to_dict()
    assert [grid["title"] for grid in chart["hconcat"]] == [
        "Egalitarian",
        "Nash",
        "Utilitarian",
    ]
    labels = {
        row[key]
        for rows in chart["datasets"].values()
        for row in rows
        for key in ("row_label", "column_label")
    }
    assert "Uniform" not in labels


def test_mixture_table_lists_answers_any_method_uses_with_weight_columns():
    pool_statements = candidates()[BACKBONE].query("question_id == 'q1'")
    pool_statements = pool_statements.sort_values("sample_index")
    policies = {
        "uniform": np.full(4, 0.25),
        "blackwell": np.array([0.25, 0.0, 0.75, 0.0]),
        "scalarised_nash": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    table = mixture_table(policies, pool_statements)
    assert table.columns.tolist() == [
        "answer",
        "originating_kind",
        "Blackwell (orthant)",
        "Scalarised Nash",
    ]
    assert table["answer"].tolist() == ["first", "third"]
    assert table["originating_kind"].tolist() == ["consensus", "pair (1, 3)"]
    assert table["Blackwell (orthant)"].tolist() == [0.25, 0.75]
    assert np.isnan(table["Scalarised Nash"].iloc[1])


def test_question_view_shows_participants_and_the_mixture_table():
    view = question_view(
        groups(), candidates(), tensors_by_backbone(), BACKBONE, "q1", MODELS[0]
    )
    for text in ["Should we?", "Participant 4:"]:
        assert text in view.text


def test_question_view_reports_unscored_questions():
    tensors = tensors_by_backbone()
    del tensors[BACKBONE][MODELS[0]]["q1"]
    view = question_view(groups(), candidates(), tensors, BACKBONE, "q1", MODELS[0])
    assert "Not scored yet" in view.text


def test_explorer_controls_offer_backbones_questions_and_selectors():
    backbone, question, selector = explorer_controls(groups(), tensors_by_backbone())
    assert backbone.value == BACKBONE
    assert question.value == "q1"
    assert sorted(selector.options.values()) == sorted(MODELS)


def test_method_matrix_heatmap_signs_differences_and_skips_ties():
    summary = pd.DataFrame(
        {
            "row_method": ["blackwell", "blackwell"],
            "column_method": ["blackwell", "maximin"],
            "welfare": ["nash_welfare", "nash_welfare"],
            "value": [0.0, 0.004],
            "interval_low": [1e-12, 0.001],
            "interval_high": [1e-12, 0.007],
            "instances": [100, 100],
        }
    )
    datasets = method_matrix_heatmap(summary, 0.0, 100.0, "Title", "Points").to_dict()[
        "datasets"
    ]
    assert [row["cell_text"] for rows in datasets.values() for row in rows] == [
        "+0.00",
        "+0.40*",
    ]
