from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from blackwell_ita.evaluation import (
    METHOD_WELFARES,
    budget_sweep,
    error_sweep,
    method_results,
    pool_size_results,
    sampling_results,
    selection_policies,
)
from blackwell_ita.habermas import WHOLE_GROUP
from blackwell_ita.habermas_views import (
    METHOD_LABELS,
    budget_chart,
    error_chart,
    explorer_controls,
    mixture_table,
    question_view,
    regret_chart,
    sampling_chart,
    welfare_row_chart,
)
from blackwell_ita.scoring import rewards_to_preferences

MODELS = ["qwen3_4b_bradley_terry", "gold"]
PARTICIPANT_PAIRS = list(combinations(range(4), 2))


def groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1", "q2"],
            "question": ["Should we?", "Must we?"],
            "opinions": [[f"opinion {i}" for i in range(4)]] * 2,
            "split": ["test", "test"],
        }
    )


def tensors_by_selector() -> dict:
    random_generator = np.random.default_rng(0)
    return {
        model_name: {
            question_id: rewards_to_preferences(random_generator.normal(size=(4, 4)))
            for question_id in ["q1", "q2"]
        }
        for model_name in MODELS
    }


def gold_tensors() -> dict:
    random_generator = np.random.default_rng(1)
    return {
        question_id: rewards_to_preferences(random_generator.normal(size=(4, 4)))
        for question_id in ["q1", "q2"]
    }


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1"] * 4 + ["q2"] * 4,
            "sample_index": [2, 0, 1, 3] * 2,
            "candidate_kind": ["pair", "consensus", "participant", "compromise"] * 2,
            "target_participants": [[1, 3], [], [2], []] * 2,
            "response": ["third", "first", "second", "fourth"] * 2,
        }
    )


def results() -> pd.DataFrame:
    return pd.concat(
        [
            method_results(
                groups(),
                selector_tensors,
                gold_tensors(),
                selector_name,
                WHOLE_GROUP + PARTICIPANT_PAIRS,
                selector_name.endswith("bradley_terry"),
            )
            for selector_name, selector_tensors in tensors_by_selector().items()
        ],
        ignore_index=True,
    )


def chart_rows(chart) -> list[dict]:
    return [row for rows in chart.to_dict()["datasets"].values() for row in rows]


def test_regret_chart_summarises_every_selector_method_and_welfare():
    rows = chart_rows(regret_chart(results(), "Whole group"))
    assert {row["selector_label"] for row in rows} == {"Qwen3 4B", "Gold"}
    assert {row["welfare_label"] for row in rows} == {
        "Egalitarian",
        "Utilitarian",
        "Nash",
    }
    # Best-of-N runs for the Bradley-Terry selector only, and the uniform pool is left out
    assert len(rows) == 3 * (8 + 7)
    assert "Uniform pool" not in {row["method_label"] for row in rows}
    assert all(
        row["interval_low"] <= row["value"] <= row["interval_high"] for row in rows
    )


def test_welfare_row_chart_summarises_each_method_welfare_and_pool_size():
    kinds = np.array(["consensus", "participant"] * 2)
    pool_sizes = pool_size_results(
        groups(),
        tensors_by_selector()[MODELS[0]],
        gold_tensors(),
        {"q1": kinds, "q2": kinds},
        [2, 4],
        3,
    )
    rows = chart_rows(welfare_row_chart(pool_sizes, "pool_size", "Pool size"))
    assert len(rows) == 3 * 2 * 7
    assert {row["welfare_label"] for row in rows} == {
        "Egalitarian",
        "Utilitarian",
        "Nash",
    }
    assert all(
        row["interval_low"] <= row["value"] <= row["optimum"] + 1e-6 for row in rows
    )


def test_sampling_chart_draws_each_target_and_the_sampling_bound():
    sampling = sampling_results(groups(), gold_tensors(), [1, 2, 4], draw_count=3)
    rows = chart_rows(sampling_chart(sampling, 4))
    bound_rows = [row for row in rows if "target_label" not in row]
    assert len(rows) == 3 * 3 + 3
    assert [row["value"] for row in bound_rows] == pytest.approx(
        [2 * 0.75**size for size in (1, 2, 4)]
    )


def test_error_chart_plots_each_target_method_against_its_model_error():
    sweep = error_sweep(
        groups(), tensors_by_selector()[MODELS[1]], gold_tensors(), [0.0, 1.0]
    )
    rows = chart_rows(error_chart(sweep))
    assert len(rows) == 2 * len(METHOD_WELFARES)
    assert max(row["model_error"] for row in rows if row["factor"] == 0.0) < 1e-12
    assert {row["method_label"] for row in rows} == {
        METHOD_LABELS[name] for name in METHOD_WELFARES
    }


def test_budget_chart_summarises_each_budget_factor():
    sweep = budget_sweep(
        groups(), tensors_by_selector()[MODELS[1]], gold_tensors(), [0.0, 1.0]
    )
    rows = chart_rows(budget_chart(sweep))
    assert [row["budget_factor"] for row in rows] == [0.0, 1.0]
    assert all(
        row["interval_low"] <= row["value"] <= row["interval_high"] for row in rows
    )


def test_mixture_table_lists_answers_any_method_uses_with_weight_columns():
    policies = {
        "uniform": np.full(4, 0.25),
        "blackwell": np.array([0.0, 1.0, 0.0, 0.0]),
    }
    pool = candidates()[candidates()["question_id"] == "q1"].sort_values("sample_index")  # pyright: ignore[reportCallIssue]
    table = mixture_table(policies, pool)
    assert table["answer"].tolist() == ["first", "second", "third", "fourth"]
    assert table["originating_kind"].tolist() == [
        "consensus",
        "participant (2)",
        "pair (1, 3)",
        "compromise",
    ]
    assert table.columns.tolist()[2:] == ["Uniform pool", "BoB (orthant)"]
    assert table["BoB (orthant)"].tolist()[1] == 1.0
    assert np.isnan(table["BoB (orthant)"].tolist()[0])


def test_question_view_shows_participants_and_the_mixture_table():
    policies = selection_policies(
        tensors_by_selector()[MODELS[0]]["q1"], True, 0.3, 0.05
    )
    view = question_view(groups(), candidates(), policies, "q1")
    text = view.text
    assert "Should we?" in text
    assert "Participant 4" in text


def test_explorer_controls_offer_questions_and_selectors():
    question, selector = explorer_controls(groups(), ["gold"] + MODELS)
    assert question.value == "q1"
    assert selector.value == "gold"
