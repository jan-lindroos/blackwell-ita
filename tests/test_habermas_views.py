import numpy as np
import pandas as pd
import pytest

from blackwell_ita.habermas import (
    PARTICIPANT_PAIRS,
    WHOLE_GROUP,
    evaluation_results,
    gold_frontier,
    instance_welfare,
)
from blackwell_ita.habermas_views import (
    METHOD_LABELS,
    beaten_histogram,
    explorer_controls,
    frontier_chart,
    mixture_table,
    nash_chart,
    question_view,
    winner_chart,
)
from blackwell_ita.scoring import rewards_to_preferences

MODELS = ["qwen3_4b_bradley_terry", "qwen3_4b_pairwise"]


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
            evaluation_results(
                groups(),
                selector_tensors,
                gold_tensors(),
                selector_name,
                WHOLE_GROUP + PARTICIPANT_PAIRS,
            )
            for selector_name, selector_tensors in tensors_by_selector().items()
        ],
        ignore_index=True,
    )


def chart_rows(chart) -> list[dict]:
    return [row for rows in chart.to_dict()["datasets"].values() for row in rows]


def test_frontier_chart_draws_the_frontier_once_per_selector_and_every_method():
    welfare = instance_welfare(results())
    whole_group: pd.DataFrame = welfare[welfare["group_size"] == 4]  # pyright: ignore[reportAssignmentType]
    rows = chart_rows(
        frontier_chart(
            gold_frontier(groups(), gold_tensors(), WHOLE_GROUP, "best"),
            whole_group,
            "best",
            "Title",
        )
    )
    frontier_rows = [row for row in rows if row["kind"] == "frontier"]
    method_rows = [row for row in rows if row["kind"] == "method"]
    assert {row["selector_label"] for row in frontier_rows} == {"Qwen BT", "Qwen PW"}
    assert len(frontier_rows) == 2 * 21
    assert {row["method_label"] for row in method_rows} == set(
        METHOD_LABELS.values()
    ) - {"Uniform pool"}
    blackwell_row = next(
        row
        for row in method_rows
        if row["method_label"] == "BoB (orthant)" and row["selector_label"] == "Qwen PW"
    )
    expected = whole_group[
        (whole_group["method"] == "blackwell")
        & (whole_group["selector"] == "qwen3_4b_pairwise")
    ]["worst"].mean()
    assert blackwell_row["worst"] == expected


def test_nash_chart_compares_the_reference_algorithm_with_its_baselines():
    welfare = instance_welfare(results())
    rows = chart_rows(nash_chart(welfare[welfare["group_size"] == 4]))  # pyright: ignore[reportArgumentType]
    assert {row["method_label"] for row in rows} == {
        "Reference BoB (Nash)",
        "Best-of-N",
        "Maximin",
        "Scalarised Nash",
    }
    assert all(
        row["interval_low"] <= row["value"] <= row["interval_high"] for row in rows
    )


def test_beaten_histogram_counts_every_whole_group_participant_per_selector():
    whole_group: pd.DataFrame = results()[results()["group_size"] == 4]  # pyright: ignore[reportAssignmentType]
    chart = beaten_histogram(whole_group, 4)
    assert [column.title for column in chart.hconcat] == ["Qwen BT", "Qwen PW"]
    assert len(chart_rows(chart)) == len(whole_group)


def test_mixture_table_lists_answers_any_method_uses_with_weight_columns():
    pool_statements = candidates().query("question_id == 'q1'")
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
        "Uniform pool",
        "Scalarised Nash",
        "BoB (orthant)",
    ]
    assert table["answer"].tolist() == ["first", "second", "third", "fourth"]
    assert table["originating_kind"].tolist() == [
        "consensus",
        "participant (2)",
        "pair (1, 3)",
        "compromise",
    ]
    assert table["BoB (orthant)"].tolist()[0] == 0.25
    assert np.isnan(table["Scalarised Nash"].iloc[1])


def test_question_view_shows_participants_and_the_mixture_table():
    view = question_view(groups(), candidates(), tensors_by_selector(), "q1", MODELS[0])
    for text in ["Should we?", "Participant 4:"]:
        assert text in view.text


def test_question_view_reports_unscored_questions():
    tensors = tensors_by_selector()
    del tensors[MODELS[0]]["q1"]
    view = question_view(groups(), candidates(), tensors, "q1", MODELS[0])
    assert "Not scored yet" in view.text


def test_explorer_controls_offer_questions_and_selectors():
    question, selector = explorer_controls(groups(), tensors_by_selector())
    assert question.value == "q1"
    assert sorted(selector.options.values()) == sorted(MODELS)


def test_winner_chart_summarises_average_and_worst_served_per_method():
    winner_rates = {"q1": np.full((4, 4), 0.25), "q2": np.full((4, 4), 0.75)}
    welfare = instance_welfare(
        pd.concat(
            [
                evaluation_results(
                    groups(),
                    selector_tensors,
                    gold_tensors(),
                    selector_name,
                    WHOLE_GROUP,
                    winner_rates,
                )
                for selector_name, selector_tensors in tensors_by_selector().items()
            ],
            ignore_index=True,
        )
    )
    rows = chart_rows(winner_chart(welfare))
    assert {row["served_label"] for row in rows} == {"Average", "Worst served"}
    assert all(row["value"] == pytest.approx(0.5) for row in rows)
    assert {row["method_label"] for row in rows} == set(METHOD_LABELS.values())
