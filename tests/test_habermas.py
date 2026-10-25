import numpy as np
import pandas as pd
import pytest

from blackwell_ita import training
from blackwell_ita.habermas import (
    CANDIDATE_SLOTS,
    GOLD_CONFIGURATION,
    PLACEHOLDER_OPINION,
    PROXY_CONFIGURATIONS,
    assign_habermas_splits,
    candidate_prompts,
    candidate_sets,
    complete_groups,
    deliberation_groups,
    generate_candidates,
    gold_pairs,
    habermas_pairs,
    habermas_prompt,
    rank_preference,
    scoring_inputs,
    style_examples,
    usable_rankings,
)
from blackwell_ita.scoring import rewards_to_preferences


def rank_tensor(participant_ranks: list[list[int]]) -> np.ndarray:
    """Preferences implied by ranks, shape (participant, statement, statement)."""
    ranks = np.asarray(participant_ranks)
    return (
        (ranks[:, :, None] < ranks[:, None, :])
        + 0.5 * (ranks[:, :, None] == ranks[:, None, :])
    ).astype(float)


def comparison_row(**overrides) -> dict:
    return {
        "question.id": "q",
        "question.text": "Should we?",
        "round_id": "r",
        "iteration_index": 0,
        "metadata.participant_id": "p",
        "own_opinion.text": "Yes.",
        "candidates.text": np.array(["a", "b", "c"], dtype=object),
        "rankings.numerical_ranks": np.array([1, 0, 1]),
        "metadata.provenance": "HUMAN_CITIZEN",
        "rankings.metadata.status": "COMPLETED",
    } | overrides


def test_assign_habermas_splits_withholds_whole_paper_questions():
    rankings = pd.DataFrame(
        {"question_id": [f"q{index // 2}" for index in range(1000)], "row": range(1000)}
    )
    split_rankings = assign_habermas_splits(rankings)
    assert split_rankings["row"].tolist() == list(range(1000))
    assert (split_rankings.groupby("question_id")["split"].nunique() == 1).all()
    question_splits = split_rankings.drop_duplicates("question_id")["split"]
    assert question_splits.value_counts().to_dict() == {
        "train": 210,
        "paper": 200,
        "validation": 45,
        "test": 45,
    }


def test_usable_rankings_keeps_only_complete_human_rankings():
    comparisons = pd.DataFrame(
        [
            comparison_row(),
            comparison_row(**{"metadata.provenance": "BOT_CITIZEN"}),
            comparison_row(**{"rankings.metadata.status": "DROPPED"}),
            comparison_row(**{"rankings.numerical_ranks": np.array([-1, -1, -1])}),
            comparison_row(**{"rankings.numerical_ranks": np.array([0, 1])}),
        ]
    )
    rankings = usable_rankings(comparisons)
    assert len(rankings) == 1
    ranking = rankings.iloc[0]
    assert ranking["statements"] == ["a", "b", "c"]
    assert ranking["ranks"] == [1, 0, 1]


def test_rank_preference_prefers_the_lower_rank_and_ties_at_half():
    assert [rank_preference(0, 1), rank_preference(2, 1), rank_preference(1, 1)] == [
        1.0,
        0.0,
        0.5,
    ]


def ranking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q", "q"],
            "question": ["Should we?", "Should we?"],
            "round_id": ["r", "r"],
            "iteration_index": [0, 0],
            "participant_id": ["p1", "p2"],
            "opinion": ["Yes.", "No."],
            "statements": [["a", "b", "c"], ["a", "b", "c"]],
            "ranks": [[0, 1, 1], [2, 1, 0]],
            "split": ["test", "test"],
        }
    )


def test_habermas_pairs_cover_every_unordered_pair_per_participant():
    pairs = habermas_pairs(ranking_frame())
    assert len(pairs) == 6
    first_participant_pairs = pairs[
        pairs["prompt"] == habermas_prompt("Should we?", "Yes.")
    ]
    assert first_participant_pairs[
        ["response_a", "response_b", "preference"]
    ].values.tolist() == [  # pyright: ignore[reportAttributeAccessIssue]
        ["a", "b", 1.0],
        ["a", "c", 1.0],
        ["b", "c", 0.5],
    ]
    assert set(pairs["split"]) == {"test"}


def test_candidate_sets_group_participants_who_saw_the_same_statements():
    sets = candidate_sets(ranking_frame(), "test")
    assert len(sets) == 1
    assert sets.iloc[0]["set_id"] == "r_0"
    assert sets.iloc[0]["opinions"] == ["Yes.", "No."]
    assert candidate_sets(ranking_frame(), "train").empty


def test_configurations_have_unique_names_and_known_model_types():
    configurations = [GOLD_CONFIGURATION, *PROXY_CONFIGURATIONS]
    names = [configuration.name for configuration in configurations]
    assert len(names) == len(set(names))
    assert {configuration.model_type for configuration in configurations} <= set(
        training.REWARD_MODEL_CLASSES
    )


def opening_rankings(split_name: str, question_count: int) -> pd.DataFrame:
    """Five participants per question, all ranking one opening set."""
    return pd.DataFrame(
        [
            {
                "question_id": f"q{question}",
                "question": f"Question {question}?",
                "round_id": f"r{question}",
                "iteration_index": 0,
                "participant_id": f"p{participant}",
                "opinion": f"opinion {question} {participant}",
                "statements": ["worst", "best", "middle"],
                "ranks": [2, 0, 1],
                "split": split_name,
            }
            for question in range(question_count)
            for participant in range(5)
        ]
    )


def test_deliberation_groups_pick_four_participants_per_question():
    groups = deliberation_groups(opening_rankings("test", 3), "test", group_count=2)
    assert len(groups) == 2
    assert groups["question_id"].is_unique
    assert all(len(opinions) == 4 for opinions in groups["opinions"])
    assert set(groups["split"]) == {"test"}


def test_deliberation_groups_without_a_count_take_every_eligible_question():
    groups = deliberation_groups(opening_rankings("test", 3), "test", group_count=None)
    assert groups["question_id"].tolist() == ["q0", "q1", "q2"]


def test_style_examples_keep_train_statements_of_habermas_length():
    rankings = pd.concat(
        [opening_rankings("train", 2), opening_rankings("test", 2)],
        ignore_index=True,
    )
    long_statement = " ".join(["word"] * 60)
    rankings["statements"] = [
        [
            long_statement,
            " ".join(["word"] * 10),
            f'"{long_statement}"',
            f"Final consensus statement: {long_statement}",
        ]
    ] * len(rankings)
    examples = style_examples(rankings, example_count=2)
    assert examples.columns.tolist() == ["question_id", "question", "statement"]
    assert set(examples["statement"]) == {long_statement}
    assert set(examples["question_id"]) == {"q0", "q1"}


def proposal_groups() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": question_id,
                "question": question_id + "?",
                "opinions": ["A", "B", "C", "D"],
                "split": split_name,
            }
            for question_id, split_name in [("q1", "train"), ("q2", "test")]
        ]
    )


def examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q1"] * 3 + ["other"] * 10,
            "question": ["q1?"] * 3 + ["Other?"] * 10,
            "statement": [" ".join(["samequestion"] * 50)] * 3
            + [" ".join(["example"] * (10 * length)) for length in range(1, 11)],
        }
    )


def test_candidate_prompts_cover_every_slot_once():
    plan = candidate_prompts(proposal_groups(), examples())
    pd.testing.assert_frame_equal(
        plan, candidate_prompts(proposal_groups(), examples())
    )
    pool_sizes = plan.groupby("question_id")["sample_index"].apply(sorted).to_dict()
    assert pool_sizes == {"q1": list(range(24)), "q2": list(range(24))}
    test_plan = plan[plan["question_id"] == "q2"].sort_values("sample_index")  # pyright: ignore[reportCallIssue]
    assert test_plan["candidate_kind"].tolist() == [kind for kind, _ in CANDIDATE_SLOTS]
    targets = test_plan[test_plan["candidate_kind"] == "participant"][
        "target_participants"
    ]
    assert sorted(tuple(target) for target in targets) == [
        (participant,) for participant in range(1, 5) for _ in range(3)
    ]


def test_consensus_orders_show_each_participant_in_each_position_three_times():
    plan = candidate_prompts(proposal_groups().iloc[:1], examples())
    orders = plan[plan["candidate_kind"] == "consensus"]["opinion_order"].tolist()
    assert len({tuple(order) for order in orders}) == 12
    for position in range(4):
        counts = pd.Series([order[position] for order in orders]).value_counts()
        assert counts.to_dict() == {1: 3, 2: 3, 3: 3, 4: 3}


def test_candidate_prompts_show_only_the_targeted_opinions():
    plan = candidate_prompts(proposal_groups(), examples())
    for row in plan.to_dict("records"):
        shown = row["target_participants"] or [1, 2, 3, 4]
        assert sorted(row["opinion_order"]) == sorted(shown)
        for participant, opinion in enumerate(["A", "B", "C", "D"], 1):
            assert (f"Opinion: {opinion}" in row["prompt"]) == (participant in shown)


def test_candidate_prompts_show_examples_from_other_questions_and_their_lengths():
    plan = candidate_prompts(proposal_groups(), examples())
    for row in plan.to_dict("records"):
        assert row["prompt"].count("Statement: ") == 8
        if row["question_id"] == "q1":
            assert "samequestion" not in row["prompt"]
        target_words = int(row["prompt"].split("Aim for about ")[1].split()[0])
        assert target_words in range(10, 110, 10)


def test_candidate_prompts_reject_wrong_group_sizes():
    groups = proposal_groups().assign(opinions=[["A", "B"], ["A", "B"]])
    with pytest.raises(ValueError, match="four participants"):
        candidate_prompts(groups, examples())


def test_generate_candidates_keeps_each_response_with_its_prompt(monkeypatch):
    monkeypatch.setattr(
        "blackwell_ita.habermas.generate_responses",
        lambda prompts: [prompt.upper() for prompt in prompts],
    )
    plan = candidate_prompts(proposal_groups(), examples())
    result = generate_candidates(plan)
    assert (result["response"] == result["prompt"].str.upper()).all()


def test_gold_pairs_label_every_pool_pair_per_participant_with_the_gold():
    groups = proposal_groups().iloc[:1]
    candidates = pd.DataFrame(
        {
            "question_id": ["q1"] * 3,
            "sample_index": [2, 0, 1],
            "response": ["c", "a", "b"],
        }
    )
    gold_tensor = rewards_to_preferences(np.arange(12.0).reshape(4, 3))
    pairs = gold_pairs(groups, candidates, {"q1": gold_tensor})
    assert len(pairs) == 4 * 3
    first_participant = pairs[pairs["prompt"] == habermas_prompt("q1?", "A")]
    assert first_participant[["response_a", "response_b"]].values.tolist() == [  # pyright: ignore[reportAttributeAccessIssue]
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
    ]
    assert first_participant["preference"].tolist() == pytest.approx(
        [gold_tensor[0, 0, 1], gold_tensor[0, 0, 2], gold_tensor[0, 1, 2]]
    )
    assert set(pairs["split"]) == {"train"}


def test_gold_pairs_skip_the_pairs_the_gold_did_not_score():
    groups = proposal_groups().iloc[:1]
    candidates = pd.DataFrame(
        {
            "question_id": ["q1"] * 3,
            "sample_index": [0, 1, 2],
            "response": ["a", "b", "c"],
        }
    )
    gold_tensor = np.full((4, 3, 3), np.nan)
    gold_tensor[:, [0, 1], [1, 0]] = [0.8, 0.2]
    pairs = gold_pairs(groups, candidates, {"q1": gold_tensor})
    assert pairs[["response_a", "response_b"]].drop_duplicates().values.tolist() == [  # pyright: ignore[reportAttributeAccessIssue]
        ["a", "b"]
    ]
    assert len(pairs) == 4


def test_scoring_inputs_sample_pairs_only_in_the_training_splits(monkeypatch):
    monkeypatch.setattr("blackwell_ita.habermas.SCORED_PAIR_COUNTS", {"train": 2})
    groups = proposal_groups()
    candidates = pd.DataFrame(
        {
            "question_id": ["q1"] * 4 + ["q2"] * 4,
            "sample_index": [0, 1, 2, 3] * 2,
            "response": list("abcd") * 2,
        }
    )
    inputs = scoring_inputs(groups, candidates)
    assert inputs == scoring_inputs(groups, candidates)
    assert "scored_pairs" not in inputs["test"]["q2"]
    train_pairs = inputs["train"]["q1"]["scored_pairs"]
    assert len(train_pairs) == 4
    assert all(len(set(pairs)) == 2 for pairs in train_pairs)
    assert all(first < second for pairs in train_pairs for first, second in pairs)


def test_complete_groups_drop_groups_with_a_placeholder_opinion():
    groups = pd.DataFrame(
        {
            "question_id": ["q1", "q2"],
            "opinions": [["a", "b"], ["a", PLACEHOLDER_OPINION]],
        }
    )
    assert complete_groups(groups)["question_id"].tolist() == ["q1"]
