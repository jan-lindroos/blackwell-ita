import numpy as np
import pandas as pd
import pytest

from blackwell_ita.habermas import (
    candidate_sets,
    consensus_prompt,
    deliberation_groups,
    diverse_candidate_prompts,
    generate_habermas_candidates,
    habermas_pairs,
    habermas_prompt,
    human_preference_tensor,
    human_track_results,
    model_track_results,
    rank_preference,
    usable_rankings,
)


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
        "ratings.agreement": np.array(["AGREE", "MOCK", "STRONGLY_DISAGREE"]),
        "metadata.provenance": "HUMAN_CITIZEN",
        "rankings.metadata.status": "COMPLETED",
    } | overrides


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
    assert ranking["agreements"][0] == 6.0
    assert np.isnan(ranking["agreements"][1])
    assert ranking["agreements"][2] == 1.0


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
            "agreements": [[7.0, 4.0, 4.0], [1.0, np.nan, 7.0]],
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


def test_human_preference_tensor_is_skew_symmetric_with_half_diagonal():
    tensor = human_preference_tensor([[0, 1, 1], [2, 1, 0]])
    assert tensor.shape == (2, 3, 3)
    assert np.allclose(tensor + tensor.transpose(0, 2, 1), 1.0)
    assert tensor[0, 0, 1] == 1.0
    assert tensor[1, 0, 2] == 0.0


def test_candidate_sets_group_participants_who_saw_the_same_statements():
    sets = candidate_sets(ranking_frame(), "test")
    assert len(sets) == 1
    assert sets.iloc[0]["set_id"] == "r_0"
    assert sets.iloc[0]["opinions"] == ["Yes.", "No."]
    assert candidate_sets(ranking_frame(), "train").empty


def test_deliberation_groups_pick_four_participants_and_their_top_statement():
    rankings = pd.DataFrame(
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
                "agreements": [1.0, 7.0, 4.0],
                "split": "test",
            }
            for question in range(3)
            for participant in range(5)
        ]
    )
    groups = deliberation_groups(rankings, "test", group_count=2)
    assert len(groups) == 2
    assert groups["question_id"].is_unique
    assert all(len(opinions) == 4 for opinions in groups["opinions"])
    assert set(groups["anchor"]) == {"best"}


def test_consensus_prompt_numbers_every_opinion():
    prompt = consensus_prompt("Should we?", ["Yes.", "No."])
    assert "Question: Should we?" in prompt
    assert "Participant 1: Yes." in prompt
    assert "Participant 2: No." in prompt


def proposal_groups():
    return pd.DataFrame(
        [
            {"question_id": q, "question": q + "?", "opinions": ["A", "B", "C", "D"]}
            for q in ["q1", "q2"]
        ]
    )


def test_diverse_pool_preserves_balance_and_target_identity():
    plan = diverse_candidate_prompts(proposal_groups())
    pd.testing.assert_frame_equal(plan, diverse_candidate_prompts(proposal_groups()))
    for _, group in plan.groupby("question_id"):
        assert group.sample_index.tolist() == list(range(16))
        assert group.candidate_kind.value_counts().to_dict() == {
            "consensus": 4,
            "participant": 4,
            "pair": 6,
            "common_ground": 1,
            "alternative_compromise": 1,
        }
        pairs = group[group.candidate_kind == "pair"].target_participants
        assert {tuple(sorted(pair)) for pair in pairs} == {
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        }
        orders = np.array(group.opinion_order.tolist())
        for position in range(4):
            assert np.bincount(orders[:, position])[1:].tolist() == [4, 4, 4, 4]
    for row in plan.to_dict("records"):
        assert "words" not in row["prompt"]
        assert "paragraph" not in row["prompt"]
        for i, opinion in enumerate(["A", "B", "C", "D"], 1):
            assert f"Participant {i}: {opinion}" in row["prompt"]
        if row["target_participants"]:
            assert (
                "participants " + ", ".join(map(str, row["target_participants"])) + "."
                in row["prompt"]
            )


def test_generation_maps_local_outputs_to_fixed_plan(monkeypatch):
    def fake_generate(model_name, prompts, samples_per_prompt, device):
        assert model_name == "mock"
        assert samples_per_prompt == 1
        assert device == "cpu"
        # Reverse the output to catch accidental positional alignment.
        return pd.DataFrame(
            [
                {"prompt_index": i, "sample_index": 0, "response": prompt}
                for i, prompt in reversed(list(enumerate(prompts)))
            ]
        )

    monkeypatch.setattr("blackwell_ita.habermas.generate_responses", fake_generate)
    for strategy in ["diverse_v2", "consensus"]:
        result = generate_habermas_candidates(
            "mock", proposal_groups(), "cpu", strategy
        )
        assert len(result) == 32
        assert result.response.equals(result.prompt)
        assert result.strategy.eq(strategy).all()
        assert not result.duplicated(["question_id", "sample_index"]).any()
        for row in result.to_dict("records"):
            assert f"Question: {row['question_id']}?" in row["response"]
            assert "paragraph" not in row["prompt"]


def test_diverse_generation_rejects_wrong_group_sizes():
    with pytest.raises(ValueError, match="four participants"):
        diverse_candidate_prompts(
            pd.DataFrame(
                [{"question_id": "q", "question": "Q?", "opinions": ["A", "B"]}]
            )
        )


def test_human_track_results_score_selections_on_human_rankings():
    sets = candidate_sets(ranking_frame(), "test")
    # The predicted tensor copies participant one's ranking for both heads
    predicted_tensor = human_preference_tensor([[0, 1, 1], [0, 1, 1]])
    results = human_track_results(sets, {"r_0": predicted_tensor}, "model")
    nash_row = results[results["method"] == "scalarised_nash"].iloc[0]
    # Statement a: participant one wins 2.5 of 3, participant two 0.5 of 3
    assert nash_row["win_rate_rawlsian_welfare"] == pytest.approx(0.5 / 3, abs=1e-5)
    assert nash_row["win_rate_utilitarian_welfare"] == pytest.approx(0.5, abs=1e-5)
    # Participant two has a missing agreement and is left out
    assert nash_row["agreement_rawlsian_welfare"] == pytest.approx(7.0, abs=1e-4)
    assert results["evidence"].eq("human ground truth").all()  # pyright: ignore[reportGeneralTypeIssues]


def test_model_track_results_exclude_the_anchor_from_selection():
    # Two participants and a pool of two plus the anchor, which the selector loves
    selector_tensor = human_preference_tensor([[1, 2, 0], [1, 2, 0]])
    grader_tensor = human_preference_tensor([[0, 1, 2], [1, 0, 2]])
    groups = pd.DataFrame({"question_id": ["q"]})
    results = model_track_results(
        groups, {"q": selector_tensor}, {"q": grader_tensor}, "backbone", "a", "b"
    )
    blackwell_row = results[results["method"] == "blackwell"].iloc[0]
    # The selector prefers candidate 0 among the pool, which beats the anchor
    # for both participants under the grader
    assert blackwell_row["anchor_rawlsian_welfare"] == pytest.approx(1.0, abs=1e-5)
    assert blackwell_row["evidence"] == "model-based proxy, graded by b"
