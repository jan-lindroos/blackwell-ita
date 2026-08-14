"""Tests for preference pair construction."""

import pandas as pd
import pytest

from blackwell_ita.human_prefs import (
    annotator_group,
    community_alignment_anchors,
    community_alignment_pairs,
    graded_target,
    helpsteer2_anchors,
    helpsteer2_pairs,
    prompt_splits,
)


def test_graded_target_five_level_grid():
    """Margins map onto the symmetric five-level grid."""
    assert [graded_target(margin) for margin in (-3, -2, -1, 0, 1, 2, 4)] == [
        0.0,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.0,
    ]


def test_annotator_group_rejects_unknown_age_or_country():
    """Known groups map to names; unknown ages and countries fail the assert."""
    assert annotator_group("united states", "55+") == "united_states_35_plus"
    assert annotator_group("india", "18-34") == "india_18_34"
    with pytest.raises(AssertionError):
        annotator_group("united states", "25-34")
    with pytest.raises(AssertionError):
        annotator_group("france", "18-34")


def test_helpsteer2_pairs_graded_targets_and_overall_sign():
    """Attribute margins grade correctly and the overall sign is oriented."""
    responses = pd.DataFrame(
        {
            "prompt": ["p", "p"],
            "response": ["r1", "r2"],
            "helpfulness": [4, 1],
            "correctness": [2, 2],
            "coherence": [3, 2],
        }
    )
    preferences = pd.DataFrame(
        {
            "prompt": ["p"],
            "response_1": ["r1"],
            "response_2": ["r2"],
            "preference_strength": [2],
        }
    )
    pairs = helpsteer2_pairs(responses, preferences)
    row = pairs.iloc[0]
    assert row["helpfulness"] == 1.0
    assert row["correctness"] == 0.5
    assert row["coherence"] == 0.75
    # Positive strength means response_2 preferred, so response_a loses
    assert row["overall"] == 0.0


def test_helpsteer2_pairs_rejects_misaligned_groups():
    """An odd-sized prompt group raises instead of silently pairing wrongly."""
    responses = pd.DataFrame(
        {
            "prompt": ["p", "p", "p", "q"],
            "response": ["r1", "r2", "r3", "r4"],
            "helpfulness": [1, 2, 3, 4],
            "correctness": [1, 2, 3, 4],
            "coherence": [1, 2, 3, 4],
        }
    )
    preferences = pd.DataFrame(
        {
            "prompt": ["p"],
            "response_1": ["r1"],
            "response_2": ["r2"],
            "preference_strength": [1],
        }
    )
    with pytest.raises(AssertionError):
        helpsteer2_pairs(responses, preferences)


def test_helpsteer2_anchors_require_strict_overall_winner():
    """Anchors follow the preference sign; unannotated and tied pairs skip."""
    validation = pd.DataFrame(
        {
            "prompt": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "response": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )
    preferences = pd.DataFrame(
        {
            "prompt": ["p1", "p3", "p4"],
            "response_1": ["a", "e", "g"],
            "response_2": ["b", "f", "h"],
            "preference_strength": [2, -1, 0],
        }
    )
    # p2 has no annotation and p4 ties, so neither yields an anchor; positive
    # strength picks the second response, negative the first
    assert helpsteer2_anchors(validation, preferences) == {"p1": "b", "p3": "e"}
    assert helpsteer2_anchors(validation, preferences, count=1) == {"p1": "b"}


def conversation_row(prompt, responses, preferred, country="india", age="18-34"):
    """Build a minimal Community Alignment conversation row."""
    return {
        "assigned_lang": "en",
        "annotator_country": country,
        "annotator_age": age,
        "first_turn_prompt": prompt,
        "first_turn_preferred_response": preferred,
        **{
            f"first_turn_response_{letter}": response
            for letter, response in zip("abcd", responses, strict=True)
        },
    }


def test_community_alignment_pairs_skips_identical_responses():
    """Pairs of a response against identical text are dropped."""
    conversations = pd.DataFrame(
        [conversation_row("p", ["same", "same", "other", None], "response_a")]
    )
    pairs = community_alignment_pairs(conversations)
    assert len(pairs) == 1
    assert (pairs["response_a"] != pairs["response_b"]).all()


def test_community_alignment_pairs_pools_disagreeing_annotators():
    """Disagreeing annotators average to 0.5 with per-group targets kept."""
    conversations = pd.DataFrame(
        [
            conversation_row("p", ["x", "y", None, None], "response_a"),
            conversation_row("p", ["x", "y", None, None], "response_b", country="united states"),
        ]
    )
    pairs = community_alignment_pairs(conversations)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["overall"] == 0.5
    assert row["india_18_34"] == (1.0 if row["response_a"] == "x" else 0.0)
    assert row["united_states_18_34"] == (0.0 if row["response_a"] == "x" else 1.0)


def test_community_alignment_anchors_pool_win_fractions():
    """The anchor is the response with the best pooled win fraction."""
    # For prompt p: x earns (0.25 + 0.5) / 2, y earns (0.75 + 1.0) / 2 and
    # z earns (0.5 + 0.0) / 2, so y is the anchor; q is not held out
    pairs = pd.DataFrame(
        {
            "prompt": ["p", "p", "p", "q"],
            "response_a": ["x", "x", "y", "u"],
            "response_b": ["y", "z", "z", "v"],
            "overall": [0.25, 0.5, 1.0, 0.0],
        }
    )
    assert community_alignment_anchors(pairs, ["p"]) == {"p": "y"}


def test_prompt_splits_are_deterministic_and_disjoint():
    """Splits are reproducible, disjoint, and cover every prompt."""
    prompts = pd.Series([f"prompt {index}" for index in range(20)] * 3)
    held_out, first_half, second_half = prompt_splits(prompts, evaluation_count=4)
    assert (held_out, first_half, second_half) == prompt_splits(
        prompts, evaluation_count=4
    )
    assert not set(held_out) & set(first_half)
    assert not set(held_out) & set(second_half)
    assert not set(first_half) & set(second_half)
    assert len(held_out) + len(first_half) + len(second_half) == 20
