"""Tests for the pure logic in the generate_candidates notebook."""

import numpy as np
import pandas as pd
import pytest
from generate_candidates import (
    batch_sizes,
    combine_with_anchors,
    select_anchors,
    token_counts,
)


def pair_row(prompt, response_a, response_b, overall, split="ita_holdout"):
    """Build a minimal pairs row with the artifact schema."""
    return {
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "helpfulness": 0.5,
        "correctness": 0.5,
        "coherence": 0.5,
        "complexity": 0.5,
        "verbosity": 0.5,
        "overall": overall,
        "split": split,
    }


def test_select_anchors_takes_preferred_side():
    """Anchor is response_a when overall > 0.5 and response_b when < 0.5."""
    pairs = pd.DataFrame(
        [pair_row("p1", "a", "b", 0.75), pair_row("p2", "c", "d", 0.0)]
    )
    anchors = select_anchors(pairs)
    assert anchors.to_dict("records") == [
        {"prompt": "p1", "anchor": "a"},
        {"prompt": "p2", "anchor": "d"},
    ]


def test_select_anchors_skips_ties_and_missing():
    """Pairs with overall 0.5 or missing carry no anchor for their prompt."""
    pairs = pd.DataFrame(
        [pair_row("p1", "a", "b", 0.5), pair_row("p2", "c", "d", np.nan)]
    )
    assert select_anchors(pairs).empty


def test_select_anchors_first_decisive_pair_wins():
    """A tied pair defers to a later decisive one; a decisive one is final."""
    pairs = pd.DataFrame(
        [
            pair_row("p1", "a", "b", 0.5),
            pair_row("p1", "c", "d", 1.0),
            pair_row("p1", "e", "f", 0.0),
        ]
    )
    assert select_anchors(pairs).to_dict("records") == [{"prompt": "p1", "anchor": "c"}]


def test_select_anchors_only_uses_the_ita_holdout_split():
    """Pairs from the train and evaluation splits never yield anchors."""
    pairs = pd.DataFrame(
        [
            pair_row("p1", "a", "b", 1.0, split="train"),
            pair_row("p2", "c", "d", 1.0, split="evaluation"),
            pair_row("p3", "e", "f", 1.0),
        ]
    )
    assert select_anchors(pairs)["prompt"].tolist() == ["p3"]


def test_combine_with_anchors_appends_anchor_as_last_index():
    """Samples get indices 0..N-1 per prompt and the anchor rides at N."""
    candidates = pd.DataFrame(
        {
            "prompt": ["p1", "p1", "p2", "p2"],
            "response": ["r0", "r1", "s0", "s1"],
        }
    )
    anchors = pd.DataFrame({"prompt": ["p1", "p2"], "anchor": ["a1", "a2"]})
    combined = combine_with_anchors(candidates, anchors, samples_per_prompt=2)
    assert combined.columns.tolist() == ["prompt", "sample_index", "response"]
    assert combined["prompt"].tolist() == ["p1", "p1", "p1", "p2", "p2", "p2"]
    assert combined["sample_index"].tolist() == [0, 1, 2, 0, 1, 2]
    assert combined["response"].tolist() == ["r0", "r1", "a1", "s0", "s1", "a2"]


def test_combine_with_anchors_rejects_wrong_sample_count():
    """A prompt with a wrong number of samples fails loudly."""
    candidates = pd.DataFrame({"prompt": ["p1"], "response": ["r0"]})
    anchors = pd.DataFrame({"prompt": ["p1"], "anchor": ["a1"]})
    with pytest.raises(AssertionError):
        combine_with_anchors(candidates, anchors, samples_per_prompt=2)


class WordTokenizer:
    """Tokenizer stub splitting on whitespace."""

    def __call__(self, texts: list[str]) -> dict:
        """Return one token id per word for each text."""
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


def test_token_counts_per_text():
    """Counts follow the tokenizer's ids, one entry per input text."""
    assert token_counts(["one two three", "", "one"], WordTokenizer()) == [3, 0, 1]


def test_batch_sizes_cover_total():
    """Batches are full-size except a smaller remainder batch at the end."""
    assert batch_sizes(128, 8) == [8] * 16
    assert batch_sizes(5, 2) == [2, 2, 1]
    assert batch_sizes(3, 8) == [3]
