"""Tests for token-count helpers."""

import pytest

import utils.tokens
from utils.tokens import max_response_tokens


class LengthTokenizer:
    """Tokenizer producing one token per character."""

    def encode(self, text: str) -> list[int]:
        """Return one token id per character."""
        return [0] * len(text)


def test_max_over_all_responses_when_sampling_disabled():
    """Without sampling, every response is considered."""
    responses = ["a", "abcd", "ab"]
    assert max_response_tokens(responses, LengthTokenizer(), sample_size=None) == 4


def test_small_collections_are_not_sampled():
    """Collections below the sample size are used in full."""
    assert max_response_tokens(["a", "abcd"], LengthTokenizer()) == 4


def test_sampling_limits_responses_considered(monkeypatch):
    """Sampling restricts the maximum to the sampled responses."""
    monkeypatch.setattr(
        utils.tokens.random, "sample", lambda population, k: population[:k]
    )
    responses = ["x" * length for length in range(1, 101)]
    assert max_response_tokens(responses, LengthTokenizer(), sample_size=10) == 10


def test_empty_responses_raise():
    """An empty collection raises rather than returning a default."""
    with pytest.raises(ValueError):
        max_response_tokens([], LengthTokenizer())
