"""Tests for token-count helpers."""

from blackwell_ita.utils.tokens import token_lengths


class LengthTokenizer:
    """Tokenizer producing one token per character."""

    def __call__(self, texts: list[str]) -> dict[str, list[list[int]]]:
        """Return one token id per character of each text."""
        return {"input_ids": [[0] * len(text) for text in texts]}


def test_lengths_cover_every_text():
    """Every text's tokenized length is returned, in order."""
    assert token_lengths(["a", "abcd", "ab"], LengthTokenizer()) == [1, 4, 2]


def test_empty_texts_give_empty_lengths():
    """An empty collection yields no lengths."""
    assert token_lengths([], LengthTokenizer()) == []
