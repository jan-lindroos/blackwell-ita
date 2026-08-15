"""Token-count helpers for choosing truncation limits."""

from collections.abc import Iterable, Mapping
from typing import Protocol


class Tokenizer(Protocol):
    """Anything that can batch-encode texts into token ids."""

    def __call__(self, texts: list[str]) -> Mapping[str, list[list[int]]]:
        """Encode texts into a mapping carrying per-text ``input_ids``."""
        ...


def token_lengths(texts: Iterable[str], tokenizer: Tokenizer) -> list[int]:
    """Tokenized length of every text, via one batched tokenizer call."""
    return [len(ids) for ids in tokenizer(list(texts))["input_ids"]]
