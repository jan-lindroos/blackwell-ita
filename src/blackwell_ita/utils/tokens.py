"""Token-count helpers for choosing truncation limits."""

from collections.abc import Iterable

from transformers import PreTrainedTokenizerBase


def token_lengths(texts: Iterable[str], tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Tokenized length of every text, via one batched tokenizer call."""
    return [len(ids) for ids in tokenizer(list(texts))["input_ids"]]
