"""Token-count helpers for choosing truncation limits."""

import random
from collections.abc import Iterable
from typing import Protocol


class Tokenizer(Protocol):
    """Anything that can encode text into token ids."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        ...


def max_response_tokens(
    responses: Iterable[str],
    tokenizer: Tokenizer,
    sample_size: int | None = 50,
) -> int:
    """Longest tokenized response length, optionally over a random sample."""
    response_list = list(responses)
    if sample_size is not None and len(response_list) > sample_size:
        response_list = random.sample(response_list, sample_size)
    return max(len(tokenizer.encode(response)) for response in response_list)
