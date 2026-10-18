import math
from collections.abc import Iterator
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

SPLIT_FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def assign_prompt_splits(
    pairs: pd.DataFrame, group_column: str = "prompt", seed: int = 1810
) -> pd.DataFrame:
    """Label every pair train, validation or test by ``group_column``, 70/15/15."""
    # Several pairs share a prompt, so a row-level split would leak
    shuffled_prompts = (
        pairs[group_column]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .tolist()
    )
    train_count = round(len(shuffled_prompts) * SPLIT_FRACTIONS["train"])
    validation_count = round(len(shuffled_prompts) * SPLIT_FRACTIONS["validation"])
    split_by_prompt = {
        prompt: "train"
        if position < train_count
        else "validation"
        if position < train_count + validation_count
        else "test"
        for position, prompt in enumerate(shuffled_prompts)
    }
    return pairs.assign(split=pairs[group_column].map(split_by_prompt))  # pyright: ignore[reportArgumentType]


class Example(TypedDict):
    """A preference pair with per-criterion targets and a validity mask."""

    prompt: str
    first_response: str
    second_response: str
    target: torch.Tensor
    mask: torch.Tensor


class Batch(TypedDict):
    """A collated batch of preference pairs."""

    prompt: list[str]
    first_response: list[str]
    second_response: list[str]
    target: torch.Tensor
    mask: torch.Tensor


class PreferencePairDataset(Dataset):
    """Preference pairs from a dataframe, optionally augmented by order swaps."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        criteria: list[str],
        augment_presentation_order: bool,
    ) -> None:
        """Build examples, masking missing targets and optionally swapping order."""
        self.examples: list[Example] = []
        for _, pair_row in pairs.iterrows():
            targets = torch.tensor(
                [pair_row[criterion] for criterion in criteria], dtype=torch.float32
            )
            mask = torch.isfinite(targets).float()
            targets = torch.nan_to_num(targets)
            self.examples.append(
                {  # pyright: ignore[reportArgumentType]
                    "prompt": pair_row["prompt"],
                    "first_response": pair_row["response_a"],
                    "second_response": pair_row["response_b"],
                    "target": targets,
                    "mask": mask,
                }
            )
            if augment_presentation_order:
                self.examples.append(
                    {  # pyright: ignore[reportArgumentType]
                        "prompt": pair_row["prompt"],
                        "first_response": pair_row["response_b"],
                        "second_response": pair_row["response_a"],
                        "target": (1.0 - targets) * mask,
                        "mask": mask,
                    }
                )

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        """Return the example at ``index``."""
        return self.examples[index]


def example_length(example: Example) -> int:
    """Character count of an example's texts, a proxy for its token count."""
    return (
        len(example["prompt"])
        + len(example["first_response"])
        + len(example["second_response"])
    )


def length_grouped_batches(
    lengths: list[int], batch_size: int, seed: int, chunk_batch_count: int = 64
) -> list[list[int]]:
    """Shuffled batches of similar-length examples, fixed by ``seed``.

    Examples are shuffled, sorted by length within chunks of
    ``chunk_batch_count`` batches, cut into batches, and the batches shuffled.
    """
    random_generator = np.random.default_rng(seed)
    shuffled_indices = random_generator.permutation(len(lengths)).tolist()
    chunk_size = batch_size * chunk_batch_count
    batches = []
    for chunk_start in range(0, len(shuffled_indices), chunk_size):
        chunk_indices = sorted(
            shuffled_indices[chunk_start : chunk_start + chunk_size],
            key=lambda index: lengths[index],
        )
        batches.extend(
            chunk_indices[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(chunk_indices), batch_size)
        )
    return [
        batches[position] for position in random_generator.permutation(len(batches))
    ]


def training_batches(
    dataset: PreferencePairDataset, batch_size: int, seed: int, start_step: int
) -> Iterator[Batch]:
    """Endless length-grouped training batches, starting at ``start_step``.

    Epoch ``e`` uses seed ``seed + e``, so a resumed run sees exactly the
    batches an uninterrupted one would.
    """
    lengths = [example_length(example) for example in dataset.examples]
    batches_per_epoch = math.ceil(len(lengths) / batch_size)
    epoch, position = divmod(start_step, batches_per_epoch)
    while True:
        for batch_indices in length_grouped_batches(lengths, batch_size, seed + epoch)[
            position:
        ]:
            yield default_collate([dataset[index] for index in batch_indices])
        epoch += 1
        position = 0


def longest_batch(dataset: PreferencePairDataset, batch_size: int) -> Batch:
    """The ``batch_size`` longest examples, the worst case for memory."""
    longest_indices = sorted(
        range(len(dataset)),
        key=lambda index: example_length(dataset[index]),
        reverse=True,
    )[:batch_size]
    return default_collate([dataset[index] for index in longest_indices])


def evaluation_loader(
    dataset: PreferencePairDataset,
    batch_size: int,
    example_indices: list[int] | None = None,
) -> DataLoader:
    """Batches sorted by length, so each pads only to similar examples.

    ``example_indices`` restricts the loader to those examples.
    """
    sorted_indices = sorted(
        range(len(dataset)) if example_indices is None else example_indices,
        key=lambda index: example_length(dataset[index]),
    )
    return DataLoader(
        dataset,
        batch_sampler=[
            sorted_indices[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(sorted_indices), batch_size)
        ],
    )


def split_datasets(
    pairs: pd.DataFrame, criteria: list[str]
) -> dict[str, PreferencePairDataset]:
    """One dataset per split, only train order-swap augmented."""
    split_names = set(pairs["split"])
    assert split_names == set(SPLIT_FRACTIONS), split_names
    return {
        split_name: PreferencePairDataset(
            pairs[pairs["split"] == split_name],  # pyright: ignore[reportArgumentType]
            criteria,
            augment_presentation_order=split_name == "train",
        )
        for split_name in SPLIT_FRACTIONS
    }


def split_summary(pairs: pd.DataFrame, group_column: str = "prompt") -> pd.DataFrame:
    """Group and pair counts per split."""
    return (
        pairs.groupby("split")
        .agg(groups=(group_column, "nunique"), pairs=(group_column, "size"))
        .assign(
            group_fraction=lambda summary: summary["groups"] / summary["groups"].sum()
        )
        .reindex(list(SPLIT_FRACTIONS))
    )
