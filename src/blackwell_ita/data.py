from collections.abc import Iterator
from typing import TypedDict

import datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

HELPSTEER2_ATTRIBUTES = [
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
]
CRITERIA = [*HELPSTEER2_ATTRIBUTES, "overall"]
SPLIT_FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}
PREFERENCES_URL = "hf://datasets/nvidia/HelpSteer2/preference/preference.jsonl.gz"


def graded_target(margin: float) -> float:
    """Map an ordinal rating margin to a graded win probability."""
    if margin >= 2:
        return 1.0
    if margin >= 1:
        return 0.75
    if margin <= -2:
        return 0.0
    if margin <= -1:
        return 0.25
    return 0.5


def helpsteer2_response_pairs(
    responses: pd.DataFrame,
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """Yield response row pairs, paired consecutively within each prompt group."""
    for prompt, prompt_responses in responses.groupby("prompt", sort=False):
        assert len(prompt_responses) % 2 == 0, f"odd response group for {prompt!r}"
        for start in range(0, len(prompt_responses), 2):
            yield prompt_responses.iloc[start], prompt_responses.iloc[start + 1]


def helpsteer2_pairs(
    responses: pd.DataFrame, preferences: pd.DataFrame
) -> pd.DataFrame:
    """Build preference pairs with per-attribute and overall targets."""
    pair_rows = []
    for first_response, second_response in helpsteer2_response_pairs(responses):
        pair_row = {
            "prompt": first_response["prompt"],
            "response_a": first_response["response"],
            "response_b": second_response["response"],
        }
        for attribute in HELPSTEER2_ATTRIBUTES:
            pair_row[attribute] = graded_target(
                first_response[attribute] - second_response[attribute]
            )
        pair_rows.append(pair_row)
    pairs = pd.DataFrame(pair_rows)
    # Positive preference_strength means response_2 is preferred. The
    # (prompt, response_1, response_2) key is unique within each source split
    # (verified against the source file, Aug 2026), so the merge cannot fan out
    overall_preferences = preferences.assign(
        overall=[
            graded_target(-strength) for strength in preferences["preference_strength"]
        ]
    )
    return pairs.merge(
        overall_preferences[["prompt", "response_1", "response_2", "overall"]],
        how="left",
        left_on=["prompt", "response_a", "response_b"],
        right_on=["prompt", "response_1", "response_2"],
    ).drop(columns=["response_1", "response_2"])


def load_helpsteer2_pairs() -> pd.DataFrame:
    """Pairs from both HelpSteer2 halves, which the preference file calls train and val."""
    preferences = pd.read_json(PREFERENCES_URL, lines=True)
    return pd.concat(
        [
            helpsteer2_pairs(
                datasets.load_dataset(  # pyright: ignore[reportArgumentType]
                    "nvidia/HelpSteer2", split=response_split
                ).to_pandas(),  # pyright: ignore[reportAttributeAccessIssue]
                preferences[preferences["split"] == preference_split],  # pyright: ignore[reportArgumentType]
            )
            for response_split, preference_split in (
                ("train", "train"),
                ("validation", "val"),
            )
        ],
        ignore_index=True,
    )


def assign_prompt_splits(pairs: pd.DataFrame, seed: int = 1810) -> pd.DataFrame:
    """Label every pair train, validation or test by prompt, 70/15/15."""
    # Several pairs share a prompt, so a row-level split would leak
    shuffled_prompts = (
        pairs["prompt"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
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
    return pairs.assign(split=pairs["prompt"].map(split_by_prompt))  # pyright: ignore[reportArgumentType]


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


def split_loaders(
    pairs: pd.DataFrame, criteria: list[str], batch_size: int
) -> dict[str, DataLoader]:
    """One loader per split, only train shuffled and order-swap augmented."""
    split_names = set(pairs["split"])
    assert split_names == set(SPLIT_FRACTIONS), split_names
    return {
        split_name: DataLoader(
            PreferencePairDataset(
                pairs[pairs["split"] == split_name],  # pyright: ignore[reportArgumentType]
                criteria,
                augment_presentation_order=split_name == "train",
            ),
            batch_size=batch_size,
            shuffle=split_name == "train",
        )
        for split_name in SPLIT_FRACTIONS
    }


def split_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Prompt and pair counts per split."""
    return (
        pairs.groupby("split")
        .agg(prompts=("prompt", "nunique"), pairs=("prompt", "size"))
        .assign(
            prompt_fraction=lambda summary: (
                summary["prompts"] / summary["prompts"].sum()
            )
        )
        .reindex(list(SPLIT_FRACTIONS))
    )
