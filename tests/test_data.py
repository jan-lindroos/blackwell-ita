import pandas as pd
import pytest
import torch

from blackwell_ita.data import (
    PreferencePairDataset,
    assign_prompt_splits,
    graded_target,
    helpsteer2_pairs,
    split_loaders,
)


def attribute_columns(values: list[int]) -> dict[str, list[int]]:
    """The same ratings for all five HelpSteer2 attributes."""
    return {
        attribute: values
        for attribute in (
            "helpfulness",
            "correctness",
            "coherence",
            "complexity",
            "verbosity",
        )
    }


def test_graded_target_five_level_grid():
    assert [graded_target(margin) for margin in (-3, -2, -1, 0, 1, 2, 4)] == [
        0.0,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.0,
    ]


def test_helpsteer2_pairs_graded_targets_and_overall_sign():
    responses = pd.DataFrame(
        {
            "prompt": ["p", "p", "q", "q"],
            "response": ["r1", "r2", "s1", "s2"],
            "helpfulness": [4, 1, 0, 0],
            "correctness": [2, 2, 0, 0],
            "coherence": [3, 2, 0, 0],
            "complexity": [1, 3, 0, 0],
            "verbosity": [2, 1, 0, 0],
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
    first_pair = pairs.iloc[0]
    assert first_pair["helpfulness"] == 1.0
    assert first_pair["correctness"] == 0.5
    assert first_pair["coherence"] == 0.75
    assert first_pair["complexity"] == 0.0
    assert first_pair["verbosity"] == 0.75
    # Positive preference_strength prefers response_2, so response_a loses
    assert first_pair["overall"] == 0.0
    assert pd.isna(pairs.iloc[1]["overall"])


def test_helpsteer2_pairs_rejects_odd_prompt_groups():
    responses = pd.DataFrame(
        {"prompt": ["p", "p", "p", "q"], "response": ["r1", "r2", "r3", "r4"]}
        | attribute_columns([1, 2, 3, 4])
    )
    preferences = pd.DataFrame(
        columns=["prompt", "response_1", "response_2", "preference_strength"]
    )
    with pytest.raises(AssertionError):
        helpsteer2_pairs(responses, preferences)


def many_prompt_pairs(prompt_count: int) -> pd.DataFrame:
    """Two pairs per prompt with a single criterion column."""
    return pd.DataFrame(
        {
            "prompt": [f"prompt {index // 2}" for index in range(2 * prompt_count)],
            "response_a": "a",
            "response_b": "b",
            "overall": 1.0,
        }
    )


def test_assign_prompt_splits_is_seventy_fifteen_fifteen_by_prompt():
    split_pairs = assign_prompt_splits(many_prompt_pairs(200))
    prompts_per_split = split_pairs.groupby("split")["prompt"].nunique()
    assert prompts_per_split.to_dict() == {"train": 140, "validation": 30, "test": 30}
    assert split_pairs.groupby("prompt")["split"].nunique().max() == 1
    assert split_pairs.equals(assign_prompt_splits(many_prompt_pairs(200)))


def test_dataset_masks_missing_targets():
    pairs = pd.DataFrame(
        {
            "prompt": ["p"],
            "response_a": ["a"],
            "response_b": ["b"],
            "x": [1.0],
            "y": [float("nan")],
        }
    )
    example = PreferencePairDataset(pairs, ["x", "y"], False)[0]
    assert example["target"].tolist() == [1.0, 0.0]
    assert example["mask"].tolist() == [1.0, 0.0]


def test_dataset_augmentation_swaps_responses_and_flips_targets():
    pairs = pd.DataFrame(
        {
            "prompt": ["p"],
            "response_a": ["a"],
            "response_b": ["b"],
            "x": [0.75],
            "y": [float("nan")],
        }
    )
    dataset = PreferencePairDataset(pairs, ["x", "y"], True)
    assert len(dataset) == 2
    swapped_example = dataset[1]
    assert swapped_example["first_response"] == "b"
    assert swapped_example["second_response"] == "a"
    assert swapped_example["target"].tolist() == [0.25, 0.0]
    assert swapped_example["mask"].tolist() == [1.0, 0.0]


def test_split_loaders_augment_and_shuffle_only_train():
    split_pairs = assign_prompt_splits(many_prompt_pairs(20))
    loaders = split_loaders(split_pairs, ["overall"], batch_size=4)
    for split_name, loader in loaders.items():
        pair_count = int((split_pairs["split"] == split_name).sum())
        expected_count = 2 * pair_count if split_name == "train" else pair_count
        assert len(loader.dataset) == expected_count  # pyright: ignore[reportArgumentType]
        assert isinstance(loader.sampler, torch.utils.data.RandomSampler) == (
            split_name == "train"
        )
