from itertools import islice

import pandas as pd
import pytest

from blackwell_ita.data import (
    PreferencePairDataset,
    assign_prompt_splits,
    evaluation_loader,
    graded_target,
    helpsteer2_pairs,
    length_grouped_batches,
    longest_batch,
    split_datasets,
    training_batches,
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


def test_split_datasets_augment_only_train():
    split_pairs = assign_prompt_splits(many_prompt_pairs(20))
    datasets_by_split = split_datasets(split_pairs, ["overall"])
    for split_name, dataset in datasets_by_split.items():
        pair_count = int((split_pairs["split"] == split_name).sum())
        expected_count = 2 * pair_count if split_name == "train" else pair_count
        assert len(dataset) == expected_count


def test_length_grouped_batches_cover_every_example_once_and_group_by_length():
    lengths = list(range(1000))
    batches = length_grouped_batches(
        lengths, batch_size=10, seed=0, chunk_batch_count=5
    )
    assert sorted(index for batch in batches for index in batch) == lengths
    # Within a chunk of 50 shuffled examples, a sorted batch spans a narrow range
    mean_batch_span = sum(max(batch) - min(batch) for batch in batches) / len(batches)
    assert mean_batch_span < 250
    assert batches == length_grouped_batches(lengths, 10, seed=0, chunk_batch_count=5)
    assert batches != length_grouped_batches(lengths, 10, seed=1, chunk_batch_count=5)


def length_dataset(example_count: int) -> PreferencePairDataset:
    pairs = pd.DataFrame(
        {
            "prompt": ["p" * (index + 1) for index in range(example_count)],
            "response_a": "a",
            "response_b": "b",
            "overall": 1.0,
        }
    )
    return PreferencePairDataset(pairs, ["overall"], augment_presentation_order=False)


def test_training_batches_resume_exactly_where_an_uninterrupted_run_would_be():
    dataset = length_dataset(7)
    uninterrupted_prompts = [
        batch["prompt"] for batch in islice(training_batches(dataset, 3, 0, 0), 9)
    ]
    resumed_prompts = [
        batch["prompt"] for batch in islice(training_batches(dataset, 3, 0, 4), 5)
    ]
    assert resumed_prompts == uninterrupted_prompts[4:]
    # Three batches per epoch, and each epoch reshuffles
    first_epoch = sorted(
        prompt for batch in uninterrupted_prompts[:3] for prompt in batch
    )
    assert first_epoch == sorted(dataset[index]["prompt"] for index in range(7))


def test_longest_batch_and_evaluation_loader_order_by_length():
    dataset = length_dataset(5)
    assert longest_batch(dataset, 2)["prompt"] == ["ppppp", "pppp"]
    assert [batch["prompt"] for batch in evaluation_loader(dataset, 2)] == [
        ["p", "pp"],
        ["ppp", "pppp"],
        ["ppppp"],
    ]


def test_evaluation_loader_can_restrict_to_a_subset():
    dataset = length_dataset(5)
    assert [batch["prompt"] for batch in evaluation_loader(dataset, 2, [4, 0, 2])] == [
        ["p", "ppp"],
        ["ppppp"],
    ]
