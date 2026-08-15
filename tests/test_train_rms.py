"""Tests for reward model training utilities."""

from typing import cast

import pandas as pd
import pytest
import torch

from blackwell_ita.train_rms import (
    Batch,
    PreferencePairDataset,
    RewardModelBase,
    bt_text,
    build_loaders,
    default_device,
    evaluate_loss,
    load_reward_model,
    masked_binary_cross_entropy,
    pairwise_text,
    train_until_no_improvement,
)

CRITERION_COLUMNS = ["first_criterion", "second_criterion"]


def pairs_frame() -> pd.DataFrame:
    """Two preference pairs, one with a missing criterion target."""
    return pd.DataFrame(
        {
            "prompt": ["p1", "p2"],
            "response_a": ["a1", "a2"],
            "response_b": ["b1", "b2"],
            "first_criterion": [1.0, 0.5],
            "second_criterion": [0.0, None],
        }
    )


def test_dataset_masks_missing_targets():
    """Missing targets are zeroed and masked out."""
    dataset = PreferencePairDataset(pairs_frame(), CRITERION_COLUMNS, False)
    assert len(dataset) == 2
    assert dataset[0]["target"].tolist() == [1.0, 0.0]
    assert dataset[0]["mask"].tolist() == [1.0, 1.0]
    assert dataset[1]["target"].tolist() == [0.5, 0.0]
    assert dataset[1]["mask"].tolist() == [1.0, 0.0]


def test_dataset_augmentation_swaps_responses_and_flips_targets():
    """Augmented examples swap the responses and flip unmasked targets."""
    dataset = PreferencePairDataset(pairs_frame(), CRITERION_COLUMNS, True)
    assert len(dataset) == 4
    original, augmented = dataset[0], dataset[1]
    assert augmented["first_response"] == original["second_response"]
    assert augmented["second_response"] == original["first_response"]
    assert augmented["target"].tolist() == [0.0, 1.0]
    assert augmented["mask"].tolist() == original["mask"].tolist()
    # A flipped masked-out target stays zeroed rather than becoming 1 - 0
    assert dataset[3]["target"].tolist() == [0.5, 0.0]
    assert dataset[3]["mask"].tolist() == [1.0, 0.0]


def test_masked_binary_cross_entropy_ignores_masked_entries():
    """Masked entries do not contribute to the loss."""
    logits = torch.tensor([[2.0, -100.0]])
    batch = cast(
        Batch, {"target": torch.tensor([[1.0, 1.0]]), "mask": torch.tensor([[1.0, 0.0]])}
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(2.0), torch.tensor(1.0)
    )
    assert torch.isclose(masked_binary_cross_entropy(logits, batch, "cpu"), expected)


def test_text_builders():
    """Text builders join prompts and responses with the expected markers."""
    assert bt_text("p", "r") == "p\n\nr"
    assert pairwise_text("p", "r1", "r2") == "p\n\n[RESPONSE 1]\nr1\n\n[RESPONSE 2]\nr2"


def test_build_loaders_splits_by_prompt_and_augments_only_training_data():
    """Loaders split prompts disjointly and augment only the training set."""
    # Three pairs per prompt: a row-level split would leak prompts across the
    # boundary here, where a one-pair-per-prompt fixture could not catch it
    frame = pd.DataFrame(
        {
            "prompt": [f"p{index // 3}" for index in range(15)],
            "response_a": [f"a{index}" for index in range(15)],
            "response_b": [f"b{index}" for index in range(15)],
            "first_criterion": [1.0] * 15,
            "second_criterion": [0.0] * 15,
        }
    )
    train_loader, validation_loader = build_loaders(
        frame,
        CRITERION_COLUMNS,
        batch_size=4,
        validation_fraction=0.2,
        augment_presentation_order=True,
        seed=0,
    )
    assert isinstance(train_loader.dataset, PreferencePairDataset)
    assert isinstance(validation_loader.dataset, PreferencePairDataset)
    assert len(train_loader.dataset) == 24
    assert len(validation_loader.dataset) == 3
    train_prompts = {example["prompt"] for example in train_loader.dataset}
    validation_prompts = {example["prompt"] for example in validation_loader.dataset}
    assert len(validation_prompts) == 1
    assert not train_prompts & validation_prompts
    assert len(train_prompts | validation_prompts) == 5


class FixedLossModel(torch.nn.Module):
    """Model that returns a scripted sequence of losses."""

    def __init__(self, losses: list[float]) -> None:
        """Store the scripted losses."""
        super().__init__()
        self.losses = iter(losses)

    def compute_loss(self, batch: dict, device: str) -> torch.Tensor:
        """Return the next scripted loss."""
        return torch.tensor(next(self.losses))


def test_evaluate_loss_averages_batches():
    """Evaluation averages the per-batch losses."""
    model = cast(RewardModelBase, FixedLossModel([1.0, 3.0]))
    assert evaluate_loss(model, cast(list[Batch], [{}, {}]), "cpu") == 2.0


class ScriptedModel(torch.nn.Module):
    """Reports scripted validation losses and stamps the step into its weight."""

    def __init__(self, validation_losses: list[float]) -> None:
        """Store the scripted validation losses."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.validation_losses = iter(validation_losses)
        self.steps_trained = 0

    def compute_loss(self, batch: dict, device: str) -> torch.Tensor:
        """Stamp the step when training; return the next scripted loss otherwise."""
        if self.training:
            self.steps_trained += 1
            self.weight.data.fill_(float(self.steps_trained))
            return self.weight * 0.0
        return self.weight * 0.0 + next(self.validation_losses)


def test_train_until_no_improvement_stops_and_restores_best_state():
    """Training stops after two non-improving rounds and restores the best weights."""
    # Round 3 is a single bad round that round 4's improvement forgives;
    # rounds 5 and 6 exhaust the patience of 2
    model = ScriptedModel([3.0, 2.0, 2.5, 1.5, 1.6, 1.7])
    best = train_until_no_improvement(
        cast(RewardModelBase, model),
        train_loader=cast(list[Batch], [{}]),
        validation_loader=cast(list[Batch], [{}]),
        learning_rate=0.0,
        warmup_steps=1,
        device="cpu",
        steps_per_epoch=1,
    )
    assert best == 1.5
    assert model.steps_trained == 6
    # The weight carries the step number, so restoring the best state
    # (step 4) undoes the final two non-improving rounds
    assert model.weight.item() == 4.0


def test_train_until_no_improvement_cycles_loader_for_longer_rounds():
    """A round longer than the loader cycles it instead of ending early."""
    model = ScriptedModel([2.0, 3.0, 3.1])
    best = train_until_no_improvement(
        cast(RewardModelBase, model),
        train_loader=cast(list[Batch], [{}]),
        validation_loader=cast(list[Batch], [{}]),
        learning_rate=0.0,
        warmup_steps=1,
        device="cpu",
        steps_per_epoch=3,
    )
    assert best == 2.0
    # Three rounds of three steps each, from a single-batch loader
    assert model.steps_trained == 9
    assert model.weight.item() == 3.0


def test_default_device_prefers_cuda(monkeypatch):
    """The default device is cuda when it is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"


def test_default_device_falls_back_to_cpu(monkeypatch):
    """The default device is cpu without cuda or mps."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == "cpu"


def test_load_reward_model_rejects_unknown_class(tmp_path):
    """Loading a checkpoint with an unknown class name raises KeyError."""
    path = tmp_path / "model.pt"
    torch.save({"class_name": "MysteryModel"}, path)
    with pytest.raises(KeyError):
        load_reward_model(path)
