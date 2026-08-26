"""Tests for the train_prefs notebook's preference and training utilities."""

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
import train_prefs
from train_prefs import (
    Batch,
    PairwisePreferenceModel,
    PreferencePairDataset,
    build_loaders,
    default_device,
    evaluate_loss,
    graded_target,
    helpsteer2_pairs,
    masked_binary_cross_entropy,
    pairwise_text,
    per_criterion_metrics,
    preference_tensor,
    prompt_splits,
    train_until_no_improvement,
    truncated_pairwise_text,
)
from transformers import PreTrainedTokenizerBase

CRITERION_COLUMNS = ["first_criterion", "second_criterion"]


def test_graded_target_five_level_grid():
    """Margins map onto the symmetric five-level grid."""
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
    """All five attribute margins grade correctly and the overall sign is oriented."""
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
    row = pairs.iloc[0]
    assert row["helpfulness"] == 1.0
    assert row["correctness"] == 0.5
    assert row["coherence"] == 0.75
    assert row["complexity"] == 0.0
    assert row["verbosity"] == 0.75
    # Positive preference_strength prefers response_2 (response_b), so
    # response_a's overall win target is 0
    assert row["overall"] == 0.0
    # A pair without a preference annotation keeps a missing overall target
    assert pd.isna(pairs.iloc[1]["overall"])


def test_helpsteer2_pairs_rejects_misaligned_groups():
    """An odd-sized prompt group raises instead of silently pairing wrongly."""
    responses = pd.DataFrame(
        {
            "prompt": ["p", "p", "p", "q"],
            "response": ["r1", "r2", "r3", "r4"],
            "helpfulness": [1, 2, 3, 4],
            "correctness": [1, 2, 3, 4],
            "coherence": [1, 2, 3, 4],
            "complexity": [1, 2, 3, 4],
            "verbosity": [1, 2, 3, 4],
        }
    )
    preferences = pd.DataFrame(
        {
            "prompt": ["p"],
            "response_1": ["r1"],
            "response_2": ["r2"],
            "preference_strength": [1],
        }
    )
    with pytest.raises(AssertionError):
        helpsteer2_pairs(responses, preferences)


def test_prompt_splits_are_deterministic_and_disjoint():
    """Splits are reproducible, disjoint, and cover every prompt."""
    prompts = pd.Series([f"prompt {index}" for index in range(20)] * 3)
    held_out, first_half, second_half = prompt_splits(prompts, evaluation_count=4)
    assert (held_out, first_half, second_half) == prompt_splits(
        prompts, evaluation_count=4
    )
    assert not set(held_out) & set(first_half)
    assert not set(held_out) & set(second_half)
    assert not set(first_half) & set(second_half)
    assert len(held_out) + len(first_half) + len(second_half) == 20


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
        Batch,
        {"target": torch.tensor([[1.0, 1.0]]), "mask": torch.tensor([[1.0, 0.0]])},
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(2.0), torch.tensor(1.0)
    )
    assert torch.isclose(masked_binary_cross_entropy(logits, batch, "cpu"), expected)


def test_pairwise_text_joins_with_markers():
    """The pairwise text carries both responses behind their markers."""
    assert pairwise_text("p", "r1", "r2") == "p\n\n[RESPONSE 1]\nr1\n\n[RESPONSE 2]\nr2"


class CharTokenizer:
    """Tokenizer with one token per character and lossless decoding."""

    def encode(self, text: str) -> list[int]:
        """Return one token id per character."""
        return [ord(character) for character in text]

    def decode(self, ids: list[int]) -> str:
        """Rebuild text from character token ids."""
        return "".join(chr(token_id) for token_id in ids)


CHAR_TOKENIZER = cast(PreTrainedTokenizerBase, CharTokenizer())


def test_truncated_pairwise_text_leaves_short_pairs_alone():
    """A pair within the token budget is formatted unchanged."""
    assert truncated_pairwise_text(
        "p", "aaa", "bbb", CHAR_TOKENIZER, max_tokens=100
    ) == pairwise_text("p", "aaa", "bbb")


def test_truncated_pairwise_text_splits_budget_equally():
    """Two overlong responses each keep half of the remaining budget."""
    # pairwise_text("p", "", "") is 31 characters, leaving a budget of 10
    assert truncated_pairwise_text(
        "p", "a" * 20, "b" * 20, CHAR_TOKENIZER, max_tokens=41
    ) == pairwise_text("p", "a" * 5, "b" * 5)


def test_truncated_pairwise_text_donates_surplus_to_longer_response():
    """A response shorter than its half donates the surplus to the other."""
    assert truncated_pairwise_text(
        "p", "a" * 3, "b" * 20, CHAR_TOKENIZER, max_tokens=41
    ) == pairwise_text("p", "a" * 3, "b" * 7)
    assert truncated_pairwise_text(
        "p", "a" * 20, "b" * 3, CHAR_TOKENIZER, max_tokens=41
    ) == pairwise_text("p", "a" * 7, "b" * 3)


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
    model = cast(PairwisePreferenceModel, FixedLossModel([1.0, 3.0]))
    assert evaluate_loss(model, cast(list[Batch], [{}, {}]), "cpu") == 2.0


class FixedLogitsModel(torch.nn.Module):
    """Model returning a scripted sequence of per-batch logits."""

    def __init__(self, logits: list[torch.Tensor]) -> None:
        """Store the scripted logits."""
        super().__init__()
        self.logits = iter(logits)

    def batch_logits(self, batch: dict, device: str) -> torch.Tensor:
        """Return the next scripted logits."""
        return next(self.logits)


def test_per_criterion_metrics_masks_and_thresholds():
    """Metrics respect the mask and count only decisive unmasked entries."""
    batches = cast(
        list[Batch],
        [
            {
                "target": torch.tensor([[1.0, 0.0], [0.0, 0.5]]),
                "mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            },
            {
                "target": torch.tensor([[1.0, 0.4]]),
                "mask": torch.tensor([[1.0, 1.0]]),
            },
        ],
    )
    model = FixedLogitsModel(
        [torch.tensor([[2.0, -1.0], [1.0, 3.0]]), torch.tensor([[-2.0, 1.0]])]
    )
    metrics = per_criterion_metrics(
        cast(PairwisePreferenceModel, model), batches, ["c1", "c2"], "cpu"
    )
    # c1: all three entries are unmasked and decisive; only the first is
    # predicted on the right side of zero
    expected_c1_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0, 1.0, -2.0]), torch.tensor([1.0, 0.0, 1.0])
    )
    assert metrics["c1"]["loss"] == pytest.approx(expected_c1_loss.item())
    assert metrics["c1"]["decisive_accuracy"] == pytest.approx(1 / 3)
    assert metrics["c1"]["decisive_count"] == 3
    # c2: the masked 0.5 entry is excluded everywhere and the unmasked 0.4
    # entry counts towards the loss but is not decisive
    expected_c2_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 0.4])
    )
    assert metrics["c2"]["loss"] == pytest.approx(expected_c2_loss.item())
    assert metrics["c2"]["decisive_accuracy"] == 1.0
    assert metrics["c2"]["decisive_count"] == 1


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
        cast(PairwisePreferenceModel, model),
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
        cast(PairwisePreferenceModel, model),
        train_loader=cast(list[Batch], [{}]),
        validation_loader=cast(list[Batch], [{}]),
        learning_rate=0.0,
        warmup_steps=1,
        device="cpu",
        steps_per_epoch=3,
    )
    assert best == 2.0
    assert model.steps_trained == 9
    assert model.weight.item() == 3.0


def test_train_until_no_improvement_raises_without_a_best_state():
    """A run whose every validation loss is NaN raises instead of returning."""
    model = ScriptedModel([float("nan"), float("nan")])
    with pytest.raises(RuntimeError, match="nan"):
        train_until_no_improvement(
            cast(PairwisePreferenceModel, model),
            train_loader=cast(list[Batch], [{}]),
            validation_loader=cast(list[Batch], [{}]),
            learning_rate=0.0,
            warmup_steps=1,
            device="cpu",
            steps_per_epoch=1,
        )


def test_default_device_prefers_cuda(monkeypatch):
    """The default device is cuda when it is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"


def test_default_device_falls_back_to_cpu(monkeypatch):
    """The default device is cpu without cuda or mps."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == "cpu"


class StubPairwiseModel:
    """Scorer with deterministic per-text logits for tensor tests."""

    def __init__(self, head_count: int) -> None:
        """Store the head count."""
        self.head_count = head_count

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        """Return logits derived from each text's length, one row per text."""
        return torch.tensor(
            [
                [float(len(text) % 7) - 3.0 + head for head in range(self.head_count)]
                for text in texts
            ]
        )


def test_preference_tensor_is_skew_symmetric():
    """The queried tensor averages both orders, with a half diagonal."""
    model = StubPairwiseModel(head_count=3)
    responses = ["alpha", "be", "gamma!"]
    tensor = preference_tensor(model, "p", responses, "cpu", batch_size=2)
    assert tensor.shape == (3, 3, 3)
    assert np.allclose(np.diagonal(tensor, axis1=1, axis2=2), 0.5)
    assert np.allclose(tensor + tensor.transpose(0, 2, 1), 1.0)
    forward = torch.sigmoid(model.score([pairwise_text("p", "alpha", "be")], "cpu"))[0]
    backward = torch.sigmoid(model.score([pairwise_text("p", "be", "alpha")], "cpu"))[0]
    expected = (forward + 1.0 - backward).numpy() / 2.0
    assert np.allclose(tensor[:, 0, 1], expected)


def hub_stub(monkeypatch, tmp_path):
    """Route the tensor checkpoint hub calls to a single local file."""
    store = tmp_path / "tensors.npz"
    monkeypatch.setattr(
        train_prefs,
        "upload_artifact",
        lambda path: store.write_bytes(path.read_bytes()),
    )
    monkeypatch.setattr(train_prefs, "artifact_exists", lambda filename: store.exists())
    monkeypatch.setattr(train_prefs, "artifact_path", lambda filename: store)


def test_upload_and_resume_tensors_round_trip(monkeypatch, tmp_path):
    """A partial checkpoint reloads exactly the scored prefix."""
    hub_stub(monkeypatch, tmp_path)
    prompts = ["p0", "p1", "p2"]
    criteria = ["c0", "c1"]
    assert train_prefs.resume_tensors("tensors.npz", prompts, criteria) == {}
    arrays = {
        "tensor_0": np.full((2, 3, 3), 0.25),
        "tensor_1": np.full((2, 3, 3), 0.75),
    }
    train_prefs.upload_tensors("tensors.npz", prompts, criteria, arrays)
    resumed = train_prefs.resume_tensors("tensors.npz", prompts, criteria)
    assert sorted(resumed) == ["tensor_0", "tensor_1"]
    np.testing.assert_array_equal(resumed["tensor_1"], arrays["tensor_1"])


def test_resume_tensors_rejects_stale_checkpoints(monkeypatch, tmp_path):
    """A checkpoint from another prompt order or head set must not resume."""
    hub_stub(monkeypatch, tmp_path)
    prompts = ["p0", "p1", "p2"]
    criteria = ["c0", "c1"]
    train_prefs.upload_tensors(
        "tensors.npz", prompts, criteria, {"tensor_0": np.full((2, 3, 3), 0.25)}
    )
    with pytest.raises(AssertionError):
        train_prefs.resume_tensors("tensors.npz", ["p1", "p0", "p2"], criteria)
    with pytest.raises(AssertionError):
        train_prefs.resume_tensors("tensors.npz", prompts, ["c0"])
