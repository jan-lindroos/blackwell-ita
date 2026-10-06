import numpy as np
import torch
from test_models import CharacterTokenizer

from blackwell_ita import scoring
from blackwell_ita.scoring import (
    pairwise_preference_tensor,
    preferences_to_rewards,
    rewards_to_preferences,
    score_with_resume,
)


class FirstResponseLengthModel:
    """Logit favours whichever response comes first, more so when it is longer."""

    model_type = "pairwise"
    tokenizer = CharacterTokenizer()
    max_tokens = 1000

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        first_lengths = [
            len(text.split("[RESPONSE 1]\n")[1].split("\n\n[RESPONSE 2]")[0])
            for text in texts
        ]
        return torch.tensor([[1.0 + 0.1 * length] for length in first_lengths])


def test_pairwise_preference_tensor_cancels_presentation_order_bias():
    tensor = pairwise_preference_tensor(
        FirstResponseLengthModel(),  # pyright: ignore[reportArgumentType]
        "p",
        ["a", "aaaa", "aa"],
        "cpu",
        batch_size=2,
    )
    assert tensor.shape == (1, 3, 3)
    assert np.allclose(tensor + tensor.transpose(0, 2, 1), 1.0)
    assert np.allclose(np.diagonal(tensor[0]), 0.5)
    # A constant first-position bonus cancels, leaving the longer response ahead
    assert tensor[0, 1, 0] > 0.5
    assert tensor[0, 1, 2] > 0.5


class ResponseLengthRewardModel:
    """Pointwise reward equal to the response length."""

    model_type = "bradley_terry"

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        return torch.tensor(
            [[float(len(text.split("[RESPONSE]\n")[1]))] for text in texts]
        )


def test_bradley_terry_preference_tensor_is_sigmoid_of_reward_differences():
    tensor = pairwise_preference_tensor(
        ResponseLengthRewardModel(),  # pyright: ignore[reportArgumentType]
        "p",
        ["a", "aaaa", "aa"],
        "cpu",
        batch_size=2,
    )
    rewards = np.array([1.0, 4.0, 2.0])
    expected = 1.0 / (1.0 + np.exp(rewards[None, :] - rewards[:, None]))
    assert tensor.shape == (1, 3, 3)
    assert np.allclose(tensor[0], expected)


def test_preferences_to_rewards_recovers_rewards_up_to_a_per_head_constant():
    rewards = np.array([[1.0, 4.0, 2.0, -3.0], [0.5, 0.0, 6.0, 1.0]])
    recovered = preferences_to_rewards(rewards_to_preferences(rewards))
    assert recovered.shape == rewards.shape
    assert np.allclose(
        recovered - recovered.mean(axis=1, keepdims=True),
        rewards - rewards.mean(axis=1, keepdims=True),
    )


def test_score_with_resume_scores_only_missing_keys_and_checkpoints(monkeypatch):
    hub_storage = {"tensors.npz": {"done": np.zeros(1)}}
    monkeypatch.setattr(
        scoring,
        "download_tensors",
        lambda repository_id, filename, prefix: dict(hub_storage.get(filename, {})),
    )
    monkeypatch.setattr(
        scoring,
        "upload_tensors",
        lambda repository_id, filename, tensors, prefix: hub_storage.update(
            {filename: dict(tensors)}
        ),
    )
    scored_inputs = []

    def score_one(value: float) -> np.ndarray:
        scored_inputs.append(value)
        return np.full(1, value)

    tensors = score_with_resume(
        "repository",
        "tensors.npz",
        "prefix",
        {"done": 9.0, "first": 1.0, "second": 2.0, "third": 3.0},
        score_one,
        checkpoint_every=2,
    )
    assert scored_inputs == [1.0, 2.0, 3.0]
    assert list(tensors) == ["done", "first", "second", "third"]
    assert tensors["done"].tolist() == [0.0]
    assert set(hub_storage["tensors.npz"]) == {"done", "first", "second", "third"}
