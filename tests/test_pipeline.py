"""End-to-end tests over training, winners, and persistence."""

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from train_rms import (
    Batch,
    BradleyTerryModel,
    PairwisePreferenceModel,
    PreferencePairDataset,
    load_reward_model,
    save_reward_model,
)
from winners import best_of_nash, blackwell_winner, preference_tensor, reward_scores

TINY_ENCODER = "yujiepan/qwen2.5-tiny-random"
CRITERION_COLUMNS = ["first_criterion", "second_criterion", "overall"]


def tiny_pairs() -> pd.DataFrame:
    """Two preference pairs with a missing criterion target."""
    return pd.DataFrame(
        {
            "prompt": ["p1", "p2"],
            "response_a": ["good answer", "short"],
            "response_b": ["bad answer", "much longer answer"],
            "first_criterion": [1.0, 0.5],
            "second_criterion": [0.0, None],
            "overall": [1.0, 0.0],
        }
    )


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    """A tiny encoder runs through training losses, winners, and reload."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_ENCODER)
    dataset = PreferencePairDataset(tiny_pairs(), CRITERION_COLUMNS, True)
    assert len(dataset) == 4
    assert dataset[2]["mask"].tolist() == [1.0, 0.0, 1.0]

    pairwise_model = PairwisePreferenceModel(TINY_ENCODER, tokenizer, 128, 3)
    bradley_terry_model = BradleyTerryModel(TINY_ENCODER, tokenizer, 128, 3)
    batch: Batch = {
        "prompt": ["p1"],
        "first_response": ["good answer"],
        "second_response": ["bad answer"],
        "target": torch.tensor([[1.0, 0.0, 1.0]]),
        "mask": torch.tensor([[1.0, 1.0, 1.0]]),
    }
    for model in (pairwise_model, bradley_terry_model):
        assert torch.isfinite(model.compute_loss(batch, "cpu"))

    responses = ["alpha", "beta", "gamma"]
    tensor = preference_tensor(pairwise_model, "p1", responses, "cpu")
    assert tensor.shape == (3, 3, 3)
    nash_policy = best_of_nash(tensor[2])
    blackwell_policy = blackwell_winner(tensor[:2])
    for policy in (nash_policy, blackwell_policy):
        assert policy.min() > -1e-6 and abs(policy.sum() - 1.0) < 1e-6
    rewards = reward_scores(bradley_terry_model, "p1", responses, "cpu")
    assert rewards.shape == (3, 3)

    checkpoint_path = tmp_path / "pairwise.pt"
    save_reward_model(pairwise_model, CRITERION_COLUMNS, TINY_ENCODER, checkpoint_path)
    loaded_model, loaded_columns = load_reward_model(checkpoint_path)
    assert loaded_columns == CRITERION_COLUMNS
    reloaded = preference_tensor(loaded_model, "p1", responses, "cpu")
    np.testing.assert_allclose(reloaded, tensor, atol=1e-5)
