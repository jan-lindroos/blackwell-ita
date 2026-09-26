import pandas as pd
import pytest
import torch

from blackwell_ita import training
from blackwell_ita.training import (
    REWARD_MODEL_CONFIGURATIONS,
    evaluate,
    logged_metrics,
    train_until_no_improvement,
)


class ScriptedModel(torch.nn.Module):
    """Validation losses follow a script; the weight counts training steps."""

    criteria = ("x", "y")

    def __init__(self, validation_losses: list[float]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.frozen_weight = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.validation_losses = iter(validation_losses)

    def compute_loss(self, batch, device):
        # The gradient is -1, so a unit-lr SGD-like step adds about one
        return -self.weight

    def batch_logits(self, batch, device):
        return batch["target"] * 0.0 + next(self.validation_losses)


def validation_batch() -> dict:
    return {"target": torch.tensor([[1.0, 0.0]]), "mask": torch.tensor([[1.0, 1.0]])}


def test_train_until_no_improvement_restores_the_best_trainable_weights():
    # Round 3 regresses and round 4 recovers; rounds 5 and 6 exhaust patience
    model = ScriptedModel([3.0, 2.0, 2.5, 1.5, 1.6, 1.7])
    weights_after_each_round = []
    logged_rows = []

    def log_metrics(row: dict) -> None:
        logged_rows.append(row)
        if "validation/loss" in row:
            weights_after_each_round.append(model.weight.item())

    train_until_no_improvement(
        model,  # pyright: ignore[reportArgumentType]
        train_loader=[{}],  # pyright: ignore[reportArgumentType]
        validation_loader=[validation_batch()],  # pyright: ignore[reportArgumentType]
        learning_rate=0.1,
        warmup_steps=1,
        steps_per_round=2,
        patience=2,
        device="cpu",
        log_metrics=log_metrics,
    )
    assert len(weights_after_each_round) == 6
    assert model.weight.item() == weights_after_each_round[3]
    assert [row["step"] for row in logged_rows if "train/loss" in row] == list(
        range(1, 13)
    )


def test_train_until_no_improvement_raises_when_validation_is_never_finite():
    model = ScriptedModel([float("nan")] * 2)
    with pytest.raises(RuntimeError):
        train_until_no_improvement(
            model,  # pyright: ignore[reportArgumentType]
            train_loader=[{}],  # pyright: ignore[reportArgumentType]
            validation_loader=[validation_batch()],  # pyright: ignore[reportArgumentType]
            learning_rate=0.1,
            warmup_steps=1,
            steps_per_round=1,
            patience=2,
            device="cpu",
            log_metrics=lambda row: None,
        )


class FixedLogitModel(torch.nn.Module):
    criteria = ("x", "y")

    def batch_logits(self, batch, device):
        return torch.tensor([[2.0, 1.0], [-3.0, 1.0], [1.0, 1.0]])


def test_evaluate_masks_and_counts_only_decisive_pairs():
    batch = {
        "target": torch.tensor([[1.0, 0.25], [0.0, 0.5], [0.75, 0.0]]),
        "mask": torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
    }
    pooled_loss, criterion_metrics = evaluate(
        FixedLogitModel(),  # pyright: ignore[reportArgumentType]
        [batch],  # pyright: ignore[reportArgumentType]
        "cpu",
    )
    metrics_by_criterion = criterion_metrics.set_index("criterion")
    assert metrics_by_criterion.loc["x", "decisive_count"] == 3
    assert metrics_by_criterion.loc["x", "decisive_accuracy"] == 1.0
    # y: pair 2 is tied and pair 3 is masked, leaving pair 1, predicted wrong
    assert metrics_by_criterion.loc["y", "decisive_count"] == 1
    assert metrics_by_criterion.loc["y", "decisive_accuracy"] == 0.0
    assert pooled_loss == pytest.approx(
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0, 1.0, -3.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.25, 0.0, 0.5, 0.75]),
        ).item()
    )


def test_logged_metrics_flattens_per_criterion_results():
    criterion_metrics = pd.DataFrame(
        {"criterion": ["x"], "loss": [0.5], "decisive_accuracy": [0.8]}
    )
    assert logged_metrics("test", 0.4, criterion_metrics) == {
        "test/loss": 0.4,
        "test/x_loss": 0.5,
        "test/x_decisive_accuracy": 0.8,
    }


def test_configurations_have_unique_names_and_known_model_types():
    names = [configuration.name for configuration in REWARD_MODEL_CONFIGURATIONS]
    assert len(names) == len(set(names))
    assert {
        configuration.model_type for configuration in REWARD_MODEL_CONFIGURATIONS
    } <= set(training.REWARD_MODEL_CLASSES)
