from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from itertools import islice

import pandas as pd
import torch
from tqdm.auto import tqdm

from blackwell_ita.data import CRITERIA, Batch, split_loaders
from blackwell_ita.models import REWARD_MODEL_CLASSES, RewardModel


@dataclass(frozen=True)
class RewardModelConfiguration:
    """Everything that identifies one trained reward model."""

    name: str
    model_type: str
    encoder_name: str
    lora_rank: int | None
    learning_rate: float
    max_tokens: int = 4096
    batch_size: int = 12
    warmup_steps: int = 100
    evaluations_per_epoch: int = 3
    patience: int = 2
    seed: int = 1810

    @property
    def checkpoint_filename(self) -> str:
        """Checkpoint filename on the hub."""
        return f"{self.name}.pt"

    @property
    def metrics_filename(self) -> str:
        """Validation and test metrics filename on the hub."""
        return f"{self.name}_metrics.parquet"


REWARD_MODEL_CONFIGURATIONS = [
    RewardModelConfiguration(
        name="qwen3_4b_pairwise",
        model_type="pairwise",
        encoder_name="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=None,
        learning_rate=5e-6,
    ),
    RewardModelConfiguration(
        name="qwen3_4b_bradley_terry",
        model_type="bradley_terry",
        encoder_name="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=None,
        learning_rate=5e-6,
    ),
    RewardModelConfiguration(
        name="gemma4_12b_pairwise_lora",
        model_type="pairwise",
        encoder_name="google/gemma-4-12B-it",
        lora_rank=16,
        learning_rate=1e-4,
    ),
]


def evaluate(
    model: RewardModel,
    data_loader: Iterable[Batch],
    device: str,
) -> tuple[float, pd.DataFrame]:
    """Pooled masked loss and per-criterion loss and decisive-pair accuracy.

    A pair is decisive for a criterion when its labelled target is at most
    0.25 or at least 0.75. Accuracy thresholds the logit at zero.
    """
    model.eval()
    logit_batches, target_batches, mask_batches = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            logit_batches.append(model.batch_logits(batch, device).cpu())
            target_batches.append(batch["target"])
            mask_batches.append(batch["mask"])
    logits = torch.cat(logit_batches)
    targets = torch.cat(target_batches)
    mask = torch.cat(mask_batches).bool()
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    decisive = mask & ((targets <= 0.25) | (targets >= 0.75))
    correct = (logits > 0) == (targets > 0.5)
    criterion_metrics = pd.DataFrame(
        [
            {
                "criterion": criterion,
                "loss": losses[:, index][mask[:, index]].mean().item(),
                "decisive_accuracy": correct[:, index][decisive[:, index]]
                .float()
                .mean()
                .item(),
                "decisive_count": int(decisive[:, index].sum().item()),
            }
            for index, criterion in enumerate(model.criteria)
        ]
    )
    return losses[mask].mean().item(), criterion_metrics


def logged_metrics(
    split_name: str, pooled_loss: float, criterion_metrics: pd.DataFrame
) -> dict[str, float]:
    """Flatten evaluation results into wandb keys under ``split_name/``."""
    return {f"{split_name}/loss": pooled_loss} | {
        f"{split_name}/{row.criterion}_{metric_name}": getattr(row, metric_name)  # pyright: ignore[reportAttributeAccessIssue]
        for row in criterion_metrics.itertuples()
        for metric_name in ("loss", "decisive_accuracy")
    }


def train_until_no_improvement(
    model: RewardModel,
    train_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
    learning_rate: float,
    warmup_steps: int,
    steps_per_round: int,
    patience: int,
    device: str,
    log_metrics: Callable[[dict], None],
) -> float:
    """Train in rounds, validating after each, until ``patience`` rounds pass
    without a new best validation loss. Restores the best trainable weights
    and returns the best validation loss.
    """
    model.to(device)
    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(trainable_parameters.values(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda scheduler_step: min(1.0, (scheduler_step + 1) / warmup_steps)
    )

    def cycling_batches() -> Iterator[Batch]:
        while True:
            yield from train_loader

    training_batches = cycling_batches()
    best_validation_loss = float("inf")
    best_trainable_state: dict[str, torch.Tensor] | None = None
    rounds_without_improvement = 0
    step = 0
    while rounds_without_improvement < patience:
        model.train()
        for batch in tqdm(
            islice(training_batches, steps_per_round),
            total=steps_per_round,
            desc=f"steps {step + 1} to {step + steps_per_round}",
        ):
            optimizer.zero_grad()
            loss = model.compute_loss(batch, device)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters.values(), float("inf")
            )
            optimizer.step()
            scheduler.step()
            step += 1
            log_metrics(
                {
                    "step": step,
                    "train/loss": loss.item(),
                    "train/gradient_norm": gradient_norm.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                }
            )
        # Free the gradients before validation: on a 4B fp32 model they are 15 GiB
        optimizer.zero_grad(set_to_none=True)
        validation_loss, validation_criterion_metrics = evaluate(
            model, validation_loader, device
        )
        log_metrics(
            {"step": step}
            | logged_metrics(
                "validation", validation_loss, validation_criterion_metrics
            )
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            rounds_without_improvement = 0
            best_trainable_state = {
                name: parameter.detach().to("cpu", copy=True)
                for name, parameter in trainable_parameters.items()
            }
        else:
            # A NaN validation loss lands here too, so divergence exhausts patience
            rounds_without_improvement += 1
    if best_trainable_state is None:
        raise RuntimeError(f"no finite validation loss, last was {validation_loss}")  # pyright: ignore[reportPossiblyUnboundVariable]
    model.load_state_dict(best_trainable_state, strict=False)
    return best_validation_loss


def train_reward_model(
    configuration: RewardModelConfiguration,
    pairs: pd.DataFrame,
    device: str,
    log_metrics: Callable[[dict], None],
) -> tuple[RewardModel, pd.DataFrame]:
    """Train on the train split, early stop on validation, evaluate on test.

    Returns the model on the CPU and its validation and test metrics.
    """
    torch.manual_seed(configuration.seed)
    loaders = split_loaders(pairs, CRITERIA, configuration.batch_size)
    model = REWARD_MODEL_CLASSES[configuration.model_type](
        configuration.encoder_name,
        CRITERIA,
        configuration.max_tokens,
        configuration.lora_rank,
    )
    train_until_no_improvement(
        model,
        loaders["train"],
        loaders["validation"],
        configuration.learning_rate,
        configuration.warmup_steps,
        steps_per_round=max(
            1, len(loaders["train"]) // configuration.evaluations_per_epoch
        ),
        patience=configuration.patience,
        device=device,
        log_metrics=log_metrics,
    )
    split_metrics = []
    for split_name in ("validation", "test"):
        pooled_loss, criterion_metrics = evaluate(model, loaders[split_name], device)
        log_metrics(
            logged_metrics(f"final_{split_name}", pooled_loss, criterion_metrics)
        )
        split_metrics.append(
            criterion_metrics.assign(
                split=split_name, pooled_loss=pooled_loss, **asdict(configuration)
            )
        )
    model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return model, pd.concat(split_metrics, ignore_index=True)
