import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm.auto import tqdm

from blackwell_ita.data import (
    Batch,
    PreferencePairDataset,
    evaluation_loader,
    longest_batch,
    split_datasets,
    training_batches,
)
from blackwell_ita.hub import (
    REWARD_MODELS_REPOSITORY,
    download_hub_file,
    hub_file_exists,
    read_hub_dataframe,
    upload_dataframe,
    upload_hub_file,
)
from blackwell_ita.models import (
    REWARD_MODEL_CLASSES,
    RewardModel,
    load_reward_model,
    save_reward_model,
)

WANDB_PROJECT = "blackwell-ita-reward-models"


@dataclass(frozen=True)
class RewardModelConfiguration:
    """Everything that identifies one trained reward model."""

    name: str
    model_type: str
    encoder_name: str
    lora_rank: int | None
    learning_rate: float
    evaluation_interval_steps: int
    max_tokens: int = 4096
    batch_size: int = 12
    warmup_steps: int = 100
    early_stopping_pair_count: int = 4000
    patience: int = 2
    seed: int = 1810
    gradient_checkpointing: bool = True

    @property
    def checkpoint_filename(self) -> str:
        """Checkpoint filename on the hub."""
        return f"{self.name}.pt"

    @property
    def training_state_filename(self) -> str:
        """Resumable mid-training state filename on the hub."""
        return f"{self.name}_training_state.pt"

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
        evaluation_interval_steps=400,
    ),
    RewardModelConfiguration(
        name="qwen3_4b_bradley_terry",
        model_type="bradley_terry",
        encoder_name="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=None,
        learning_rate=5e-6,
        evaluation_interval_steps=400,
    ),
    RewardModelConfiguration(
        name="gemma4_12b_pairwise_lora",
        model_type="pairwise",
        encoder_name="google/gemma-4-12B-it",
        lora_rank=16,
        learning_rate=1e-4,
        evaluation_interval_steps=400,
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
    train_dataset: PreferencePairDataset,
    validation_loader: Iterable[Batch],
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    steps_per_round: int,
    patience: int,
    seed: int,
    device: str,
    log_metrics: Callable[[dict], None],
    saved_state: dict | None = None,
    save_state: Callable[[dict], None] | None = None,
) -> float:
    """Train in rounds, validating after each, until ``patience`` rounds pass
    without a new best validation loss. Restores the best trainable weights
    and returns the best validation loss.

    ``save_state`` receives everything needed to resume after every round;
    passing that back as ``saved_state`` continues exactly where it stopped.
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
    # Fail in the first minute, not hours in, if the worst batch does not fit
    model.compute_loss(longest_batch(train_dataset, batch_size), device).backward()
    optimizer.zero_grad(set_to_none=True)
    best_validation_loss = float("inf")
    best_trainable_state: dict[str, torch.Tensor] | None = None
    rounds_without_improvement = 0
    step = 0
    if saved_state is not None:
        model.load_state_dict(saved_state["trainable_state"], strict=False)
        optimizer.load_state_dict(saved_state["optimizer"])
        scheduler.load_state_dict(saved_state["scheduler"])
        best_validation_loss = saved_state["best_validation_loss"]
        best_trainable_state = saved_state["best_trainable_state"]
        rounds_without_improvement = saved_state["rounds_without_improvement"]
        step = saved_state["step"]
    batches = training_batches(train_dataset, batch_size, seed, start_step=step)
    while rounds_without_improvement < patience:
        model.train()
        for batch in tqdm(
            islice(batches, steps_per_round),
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
                "early_stopping", validation_loss, validation_criterion_metrics
            )
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            rounds_without_improvement = 0
            best_trainable_state = cpu_copy(trainable_parameters)
        else:
            # A NaN validation loss lands here too, so divergence exhausts patience
            rounds_without_improvement += 1
        if save_state is not None:
            save_state(
                {
                    "step": step,
                    "best_validation_loss": best_validation_loss,
                    "rounds_without_improvement": rounds_without_improvement,
                    "trainable_state": cpu_copy(trainable_parameters),
                    "best_trainable_state": best_trainable_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
            )
    if best_trainable_state is None:
        raise RuntimeError(f"no finite validation loss, last was {validation_loss}")  # pyright: ignore[reportPossiblyUnboundVariable]
    model.load_state_dict(best_trainable_state, strict=False)
    return best_validation_loss


def cpu_copy(parameters: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detached CPU copies, so a snapshot survives further training."""
    return {
        name: parameter.detach().to("cpu", copy=True)
        for name, parameter in parameters.items()
    }


def train_reward_model(
    configuration: RewardModelConfiguration,
    pairs: pd.DataFrame,
    criteria: list[str],
    device: str,
    log_metrics: Callable[[dict], None],
    saved_state: dict | None = None,
    save_state: Callable[[dict], None] | None = None,
) -> tuple[RewardModel, pd.DataFrame]:
    """Train on the train split, early stop on validation, evaluate on test.

    Returns the model on the CPU and its validation and test metrics.
    """
    torch.manual_seed(configuration.seed)
    datasets_by_split = split_datasets(pairs, criteria)
    evaluation_loaders = {
        split_name: evaluation_loader(
            datasets_by_split[split_name], configuration.batch_size
        )
        for split_name in ("validation", "test")
    }
    validation_dataset = datasets_by_split["validation"]
    early_stopping_indices = sorted(
        np.random.default_rng(configuration.seed)
        .choice(
            len(validation_dataset),
            size=min(configuration.early_stopping_pair_count, len(validation_dataset)),
            replace=False,
        )
        .tolist()
    )
    early_stopping_loader = evaluation_loader(
        validation_dataset, configuration.batch_size, early_stopping_indices
    )
    model = REWARD_MODEL_CLASSES[configuration.model_type](
        configuration.encoder_name,
        criteria,
        configuration.max_tokens,
        configuration.lora_rank,
        configuration.gradient_checkpointing,
    )
    train_until_no_improvement(
        model,
        datasets_by_split["train"],
        early_stopping_loader,
        configuration.batch_size,
        configuration.learning_rate,
        configuration.warmup_steps,
        steps_per_round=configuration.evaluation_interval_steps,
        patience=configuration.patience,
        seed=configuration.seed,
        device=device,
        log_metrics=log_metrics,
        saved_state=saved_state,
        save_state=save_state,
    )
    split_metrics = []
    for split_name, split_loader in evaluation_loaders.items():
        pooled_loss, criterion_metrics = evaluate(model, split_loader, device)
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


def ensure_trained(
    configuration: RewardModelConfiguration,
    pairs: pd.DataFrame,
    criteria: list[str],
    prefix: str,
    device: str,
) -> pd.DataFrame:
    """Train and upload the checkpoint and metrics unless the hub has them.

    LoRA runs upload their training state after every round and resume from
    it. A full fine-tune's state would be about 48 GB, so it trains in one go.
    Returns the validation and test metrics.
    """
    if not hub_file_exists(
        REWARD_MODELS_REPOSITORY, configuration.checkpoint_filename, prefix
    ):
        resumable = configuration.lora_rank is not None
        saved_state = (
            torch.load(
                download_hub_file(
                    REWARD_MODELS_REPOSITORY,
                    configuration.training_state_filename,
                    prefix,
                ),
                map_location="cpu",
                weights_only=True,
            )
            if resumable
            and hub_file_exists(
                REWARD_MODELS_REPOSITORY, configuration.training_state_filename, prefix
            )
            else None
        )
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{prefix}_{configuration.name}",
            config=asdict(configuration) | {"hub_prefix": prefix},
            id=saved_state["wandb_run_id"] if saved_state is not None else None,
            resume="allow",
        )
        wandb_run.define_metric("*", step_metric="step")

        def upload_training_state(training_state: dict) -> None:
            with tempfile.TemporaryDirectory() as temporary_directory:
                local_state_path = (
                    Path(temporary_directory) / configuration.training_state_filename
                )
                torch.save(
                    training_state | {"wandb_run_id": wandb_run.id}, local_state_path
                )
                upload_hub_file(REWARD_MODELS_REPOSITORY, local_state_path, prefix)

        trained_model, trained_metrics = train_reward_model(
            configuration,
            pairs,
            criteria,
            device,
            wandb_run.log,
            saved_state=saved_state,
            save_state=upload_training_state if resumable else None,
        )
        wandb_run.finish()
        upload_dataframe(
            REWARD_MODELS_REPOSITORY,
            configuration.metrics_filename,
            trained_metrics,
            prefix,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            local_checkpoint_path = (
                Path(temporary_directory) / configuration.checkpoint_filename
            )
            save_reward_model(trained_model, local_checkpoint_path)
            upload_hub_file(REWARD_MODELS_REPOSITORY, local_checkpoint_path, prefix)
    return read_hub_dataframe(
        REWARD_MODELS_REPOSITORY, configuration.metrics_filename, prefix
    )


def load_trained(
    configuration: RewardModelConfiguration, prefix: str, device: str
) -> RewardModel:
    """Load a trained checkpoint from the hub in eval mode."""
    return load_reward_model(
        download_hub_file(
            REWARD_MODELS_REPOSITORY, configuration.checkpoint_filename, prefix
        ),
        device,
    )
