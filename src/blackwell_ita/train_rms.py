"""Training and persistence of multi-head reward models on preference pairs."""

from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


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
        dataframe: pd.DataFrame,
        criterion_columns: list[str],
        augment_presentation_order: bool,
    ) -> None:
        """Build examples, masking missing targets and optionally swapping order."""
        self.examples: list[Example] = []
        for _, row in dataframe.iterrows():
            targets = torch.tensor(
                [row[column] for column in criterion_columns], dtype=torch.float32
            )
            mask = torch.isfinite(targets).float()
            targets = torch.nan_to_num(targets)
            self.examples.append(
                {  # pyright: ignore[reportArgumentType]
                    "prompt": row["prompt"],
                    "first_response": row["response_a"],
                    "second_response": row["response_b"],
                    "target": targets,
                    "mask": mask,
                }
            )
            if augment_presentation_order:
                self.examples.append(
                    {  # pyright: ignore[reportArgumentType]
                        "prompt": row["prompt"],
                        "first_response": row["response_b"],
                        "second_response": row["response_a"],
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


def masked_binary_cross_entropy(
    logits: torch.Tensor, batch: Batch, device: str
) -> torch.Tensor:
    """Mean binary cross-entropy over the unmasked criterion entries."""
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch["target"].to(device), reduction="none"
    )
    mask = batch["mask"].to(device)
    return (losses * mask).sum() / mask.sum()


class MultiHeadEncoder(torch.nn.Module):
    """Pretrained encoder with a linear head giving one logit per criterion."""

    def __init__(self, encoder_name: str, head_count: int) -> None:
        """Load the encoder and attach a fresh linear head."""
        super().__init__()
        # Parameters stay fp32: AdamW updates at lr=1e-5 underflow pure-bf16
        # weights, leaving them bit-identical after "training"
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        # Without checkpointing, the two live activation graphs of a 16-pair
        # batch at 2000+ tokens need well over the 95 GiB a single GPU offers
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, head_count)
        # Zero-init keeps encoder gradients exactly zero until the head grows,
        # so the untrained head's noise never distorts the pretrained features
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, tokenized: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score tokenized inputs from the last non-padding token's hidden state."""
        hidden_states = self.encoder(**tokenized).last_hidden_state
        last_indices = tokenized["attention_mask"].sum(dim=1) - 1
        pooled = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            last_indices,
        ]
        return self.head(pooled).float()


class RewardModelBase(torch.nn.Module):
    """Shared tokenisation and scoring for the two reward model variants."""

    def __init__(
        self,
        encoder_name: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        head_count: int,
    ) -> None:
        """Wrap a multi-head encoder with its tokenizer and truncation limit."""
        super().__init__()
        self.scorer = MultiHeadEncoder(encoder_name, head_count)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def compute_loss(self, batch: Batch, device: str) -> torch.Tensor:
        """Loss for one batch; implemented by the subclasses."""
        raise NotImplementedError

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        """Tokenize texts and return per-criterion logits."""
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in tokenized.items()}
        # bf16 autocast recovers bf16 memory and speed on cuda while the fp32
        # master weights keep the optimiser updates representable
        if device == "cuda":
            with torch.autocast("cuda", torch.bfloat16):
                return self.scorer(inputs)
        return self.scorer(inputs)


def bt_text(prompt: str, response: str) -> str:
    """Format a prompt-response pair for Bradley-Terry scoring."""
    return f"{prompt}\n\n{response}"


class BradleyTerryModel(RewardModelBase):
    """Reward model scoring each response alone; preferences via reward gaps."""

    def compute_loss(self, batch: Batch, device: str) -> torch.Tensor:
        """Bradley-Terry loss on the reward difference between the responses."""
        first_rewards = self.score(
            [
                bt_text(prompt, response)
                for prompt, response in zip(
                    batch["prompt"], batch["first_response"], strict=True
                )
            ],
            device,
        )
        second_rewards = self.score(
            [
                bt_text(prompt, response)
                for prompt, response in zip(
                    batch["prompt"], batch["second_response"], strict=True
                )
            ],
            device,
        )
        return masked_binary_cross_entropy(
            first_rewards - second_rewards, batch, device
        )


def pairwise_text(prompt: str, first: str, second: str) -> str:
    """Format a prompt with both responses for joint pairwise scoring."""
    return f"{prompt}\n\n[RESPONSE 1]\n{first}\n\n[RESPONSE 2]\n{second}"


class PairwisePreferenceModel(RewardModelBase):
    """Preference model scoring both responses jointly in one input."""

    def compute_loss(self, batch: Batch, device: str) -> torch.Tensor:
        """Cross-entropy between the joint logits and the preference targets."""
        joint_texts = [
            pairwise_text(prompt, first, second)
            for prompt, first, second in zip(
                batch["prompt"],
                batch["first_response"],
                batch["second_response"],
                strict=True,
            )
        ]
        return masked_binary_cross_entropy(
            self.score(joint_texts, device), batch, device
        )


def save_reward_model(
    model: RewardModelBase,
    criterion_columns: list[str],
    encoder_name: str,
    path: str | Path,
) -> None:
    """Save a checkpoint with everything needed to rebuild the model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "class_name": type(model).__name__,
            "encoder_name": encoder_name,
            "criterion_columns": criterion_columns,
            "max_tokens": model.max_tokens,
            "state_dict": model.scorer.state_dict(),
        },
        path,
    )


def load_reward_model(
    path: str | Path, device: str | None = None
) -> tuple["RewardModelBase", list[str]]:
    """Rebuild a saved reward model; returns it with its criterion columns."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_class = {
        "BradleyTerryModel": BradleyTerryModel,
        "PairwisePreferenceModel": PairwisePreferenceModel,
    }[checkpoint["class_name"]]
    model = model_class(
        checkpoint["encoder_name"],
        AutoTokenizer.from_pretrained(checkpoint["encoder_name"]),
        checkpoint["max_tokens"],
        len(checkpoint["criterion_columns"]),
    )
    model.scorer.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, checkpoint["criterion_columns"]


def evaluate_loss(
    model: RewardModelBase,
    data_loader: Iterable[Batch],
    device: str,
) -> float:
    """Average the model's loss over all batches without gradients."""
    model.eval()
    with torch.no_grad():
        losses = [model.compute_loss(batch, device).item() for batch in data_loader]
    return sum(losses) / len(losses)


def train_until_no_improvement(
    model: RewardModelBase,
    train_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
    learning_rate: float,
    warmup_steps: int,
    device: str,
) -> float:
    """Train epoch by epoch, stopping when validation loss stops improving.

    Restores the best-validation weights and returns the best validation loss.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epoch = 0
    while True:
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            loss = model.compute_loss(batch, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        validation_loss = evaluate_loss(model, validation_loader, device)
        epoch += 1
        print(f"epoch {epoch}: validation_loss={validation_loss:.4f}")
        # A NaN validation loss also fails this comparison, so a diverged run
        # stops instead of looping forever on NaN >= best
        if not (validation_loss < best_validation_loss):
            break
        best_validation_loss = validation_loss
        # CPU copies, not an on-device deepcopy: a 3B model cannot afford a
        # second device-resident set of weights
        best_state = {
            key: value.detach().to("cpu", copy=True)
            for key, value in model.state_dict().items()
        }
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_validation_loss


def default_device() -> str:
    """Pick the best available torch device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_loaders(
    dataframe: pd.DataFrame,
    criterion_columns: list[str],
    batch_size: int,
    validation_fraction: float,
    augment_presentation_order: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Split the pairs into train and validation loaders."""
    # Split by prompt, not by row: datasets carry several pairs per prompt, and
    # a row-level split leaks nearly every validation prompt into training,
    # miscalibrating the early stopping that selects the model
    prompts = dataframe["prompt"].drop_duplicates().sample(frac=1.0, random_state=seed)
    validation_prompts = list(prompts.iloc[: int(len(prompts) * validation_fraction)])
    in_validation = dataframe["prompt"].isin(validation_prompts)
    train_dataset = PreferencePairDataset(
        dataframe.loc[~in_validation], criterion_columns, augment_presentation_order
    )
    validation_dataset = PreferencePairDataset(
        dataframe.loc[in_validation], criterion_columns, False
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(validation_dataset, batch_size=batch_size),
    )


def train_reward_model(
    dataframe: pd.DataFrame,
    criterion_columns: list[str],
    model_class: type[RewardModelBase],
    max_tokens: int,
    encoder_name: str = "Qwen/Qwen2.5-3B",
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    warmup_steps: int = 100,
    validation_fraction: float = 0.1,
    augment_presentation_order: bool = True,
    seed: int = 1810,
    device: str | None = None,
) -> tuple[RewardModelBase, float]:
    """Train one reward model; returns it on the CPU with its validation loss.

    Training opens with a head-only half-epoch (epoch 0) over a frozen
    encoder, so full fine-tuning starts from a settled head rather than the
    zero-initialised one. The train/validation split is deterministic in
    ``seed``, so successive calls on the same dataframe train against
    identical splits. The model is moved off the device before returning,
    freeing the GPU for the next call: a 95 GiB GPU cannot train a 3B model
    beside a finished one still holding its weights and gradients.
    """
    if device is None:
        device = default_device()
    train_loader, validation_loader = build_loaders(
        dataframe,
        criterion_columns,
        batch_size,
        validation_fraction,
        augment_presentation_order,
        seed,
    )
    model = model_class(
        encoder_name,
        AutoTokenizer.from_pretrained(encoder_name),
        max_tokens,
        len(criterion_columns),
    )
    model.to(device)
    # The frozen encoder keeps epoch 0 cheap: no gradients flow through it,
    # so its forward pass runs without a backward. A lone linear head can
    # take a far higher learning rate than the encoder fine-tuning
    model.scorer.encoder.requires_grad_(False)
    head_optimizer = torch.optim.AdamW(model.scorer.head.parameters(), lr=1e-3)
    head_steps = max(1, len(train_loader) // 2)
    model.train()
    for batch in tqdm(
        islice(iter(train_loader), head_steps),
        total=head_steps,
        desc="epoch 0",
        leave=False,
    ):
        head_optimizer.zero_grad()
        model.compute_loss(batch, device).backward()
        head_optimizer.step()
    model.scorer.encoder.requires_grad_(True)
    validation_loss = train_until_no_improvement(
        model, train_loader, validation_loader, learning_rate, warmup_steps, device
    )
    model.to("cpu")
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.empty_cache()
    return model, validation_loss
