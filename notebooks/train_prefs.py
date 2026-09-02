# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
#     "datasets",
#     "huggingface_hub",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "torch",
#     # molab's base image ships a torchvision built against a mismatched
#     # torch. Install a matching one so transformers doesn't import the
#     # broken system copy
#     "torchvision",
#     "transformers",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import math
    import shutil
    import tempfile
    from collections.abc import Callable, Iterable, Iterator
    from itertools import islice
    from pathlib import Path
    from typing import TypedDict

    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import HfApi, file_exists, hf_hub_download
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

    RMS_REPO = "blackwell-ita/reward-models"
    ARTIFACTS_REPO = "blackwell-ita/artifacts"
    SPLITS_REPO = "blackwell-ita/helpsteer2-splits"
    DATASET = "helpsteer2"
    DEFAULT_BASE_MODEL = "RLHFlow/LLaMA3-SFT-v2"

    HELPSTEER2_ATTRIBUTES = [
        "helpfulness",
        "correctness",
        "coherence",
        "complexity",
        "verbosity",
    ]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before running, switch to GPU, then sign in with `hf auth login`
    (open terminal available via command menu).
    """)


@app.function
def graded_target(margin: float) -> float:
    """Map an ordinal rating margin to a graded win probability."""
    if margin >= 2:
        return 1.0
    if margin >= 1:
        return 0.75
    if margin <= -2:
        return 0.0
    if margin <= -1:
        return 0.25
    return 0.5


@app.function
def helpsteer2_response_pairs(
    responses: pd.DataFrame,
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """Yield response row pairs, paired consecutively within each prompt group."""
    for prompt, group in responses.groupby("prompt", sort=False):
        assert len(group) % 2 == 0, f"odd response group for prompt {prompt!r}"
        for start in range(0, len(group), 2):
            yield group.iloc[start], group.iloc[start + 1]


@app.function
def helpsteer2_pairs(
    responses: pd.DataFrame, preferences: pd.DataFrame
) -> pd.DataFrame:
    """Build preference pairs with per-attribute and overall targets."""
    rows = []
    for first, second in helpsteer2_response_pairs(responses):
        row = {
            "prompt": first["prompt"],
            "response_a": first["response"],
            "response_b": second["response"],
        }
        for attribute in HELPSTEER2_ATTRIBUTES:
            row[attribute] = graded_target(first[attribute] - second[attribute])
        rows.append(row)
    pairs = pd.DataFrame(rows)
    # Positive preference_strength means response_2 is preferred. The
    # (prompt, response_1, response_2) key is unique within the train split
    # (verified against the source file, Aug 2026), so the merge cannot fan out
    overall = preferences.assign(
        overall=[
            graded_target(-strength) for strength in preferences["preference_strength"]
        ]
    )
    return pairs.merge(
        overall[["prompt", "response_1", "response_2", "overall"]],
        how="left",
        left_on=["prompt", "response_a", "response_b"],
        right_on=["prompt", "response_1", "response_2"],
    ).drop(columns=["response_1", "response_2"])


@app.function
def ita_holdout_prompts(
    pairs: pd.DataFrame, count: int = 100, seed: int = 1810
) -> list[str]:
    """Pick the prompts held out for the downstream ITA experiment.

    Drawn only from pairs carrying a decisive overall preference, because
    ``generate_candidates.select_anchors`` skips pairs whose overall target is
    missing or tied: sampling from every pair would silently yield fewer than
    ``count`` anchors.
    """
    decisive = pairs[pairs["overall"].notna() & (pairs["overall"] != 0.5)]
    prompts = decisive["prompt"].drop_duplicates()
    if len(prompts) < count:
        raise ValueError(
            f"only {len(prompts)} decisive prompts available for a held-out"
            f" set of {count}"
        )
    return sorted(prompts.sample(n=count, random_state=seed))


@app.function
def build_explicit_loaders(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    criterion_columns: list[str],
    batch_size: int,
    augment_presentation_order: bool,
) -> tuple[DataLoader, DataLoader]:
    """Loaders over two frames that are already disjoint by construction.

    The counterpart of ``build_loaders`` for when validation comes from its own
    source split rather than a carve-out of the training data. Only the
    training loader is shuffled and augmented, exactly as in ``build_loaders``.
    """
    if train_frame.empty or validation_frame.empty:
        raise ValueError(
            f"empty frame: {len(train_frame)} training and"
            f" {len(validation_frame)} validation pairs"
        )
    # Cheap here and catastrophic if missed: a shared prompt would leak
    # training data into the loss that selects the checkpoint
    shared = set(train_frame["prompt"]) & set(validation_frame["prompt"])
    if shared:
        raise ValueError(f"{len(shared)} prompts appear in both frames")
    return (
        DataLoader(
            PreferencePairDataset(
                train_frame, criterion_columns, augment_presentation_order
            ),
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            PreferencePairDataset(validation_frame, criterion_columns, False),
            batch_size=batch_size,
        ),
    )


@app.function
def model_path(filename: str) -> Path:
    """Download a reward-model artifact, returning its local cache path."""
    return Path(hf_hub_download(RMS_REPO, f"{DATASET}/{filename}"))


@app.function
def pairs_path() -> Path:
    """Download the canonical split pairs, returning the local cache path."""
    return Path(hf_hub_download(SPLITS_REPO, "pairs.parquet", repo_type="dataset"))


@app.function
def upload_pairs(dataframe: pd.DataFrame) -> None:
    """Upload the canonical split pairs to the splits repo."""
    api = HfApi()
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / "pairs.parquet"
        dataframe.to_parquet(path)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo="pairs.parquet",
            repo_id=SPLITS_REPO,
            repo_type="dataset",
        )


@app.function
def pool_prefix(model_name: str) -> str:
    """Artifact path prefix for a backbone's candidate pools.

    The default backbone keeps the original flat helpsteer2/ layout the hub
    artifacts already use; other backbones get their own subfolder so runs
    cannot clobber each other.
    """
    if model_name == DEFAULT_BASE_MODEL:
        return "helpsteer2"
    return f"helpsteer2/{model_name.split('/')[-1].lower()}"


@app.function
def artifact_path(filename: str, prefix: str = DATASET) -> Path:
    """Download a results artifact, returning its local cache path."""
    return Path(
        hf_hub_download(ARTIFACTS_REPO, f"{prefix}/{filename}", repo_type="dataset")
    )


@app.function
def artifact_exists(filename: str, prefix: str = DATASET) -> bool:
    """Check whether a results artifact exists on the hub."""
    return file_exists(ARTIFACTS_REPO, f"{prefix}/{filename}", repo_type="dataset")


@app.function
def upload_model(local_path: Path) -> None:
    """Upload a reward-model artifact to the hub."""
    api = HfApi()
    api.create_repo(RMS_REPO, exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{DATASET}/{local_path.name}",
        repo_id=RMS_REPO,
    )


@app.function
def upload_artifact(local_path: Path, prefix: str = DATASET) -> None:
    """Upload a results artifact to the hub."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{prefix}/{local_path.name}",
        repo_id=ARTIFACTS_REPO,
        repo_type="dataset",
    )


@app.class_definition
class Example(TypedDict):
    """A preference pair with per-criterion targets and a validity mask."""

    prompt: str
    first_response: str
    second_response: str
    target: torch.Tensor
    mask: torch.Tensor


@app.class_definition
class Batch(TypedDict):
    """A collated batch of preference pairs."""

    prompt: list[str]
    first_response: list[str]
    second_response: list[str]
    target: torch.Tensor
    mask: torch.Tensor


@app.class_definition
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


@app.function
def masked_binary_cross_entropy(
    logits: torch.Tensor, batch: Batch, device: str
) -> torch.Tensor:
    """Mean binary cross-entropy over the unmasked criterion entries."""
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch["target"].to(device), reduction="none"
    )
    mask = batch["mask"].to(device)
    return (losses * mask).sum() / mask.sum()


@app.class_definition
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

    def forward(self, tokenized: dict[str, torch.Tensor]) -> torch.Tensor:
        """Score tokenized inputs from the last non-padding token's hidden state."""
        hidden_states = self.encoder(**tokenized).last_hidden_state
        last_indices = tokenized["attention_mask"].sum(dim=1) - 1
        pooled = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            last_indices,
        ]
        return self.head(pooled).float()


@app.function
def pairwise_text(prompt: str, first: str, second: str) -> str:
    """Format a prompt with both responses for joint pairwise scoring."""
    return f"{prompt}\n\n[RESPONSE 1]\n{first}\n\n[RESPONSE 2]\n{second}"


@app.function
def truncated_pairwise_text(
    prompt: str,
    first: str,
    second: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> str:
    """Pairwise text whose responses are truncated to fit within ``max_tokens``.

    The tokenizer's own right truncation would consume [RESPONSE 2]'s tail
    first, biasing supervision towards [RESPONSE 1]. Instead the token budget
    left after the full prompt and markers is split equally between the
    responses, a response shorter than its half donating the surplus to the
    other.
    """
    first_ids = tokenizer.encode(first)
    second_ids = tokenizer.encode(second)
    budget = max_tokens - len(tokenizer.encode(pairwise_text(prompt, "", "")))
    if len(first_ids) + len(second_ids) <= budget:
        return pairwise_text(prompt, first, second)
    first_keep = max(
        0, min(len(first_ids), max(budget // 2, budget - len(second_ids)))
    )
    second_keep = max(0, budget - first_keep)
    return pairwise_text(
        prompt,
        tokenizer.decode(first_ids[:first_keep]),  # pyright: ignore[reportArgumentType]
        tokenizer.decode(second_ids[:second_keep]),  # pyright: ignore[reportArgumentType]
    )


@app.class_definition
class PairwisePreferenceModel(torch.nn.Module):
    """Preference model scoring both responses jointly in one input."""

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
        if device.startswith("cuda"):
            with torch.autocast("cuda", torch.bfloat16):
                return self.scorer(inputs)
        return self.scorer(inputs)

    def batch_logits(self, batch: Batch, device: str) -> torch.Tensor:
        """Joint logits over both responses, truncated symmetrically if overlong."""
        # score()'s truncation=True still backstops the token or two of drift
        # a decode and re-encode round trip can introduce at merge boundaries
        joint_texts = [
            truncated_pairwise_text(
                prompt, first, second, self.tokenizer, self.max_tokens
            )
            for prompt, first, second in zip(
                batch["prompt"],
                batch["first_response"],
                batch["second_response"],
                strict=True,
            )
        ]
        return self.score(joint_texts, device)

    def forward(
        self, batch: Batch, device: str, *, as_loss: bool = False
    ) -> torch.Tensor:
        """The batch's logits, or its masked loss when ``as_loss``.

        The single entry point a wrapper can hook. FSDP installs its
        all-gather hooks on forward and nowhere else, so a caller that reaches
        the model through a custom method instead gets the unwrapped module,
        its parameters still one-dimensional shards, and an embedding lookup
        that fails on a 1-D weight. Both the training loss and the metrics
        need a forward, hence the flag rather than two methods: only one of
        them would be hooked.
        """
        logits = self.batch_logits(batch, device)
        if not as_loss:
            return logits
        return masked_binary_cross_entropy(logits, batch, device)

    def compute_loss(self, batch: Batch, device: str) -> torch.Tensor:
        """Masked binary cross-entropy between the batch logits and targets."""
        return masked_binary_cross_entropy(
            self.batch_logits(batch, device), batch, device
        )


@app.function
def save_reward_model(
    model: PairwisePreferenceModel,
    criterion_columns: list[str],
    encoder_name: str,
    path: str | Path,
    scorer_state: dict | None = None,
) -> None:
    """Save a checkpoint with everything needed to rebuild the model.

    ``scorer_state`` overrides the encoder's own state dict, for the gathered
    copy a model sharded across processes has to be saved from: its local
    state dict holds only this rank's slice, which nothing can reload alone.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_name": encoder_name,
            "criterion_columns": criterion_columns,
            "max_tokens": model.max_tokens,
            "state_dict": (
                model.scorer.state_dict() if scorer_state is None else scorer_state
            ),
        },
        path,
    )


@app.function
def load_reward_model(
    path: str | Path, device: str | None = None
) -> tuple[PairwisePreferenceModel, list[str]]:
    """Rebuild a saved pairwise model; returns it with its criterion columns."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = PairwisePreferenceModel(
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


@app.function
def pointwise_text(prompt: str, response: str) -> str:
    """Format a prompt with a single response for pointwise reward scoring."""
    return f"{prompt}\n\n[RESPONSE]\n{response}"


@app.class_definition
class BradleyTerryRewardModel(torch.nn.Module):
    """Pointwise reward model trained with a graded Bradley-Terry objective.

    Reuses the pairwise model's ``MultiHeadEncoder`` backbone unchanged; only
    the input format (one response per input) and the logit construction
    (difference of two pointwise scores) differ. Implements ``batch_logits``
    and ``compute_loss`` with the ``PairwisePreferenceModel`` signatures so
    the training loop and metrics work on it unmodified.
    """

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

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        """Tokenize texts and return per-criterion pointwise rewards.

        Right truncation cuts an overlong response's tail; both responses of a
        pair are scored in separate inputs under the same rule, so unlike the
        joint pairwise format no presentation-order bias can arise.
        """
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in tokenized.items()}
        if device.startswith("cuda"):
            with torch.autocast("cuda", torch.bfloat16):
                return self.scorer(inputs)
        return self.scorer(inputs)

    def batch_logits(self, batch: Batch, device: str) -> torch.Tensor:
        """Bradley-Terry pair logits: pointwise reward difference per criterion.

        Both sides go through one forward rather than two. Autocast's bfloat16
        copies of the fp32 weights are held by the autograd graph until
        backward, so two graphs mean two copies -- 15 GiB on a 4B model,
        independent of batch size, which is what made a separate-forwards
        version run out of memory even at a micro-batch of one. The cost is
        that the batch now pads to the longest response across both sides
        instead of each side's own longest.
        """
        count = len(batch["prompt"])
        scores = self.score(
            [
                pointwise_text(prompt, response)
                for prompt, response in zip(
                    batch["prompt"], batch["first_response"], strict=True
                )
            ]
            + [
                pointwise_text(prompt, response)
                for prompt, response in zip(
                    batch["prompt"], batch["second_response"], strict=True
                )
            ],
            device,
        )
        return scores[:count] - scores[count:]

    def forward(
        self, batch: Batch, device: str, *, as_loss: bool = False
    ) -> torch.Tensor:
        """The batch's logits, or its masked loss when ``as_loss``.

        The single entry point a wrapper can hook. FSDP installs its
        all-gather hooks on forward and nowhere else, so a caller that reaches
        the model through a custom method instead gets the unwrapped module,
        its parameters still one-dimensional shards, and an embedding lookup
        that fails on a 1-D weight. Both the training loss and the metrics
        need a forward, hence the flag rather than two methods: only one of
        them would be hooked.
        """
        logits = self.batch_logits(batch, device)
        if not as_loss:
            return logits
        return masked_binary_cross_entropy(logits, batch, device)

    def compute_loss(self, batch: Batch, device: str) -> torch.Tensor:
        """Masked binary cross-entropy between the pair logits and targets."""
        return masked_binary_cross_entropy(
            self.batch_logits(batch, device), batch, device
        )


@app.function
def save_bt_reward_model(
    model: BradleyTerryRewardModel,
    criterion_columns: list[str],
    encoder_name: str,
    path: str | Path,
    scorer_state: dict | None = None,
) -> None:
    """Save a checkpoint with everything needed to rebuild the model.

    ``scorer_state`` overrides the encoder's own state dict, for the gathered
    copy a model sharded across processes has to be saved from: its local
    state dict holds only this rank's slice, which nothing can reload alone.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # The type tag keeps a BT checkpoint from being silently rebuilt
            # as a pairwise model by load_reward_model
            "model_type": "bradley_terry",
            "encoder_name": encoder_name,
            "criterion_columns": criterion_columns,
            "max_tokens": model.max_tokens,
            "state_dict": (
                model.scorer.state_dict() if scorer_state is None else scorer_state
            ),
        },
        path,
    )


@app.function
def load_bt_reward_model(
    path: str | Path, device: str | None = None
) -> tuple[BradleyTerryRewardModel, list[str]]:
    """Rebuild a saved BT model; returns it with its criterion columns."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint.get("model_type") == "bradley_terry", checkpoint.get(
        "model_type"
    )
    model = BradleyTerryRewardModel(
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


@app.function
def evaluate_loss(
    model: PairwisePreferenceModel,
    data_loader: Iterable[Batch],
    device: str,
) -> float:
    """Average the model's loss over all batches without gradients."""
    model.eval()
    with torch.no_grad():
        losses = [
            model(batch, device, as_loss=True).item() for batch in data_loader
        ]
    return sum(losses) / len(losses)


@app.function
def validation_metrics(
    model: PairwisePreferenceModel,
    validation_loader: Iterable[Batch],
    criterion_columns: list[str],
    device: str,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Pooled masked loss and per-criterion metrics from a single pass.

    A pair is decisive for a criterion when its unmasked target is at most
    0.25 or at least 0.75; accuracy thresholds the logit at zero against the
    preferred side.
    """
    model.eval()
    logit_batches, target_batches, mask_batches = [], [], []
    with torch.no_grad():
        for batch in validation_loader:
            logit_batches.append(model(batch, device).cpu())
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
    return losses[mask].mean().item(), {
        column: {
            "loss": losses[:, index][mask[:, index]].mean().item(),
            "decisive_accuracy": correct[:, index][decisive[:, index]]
            .float()
            .mean()
            .item(),
            "decisive_count": int(decisive[:, index].sum().item()),
        }
        for index, column in enumerate(criterion_columns)
    }


@app.function
def per_criterion_metrics(
    model: PairwisePreferenceModel,
    validation_loader: Iterable[Batch],
    criterion_columns: list[str],
    device: str,
) -> dict[str, dict[str, float]]:
    """Per-criterion mean loss and decisive-pair accuracy on the validation set."""
    return validation_metrics(model, validation_loader, criterion_columns, device)[1]


@app.function
def train_until_no_improvement(
    model: PairwisePreferenceModel,
    train_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
    learning_rate: float,
    warmup_steps: int,
    device: str,
    steps_per_epoch: int,
    patience: int = 2,
    max_rounds: int | None = None,
    total_steps: int | None = None,
    accumulation_steps: int = 1,
    log_metrics: Callable[[dict], None] | None = None,
    eval_every: int | None = None,
    eval_loader: Iterable[Batch] | None = None,
    eval_criteria: list[str] | None = None,
    accelerator=None,
    checkpoint_directory: Path | None = None,
) -> float:
    """Train in rounds of ``steps_per_epoch`` steps, validating after each round.

    Stops after ``patience`` rounds without a new best validation loss,
    restores the best-validation weights and returns the best validation loss.
    Raises RuntimeError if no round ever produced a finite best loss, so a
    diverged model is never handed back as if it had trained.

    ``max_rounds``, when given, caps the number of training rounds — for smoke
    tests and timing runs — stopping after that round's validation even while
    the loss is still improving.

    ``total_steps``, when given, switches the post-warmup schedule from
    constant to cosine decay reaching zero at that step — for fixed-length
    runs, typically paired with ``max_rounds`` covering the same horizon.

    ``accumulation_steps`` micro-batches feed each optimizer step (their mean
    loss is what steps the model and the logs), so the effective batch is the
    loader batch times this factor at the memory cost of one micro-batch.

    ``log_metrics``, when given, receives one dict per training step and one
    per validation round, for external monitors such as ``wandb.log``.

    ``eval_every``, when given with ``log_metrics`` and ``eval_criteria``,
    additionally logs ``eval/...`` pooled and per-criterion metrics every that
    many steps, scored on ``eval_loader`` (default: the validation loader),
    plus one pass at step 0 before any update and one on the restored best
    weights at the end. The first is the untrained baseline every curve is
    read against; the last is the checkpoint that actually gets saved, which
    a periodic tick lands on only by coincidence. This is monitoring only;
    early stopping still follows the per-round validation loss.

    ``accelerator``, when given, is an ``accelerate.Accelerator`` whose model
    the caller has already prepared, so the parameters are sharded across
    processes. Every rank must run the same forward passes — a sharded
    forward is a collective, and a rank that skips one leaves the others
    waiting on it — so the eval passes run everywhere and only the logging is
    confined to the main process. For the same reason ``validation_loader``
    and ``eval_loader`` must NOT be prepared: run whole on every rank they
    give every rank the same loss, so every rank picks the same round as best
    and the shards of the restored weights stay in step.

    ``checkpoint_directory`` is where a sharded model's best weights are kept
    between rounds, and it is required alongside ``accelerator``: every rank
    writes one shard of the same checkpoint into it, so a directory each rank
    names for itself scatters the shards and leaves every copy incomplete. It
    must therefore be derived from something all ranks agree on, and live
    somewhere they can all read.
    """
    if eval_every is not None and eval_criteria is None:
        raise ValueError("eval_every requires eval_criteria for metric names")
    if accelerator is None:
        model.to(device)
    else:
        device = str(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if accelerator is not None:
        # The loader is prepared; the optimizer deliberately is not. Preparing
        # it registers it for checkpointing, and a sharded fp32 AdamW's moments
        # are twice the weights -- 64 GB of the 96 GB a snapshot of an 8B model
        # would write, for state nothing here ever reads back. The snapshot
        # exists to restore the best weights at the end, not to resume
        # training. With use_orig_params the parameters keep their own
        # identities, so a plain optimizer steps the local shards correctly
        train_loader = accelerator.prepare(train_loader)

    def on_main() -> bool:
        return accelerator is None or accelerator.is_main_process

    def lr_scale(scheduler_step: int) -> float:
        scale = min(1.0, (scheduler_step + 1) / warmup_steps)
        if total_steps is not None:
            progress = min(
                1.0,
                max(0.0, scheduler_step + 1 - warmup_steps)
                / max(1, total_steps - warmup_steps),
            )
            scale *= 0.5 * (1.0 + math.cos(math.pi * progress))
        return scale

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    def log_eval(at_step: int, prefix: str = "eval") -> None:
        """Score the monitoring loader and log its metrics at ``at_step``.

        ``prefix`` separates the end-of-training pass from the curve: it
        scores the restored best weights, which the model never held at the
        step it is logged against, so it must not land on the ``eval/`` line.
        """
        eval_loss, logged_metrics = validation_metrics(
            model,
            eval_loader if eval_loader is not None else validation_loader,
            eval_criteria,  # pyright: ignore[reportArgumentType]
            device,
        )
        assert log_metrics is not None
        if not on_main():
            model.train()
            return
        log_metrics(
            {"step": at_step, f"{prefix}/loss": eval_loss}
            | {
                f"{prefix}/{criterion}_loss": values["loss"]
                for criterion, values in logged_metrics.items()
            }
            | {
                f"{prefix}/{criterion}_accuracy": values["decisive_accuracy"]
                for criterion, values in logged_metrics.items()
            }
        )
        model.train()

    evaluating = eval_every is not None and log_metrics is not None

    def cycling_batches() -> Iterator[Batch]:
        while True:
            yield from train_loader

    batches = cycling_batches()
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    has_best = False
    # A sharded model's state_dict is not portable between ranks and what it
    # contains varies with the FSDP state-dict type, so the snapshot goes
    # through the accelerator's own checkpointing instead of memory
    if accelerator is not None and checkpoint_directory is None:
        raise ValueError("a sharded model needs checkpoint_directory to snapshot to")
    best_directory = str(checkpoint_directory) if accelerator is not None else None
    if best_directory is not None and on_main():
        Path(best_directory).mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        # The directory has to exist before any rank writes into it
        accelerator.wait_for_everyone()

    def snapshot_best() -> None:
        nonlocal best_state
        if accelerator is None:
            # CPU copies, not an on-device deepcopy: a 4B model cannot afford
            # a second device-resident set of weights
            best_state = {
                key: value.detach().to("cpu", copy=True)
                for key, value in model.state_dict().items()
            }
        else:
            accelerator.save_state(best_directory)

    def restore_best() -> None:
        if accelerator is None:
            assert best_state is not None
            model.load_state_dict(best_state)
            return
        accelerator.load_state(best_directory)
        # Read once and never again: leaving it costs the weights of the model
        # on a shared filesystem, and a failed run already filled one
        accelerator.wait_for_everyone()
        if on_main():
            shutil.rmtree(best_directory, ignore_errors=True)

    rounds_without_improvement = 0
    step = 0
    # The untrained baseline: a randomly initialised head on the pretrained
    # backbone, which is the floor every later eval point is read against
    if evaluating:
        log_eval(0)
    while True:
        model.train()
        with mo.status.progress_bar(
            total=steps_per_epoch,
            title=f"steps {step + 1}-{step + steps_per_epoch}",
            remove_on_exit=True,
        ) as progress:
            for round_step in range(1, steps_per_epoch + 1):
                optimizer.zero_grad()
                step_loss = 0.0
                for batch in islice(batches, accumulation_steps):
                    # Scaling before backward keeps the accumulated gradient
                    # the mean over the effective batch
                    loss = model(batch, device, as_loss=True) / accumulation_steps
                    if accelerator is None:
                        loss.backward()
                    else:
                        accelerator.backward(loss)
                    step_loss += loss.item()
                optimizer.step()
                scheduler.step()
                if log_metrics is not None:
                    # max_norm=inf computes the total gradient norm without
                    # clipping anything; gradients are unchanged by step().
                    # Through the accelerator when sharded, since the plain
                    # call would only see this rank's slice of them
                    clip = (
                        torch.nn.utils.clip_grad_norm_
                        if accelerator is None
                        else accelerator.clip_grad_norm_
                    )
                    grad_norm = clip(model.parameters(), float("inf"))
                    if on_main():
                        log_metrics(
                            {
                                "step": step + round_step,
                                "train_loss": step_loss,
                                "grad_norm": grad_norm.item(),
                                "learning_rate": scheduler.get_last_lr()[0],
                            }
                            # The peak is what an OOM is decided on, and it is
                            # reached inside a step rather than between them,
                            # so it cannot be recovered after the fact
                            | (
                                {
                                    "peak_gib": torch.cuda.max_memory_allocated()
                                    / 1024**3
                                }
                                if device.startswith("cuda")
                                else {}
                            )
                        )
                    if evaluating and (step + round_step) % eval_every == 0:
                        # step() has consumed the gradients and grad_norm has
                        # read them, so they are dead weight — but the next
                        # iteration's zero_grad() is what would free them,
                        # and that runs after this eval. On a 4B fp32 model
                        # they are 15 GiB, which is the difference between
                        # the eval pass fitting and not
                        optimizer.zero_grad(set_to_none=True)
                        log_eval(step + round_step)
                progress.update(subtitle=f"loss={step_loss:.4f}")
        step += steps_per_epoch
        # Same reason as before the periodic eval, and this pass is the larger
        # of the two: it covers the whole validation set, not an --eval-pairs
        # slice of it
        optimizer.zero_grad(set_to_none=True)
        validation_loss = evaluate_loss(model, validation_loader, device)
        if on_main():
            print(f"step {step}: validation_loss={validation_loss:.4f}")
        if log_metrics is not None and on_main():
            log_metrics({"step": step, "validation_loss": validation_loss})
        if validation_loss < best_validation_loss:
            rounds_without_improvement = 0
            best_validation_loss = validation_loss
            snapshot_best()
            has_best = True
        else:
            # A NaN validation loss also lands here, so a diverged run
            # exhausts its patience instead of looping forever on NaN >= best
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break
        if max_rounds is not None and step >= max_rounds * steps_per_epoch:
            break
    if not has_best:
        raise RuntimeError(
            "training produced no best state to restore; "
            f"last validation loss was {validation_loss:.4f}"
        )
    restore_best()
    # The restored best weights, not wherever the last periodic tick landed:
    # this is the checkpoint that gets written out
    if evaluating:
        log_eval(step, prefix="final_eval")
    return best_validation_loss


@app.function
def default_device() -> str:
    """Pick the best available torch device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.function
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
    # At least one validation prompt even on tiny smoke-test subsets: early
    # stopping cannot run against an empty validation loader
    validation_count = max(1, int(len(prompts) * validation_fraction))
    validation_prompts = list(prompts.iloc[:validation_count])
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


@app.function
def train_reward_model(
    dataframe: pd.DataFrame,
    criterion_columns: list[str],
    max_tokens: int,
    encoder_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    warmup_steps: int = 100,
    validation_fraction: float = 0.1,
    augment_presentation_order: bool = True,
    seed: int = 1810,
    device: str | None = None,
) -> tuple[PairwisePreferenceModel, dict]:
    """Train one pairwise model; returns it on the CPU with validation metrics.

    Validation (and with it the early-stopping check) runs every third of a
    pass over the training pairs, so checkpoint selection can catch a peak
    inside the first pass. Early stopping is driven by the pooled validation
    loss; the per-criterion diagnostics are computed once from the restored
    best weights. The model is moved off the device before returning, freeing
    the GPU for the next call.
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
    model = PairwisePreferenceModel(
        encoder_name,
        AutoTokenizer.from_pretrained(encoder_name),
        max_tokens,
        len(criterion_columns),
    )
    validation_loss = train_until_no_improvement(
        model,
        train_loader,
        validation_loader,
        learning_rate,
        warmup_steps,
        device,
        steps_per_epoch=max(1, len(train_loader) // 3),
    )
    criterion_metrics = per_criterion_metrics(
        model, validation_loader, criterion_columns, device
    )
    model.to("cpu")
    model.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return model, {"validation_loss": validation_loss, "criteria": criterion_metrics}


@app.function
def metrics_markdown(checkpoint_name: str, metrics: dict) -> str:
    """Summarise a trained checkpoint's validation metrics."""
    criterion_lines = "\n\n".join(
        f"{criterion}: loss {values['loss']:.4f}, decisive accuracy "
        f"{values['decisive_accuracy']:.1%} over {values['decisive_count']} entries"
        for criterion, values in metrics["criteria"].items()
    )
    return (
        f"Uploaded {checkpoint_name} "
        f"(validation loss {metrics['validation_loss']:.4f}).\n\n{criterion_lines}"
    )


@app.function
def preference_tensor(
    model,
    prompt: str,
    responses: list[str],
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Skew-symmetrised win probabilities, shape (head, response, response)."""
    count = len(responses)
    index_pairs = [(i, j) for i in range(count) for j in range(count) if i != j]
    texts = [
        pairwise_text(prompt, responses[i], responses[j]) for i, j in index_pairs
    ]
    # Length-sorted batches pad to their own longest member rather than the
    # pool's, so the forward spends little of its compute on padding;
    # character length is a good-enough proxy for token length
    order = sorted(range(len(texts)), key=lambda text_index: len(texts[text_index]))
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            logits = model.score(
                [texts[text_index] for text_index in order[start : start + batch_size]],
                device,
            )
            probability_batches.append(torch.sigmoid(logits).cpu())
    probabilities = torch.cat(probability_batches).numpy()[np.argsort(order)]
    tensor = np.full((probabilities.shape[1], count, count), 0.5)
    for (i, j), row in zip(index_pairs, probabilities, strict=True):
        tensor[:, i, j] = row
    return (tensor + 1.0 - tensor.transpose(0, 2, 1)) / 2.0


@app.function
def bt_anchor_preference_rates(
    model: BradleyTerryRewardModel,
    prompt: str,
    responses: list[str],
    anchor: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Per-head win probabilities of each response against the anchor.

    The pointwise counterpart of ``experiments.anchor_preference_rates``.
    Scoring the anchor once alongside the responses gives every probability as
    ``sigmoid(r_response - r_anchor)``, so n+1 forwards replace the joint
    model's 2n -- and no symmetrisation is needed, because a difference of
    scalars is antisymmetric by construction where a joint model has to be
    averaged over both presentation orders.
    """
    texts = [pointwise_text(prompt, response) for response in [*responses, anchor]]
    reward_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            reward_batches.append(
                model.score(texts[start : start + batch_size], device).cpu()
            )
    rewards = torch.cat(reward_batches)
    return torch.sigmoid(rewards[:-1] - rewards[-1]).double().numpy()


@app.function
def bt_preference_tensor(
    model,
    prompt: str,
    responses: list[str],
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Bradley-Terry win probabilities, shape (head, response, response).

    One pointwise forward per response; the sigmoid of reward differences is
    exactly skew-symmetric with a 0.5 diagonal by construction.
    """
    texts = [pointwise_text(prompt, response) for response in responses]
    reward_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            reward_batches.append(
                model.score(texts[start : start + batch_size], device).cpu()
            )
    rewards = torch.cat(reward_batches).T
    logits = rewards.unsqueeze(2) - rewards.unsqueeze(1)
    return torch.sigmoid(logits).double().numpy()


@app.function
def upload_tensors(
    filename: str,
    prompts: list[str],
    criteria: list[str],
    tensor_arrays: dict[str, np.ndarray],
    prefix: str = DATASET,
) -> None:
    """Upload scored tensors, the prompts key covering only scored prompts.

    ``tensor_arrays`` must hold a contiguous ``tensor_0..tensor_{k-1}`` prefix
    of ``prompts``; the truncated prompts key is what lets a resumed run tell
    how far a dead session got.
    """
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / filename
        np.savez(
            path,
            prompts=np.array(prompts[: len(tensor_arrays)]),
            criteria=np.array(criteria),
            **tensor_arrays,  # pyright: ignore[reportArgumentType]
        )
        upload_artifact(path, prefix)


@app.function
def resume_tensors(
    filename: str, prompts: list[str], criteria: list[str], prefix: str = DATASET
) -> dict[str, np.ndarray]:
    """Reload previously scored tensors from a partial hub artifact, if any."""
    if not artifact_exists(filename, prefix):
        return {}
    saved = np.load(artifact_path(filename, prefix))
    saved_prompts = saved["prompts"].tolist()
    assert saved["criteria"].tolist() == criteria
    # A checkpoint from a different prompt order or split must not be resumed:
    # tensor_{i} keys are positional
    assert saved_prompts == prompts[: len(saved_prompts)]
    return {
        f"tensor_{index}": saved[f"tensor_{index}"]
        for index in range(len(saved_prompts))
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prep
    """)


@app.cell
def _():
    import datasets

    return (datasets,)


@app.cell
def _(datasets):
    # HelpSteer2 ships train and validation only; validation is this project's
    # evaluation data. Its preference file labels the same two "train"/"val"
    SOURCE_SPLITS = (("train", "train", "train"), ("validation", "val", "evaluation"))
    responses_by_split = {
        label: datasets.load_dataset(
            "nvidia/HelpSteer2", split=response_split
        ).to_pandas()
        for response_split, _, label in SOURCE_SPLITS
    }
    responses_by_split
    return SOURCE_SPLITS, responses_by_split


@app.cell
def _():
    preferences_dataframe = pd.read_json(
        "hf://datasets/nvidia/HelpSteer2/preference/preference.jsonl.gz", lines=True
    )
    preferences_dataframe
    return (preferences_dataframe,)


@app.cell
def _(SOURCE_SPLITS, preferences_dataframe, responses_by_split):
    pairs_dataframe = pd.concat(
        [
            helpsteer2_pairs(
                responses_by_split[label],
                preferences_dataframe[
                    preferences_dataframe["split"] == preference_split
                ],
            ).assign(split=label)
            for _, preference_split, label in SOURCE_SPLITS
        ],
        ignore_index=True,
    )
    pairs_dataframe
    return (pairs_dataframe,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Split and upload
    """)


@app.cell
def _(pairs_dataframe):
    # HelpSteer2's own train/validation halves are the split: all of train
    # trains the model, validation is the evaluation set, and the ITA prompts
    # are carved out of validation so they are never trained on
    evaluation_rows = pairs_dataframe[pairs_dataframe["split"] == "evaluation"]
    held_out_prompts = ita_holdout_prompts(evaluation_rows)
    split_pairs_dataframe = pairs_dataframe.assign(
        split=pairs_dataframe["split"].mask(
            pairs_dataframe["prompt"].isin(set(held_out_prompts)), "ita_holdout"
        )
    )
    split_pairs_dataframe
    return (split_pairs_dataframe,)


@app.cell
def _():
    upload_button = mo.ui.run_button(label="Upload pairs")
    upload_button
    return (upload_button,)


@app.cell
def _(split_pairs_dataframe, upload_button):
    mo.stop(not upload_button.value)
    upload_pairs(split_pairs_dataframe)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train
    """)


@app.cell
def _():
    encoder_name = "Qwen/Qwen3-4B-Instruct-2507"
    max_tokens = 4000
    learning_rate = 1e-5
    batch_size = 12
    warmup_steps = 100
    return batch_size, encoder_name, learning_rate, max_tokens, warmup_steps


@app.cell
def _():
    downloaded_pairs = pd.read_parquet(pairs_path())
    criterion_columns = [*HELPSTEER2_ATTRIBUTES, "overall"]
    criterion_columns
    return criterion_columns, downloaded_pairs


@app.cell
def _():
    train_button = mo.ui.run_button(label="Train pairwise")
    train_button
    return (train_button,)


@app.cell
def _(
    batch_size,
    criterion_columns,
    downloaded_pairs,
    encoder_name,
    learning_rate,
    max_tokens,
    train_button,
    warmup_steps,
):
    mo.stop(not train_button.value)
    pairwise_model, pairwise_metrics = train_reward_model(
        downloaded_pairs[downloaded_pairs["split"] == "train"],
        criterion_columns,
        max_tokens,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
    )
    checkpoint_name = "pairwise.pt"
    with tempfile.TemporaryDirectory() as checkpoint_temp:
        checkpoint_path = Path(checkpoint_temp) / checkpoint_name
        save_reward_model(
            pairwise_model, criterion_columns, encoder_name, checkpoint_path
        )
        upload_model(checkpoint_path)
    # Release the GPU for later cells: the model and its leftover gradients
    # otherwise sit on it for the rest of the session
    pairwise_model.zero_grad(set_to_none=True)
    pairwise_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mo.md(metrics_markdown(checkpoint_name, pairwise_metrics))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Preference tensors

    Scores the selected backbone's pool under both hub checkpoints: the
    pairwise model into preference_tensors.npz and the Bradley-Terry model
    into preference_tensors_bt.npz. Scoring checkpoints to the hub every 10
    prompts and resumes from the partial artifact, so a run can pick up where
    a dead session stopped.
    """)


@app.cell
def _():
    score_model_dropdown = mo.ui.dropdown(
        options=[
            DEFAULT_BASE_MODEL,
            "google/gemma-2b-it",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        value=DEFAULT_BASE_MODEL,
        label="base model",
    )
    score_button = mo.ui.run_button(label="Score preference tensors")
    mo.hstack([score_model_dropdown, score_button], justify="start")
    return score_button, score_model_dropdown


@app.cell
def _(score_button, score_model_dropdown):
    mo.stop(not score_button.value)
    scoring_prefix = pool_prefix(score_model_dropdown.value)
    mo.stop(
        not artifact_exists("candidates.parquet", scoring_prefix),
        mo.md("candidates.parquet is not on the hub yet, generate candidates first."),
    )
    candidates_dataframe = pd.read_parquet(
        artifact_path("candidates.parquet", scoring_prefix)
    )
    # Row order is the canonical evaluation-prompt order
    evaluation_prompts = candidates_dataframe["prompt"].drop_duplicates().tolist()
    scoring_device = default_device()
    # The full n-by-n tensor is precomputed here because the Blackwell winner
    # is a max-min over the whole pool; the anchor rides at the last
    # sample_index and pool prefixes slice past it
    for checkpoint_file, tensors_file, load_scorer, build_tensor in (
        (
            "pairwise_lr5e-6_bf16.pt",
            "preference_tensors.npz",
            load_reward_model,
            preference_tensor,
        ),
        (
            "bt_lr5e-6_bf16.pt",
            "preference_tensors_bt.npz",
            load_bt_reward_model,
            bt_preference_tensor,
        ),
    ):
        scoring_model, scoring_columns = load_scorer(
            model_path(checkpoint_file), scoring_device
        )
        tensor_arrays = resume_tensors(
            tensors_file, evaluation_prompts, scoring_columns, scoring_prefix
        )
        for tensor_index, tensor_prompt in enumerate(
            mo.status.progress_bar(evaluation_prompts, title=checkpoint_file)
        ):
            if f"tensor_{tensor_index}" in tensor_arrays:
                continue
            prompt_candidates = candidates_dataframe[
                candidates_dataframe["prompt"] == tensor_prompt
            ].sort_values("sample_index")  # pyright: ignore[reportCallIssue]
            tensor_arrays[f"tensor_{tensor_index}"] = build_tensor(
                scoring_model,
                tensor_prompt,
                prompt_candidates["response"].tolist(),
                scoring_device,
                batch_size=16,
            )
            if (tensor_index + 1) % 10 == 0:
                upload_tensors(
                    tensors_file,
                    evaluation_prompts,
                    scoring_columns,
                    tensor_arrays,
                    scoring_prefix,
                )
                if scoring_device.startswith("cuda"):
                    print(
                        f"prompt {tensor_index + 1}: "
                        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated"
                    )
        upload_tensors(
            tensors_file,
            evaluation_prompts,
            scoring_columns,
            tensor_arrays,
            scoring_prefix,
        )
        scoring_model.to("cpu")
        if scoring_device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app.run()
