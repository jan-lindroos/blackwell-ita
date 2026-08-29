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
    import tempfile
    from collections.abc import Iterable, Iterator
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

    RMS_REPO = "jan-lindroos/blackwell-ita-rms"
    ARTIFACTS_REPO = "jan-lindroos/blackwell-ita-artifacts"
    DATASET = "helpsteer2"

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
def prompt_splits(
    prompts: pd.Series, evaluation_count: int = 100, seed: int = 1810
) -> tuple[list[str], list[str], list[str]]:
    """Split unique prompts into a held-out evaluation set and two halves."""
    shuffled = prompts.drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    held_out = shuffled[:evaluation_count]
    remaining = shuffled[evaluation_count:]
    half = len(remaining) // 2
    return held_out, remaining[:half], remaining[half:]


@app.function
def model_path(filename: str) -> Path:
    """Download a reward-model artifact, returning its local cache path."""
    return Path(hf_hub_download(RMS_REPO, f"{DATASET}/{filename}"))


@app.function
def artifact_path(filename: str) -> Path:
    """Download a results artifact, returning its local cache path."""
    return Path(
        hf_hub_download(ARTIFACTS_REPO, f"{DATASET}/{filename}", repo_type="dataset")
    )


@app.function
def artifact_exists(filename: str) -> bool:
    """Check whether a results artifact exists on the hub."""
    return file_exists(ARTIFACTS_REPO, f"{DATASET}/{filename}", repo_type="dataset")


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
def upload_artifact(local_path: Path) -> None:
    """Upload a results artifact to the hub."""
    api = HfApi()
    api.create_repo(ARTIFACTS_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{DATASET}/{local_path.name}",
        repo_id=ARTIFACTS_REPO,
        repo_type="dataset",
    )


@app.function
def upload_dataframe(filename: str, dataframe: pd.DataFrame) -> None:
    """Upload a dataframe to the artifacts repo as a parquet file."""
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / filename
        dataframe.to_parquet(path)
        upload_artifact(path)


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
) -> None:
    """Save a checkpoint with everything needed to rebuild the model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_name": encoder_name,
            "criterion_columns": criterion_columns,
            "max_tokens": model.max_tokens,
            "state_dict": model.scorer.state_dict(),
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
def evaluate_loss(
    model: PairwisePreferenceModel,
    data_loader: Iterable[Batch],
    device: str,
) -> float:
    """Average the model's loss over all batches without gradients."""
    model.eval()
    with torch.no_grad():
        losses = [model.compute_loss(batch, device).item() for batch in data_loader]
    return sum(losses) / len(losses)


@app.function
def per_criterion_metrics(
    model: PairwisePreferenceModel,
    validation_loader: Iterable[Batch],
    criterion_columns: list[str],
    device: str,
) -> dict[str, dict[str, float]]:
    """Per-criterion mean loss and decisive-pair accuracy on the validation set.

    A pair is decisive for a criterion when its unmasked target is at most
    0.25 or at least 0.75; accuracy thresholds the logit at zero against the
    preferred side.
    """
    model.eval()
    logit_batches, target_batches, mask_batches = [], [], []
    with torch.no_grad():
        for batch in validation_loader:
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
    return {
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
def train_until_no_improvement(
    model: PairwisePreferenceModel,
    train_loader: Iterable[Batch],
    validation_loader: Iterable[Batch],
    learning_rate: float,
    warmup_steps: int,
    device: str,
    steps_per_epoch: int,
    patience: int = 2,
) -> float:
    """Train in rounds of ``steps_per_epoch`` steps, validating after each round.

    Stops after ``patience`` rounds without a new best validation loss,
    restores the best-validation weights and returns the best validation loss.
    Raises RuntimeError if no round ever produced a finite best loss, so a
    diverged model is never handed back as if it had trained.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    def cycling_batches() -> Iterator[Batch]:
        while True:
            yield from train_loader

    batches = cycling_batches()
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    rounds_without_improvement = 0
    step = 0
    while True:
        model.train()
        with mo.status.progress_bar(
            total=steps_per_epoch,
            title=f"steps {step + 1}-{step + steps_per_epoch}",
            remove_on_exit=True,
        ) as progress:
            for batch in islice(batches, steps_per_epoch):
                optimizer.zero_grad()
                loss = model.compute_loss(batch, device)
                loss.backward()
                optimizer.step()
                scheduler.step()
                progress.update(subtitle=f"loss={loss.item():.4f}")
        step += steps_per_epoch
        validation_loss = evaluate_loss(model, validation_loader, device)
        print(f"step {step}: validation_loss={validation_loss:.4f}")
        # A NaN validation loss also fails this comparison, so a diverged run
        # exhausts its patience instead of looping forever on NaN >= best
        if not (validation_loss < best_validation_loss):
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break
            continue
        rounds_without_improvement = 0
        best_validation_loss = validation_loss
        # CPU copies, not an on-device deepcopy: a 4B model cannot afford a
        # second device-resident set of weights
        best_state = {
            key: value.detach().to("cpu", copy=True)
            for key, value in model.state_dict().items()
        }
    if best_state is None:
        raise RuntimeError(
            "training produced no best state to restore; "
            f"last validation loss was {validation_loss:.4f}"
        )
    model.load_state_dict(best_state)
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
def upload_tensors(
    filename: str,
    prompts: list[str],
    criteria: list[str],
    tensor_arrays: dict[str, np.ndarray],
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
        upload_artifact(path)


@app.function
def resume_tensors(
    filename: str, prompts: list[str], criteria: list[str]
) -> dict[str, np.ndarray]:
    """Reload previously scored tensors from a partial hub artifact, if any."""
    if not artifact_exists(filename):
        return {}
    saved = np.load(artifact_path(filename))
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
    responses_dataframe = datasets.load_dataset(
        "nvidia/HelpSteer2", split="train"
    ).to_pandas()
    responses_dataframe
    return (responses_dataframe,)


@app.cell
def _():
    preferences_dataframe = pd.read_json(
        "hf://datasets/nvidia/HelpSteer2/preference/preference.jsonl.gz", lines=True
    )
    preferences_dataframe = preferences_dataframe[
        preferences_dataframe["split"] == "train"
    ]
    preferences_dataframe
    return (preferences_dataframe,)


@app.cell
def _(preferences_dataframe, responses_dataframe):
    pairs_dataframe = helpsteer2_pairs(responses_dataframe, preferences_dataframe)
    pairs_dataframe
    return (pairs_dataframe,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Split and upload
    """)


@app.cell
def _(pairs_dataframe):
    held_out_prompts, inference_prompts, evaluate_prompts = prompt_splits(
        pairs_dataframe["prompt"]
    )
    split_labels = (
        {prompt: "evaluation" for prompt in held_out_prompts}
        | {prompt: "inference" for prompt in inference_prompts}
        | {prompt: "evaluate" for prompt in evaluate_prompts}
    )
    split_pairs_dataframe = pairs_dataframe.assign(
        split=pairs_dataframe["prompt"].map(split_labels)
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
    upload_dataframe("pairs.parquet", split_pairs_dataframe)


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
    downloaded_pairs = pd.read_parquet(artifact_path("pairs.parquet"))
    criterion_columns = [*HELPSTEER2_ATTRIBUTES, "overall"]
    criterion_columns
    return criterion_columns, downloaded_pairs


@app.cell
def _():
    half_picker = mo.ui.dropdown(
        options=["inference", "evaluate"], value="inference", label="Training half"
    )
    half_picker
    return (half_picker,)


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
    half_picker,
    learning_rate,
    max_tokens,
    train_button,
    warmup_steps,
):
    mo.stop(not train_button.value)
    training_half = half_picker.value
    pairwise_model, pairwise_metrics = train_reward_model(
        downloaded_pairs[downloaded_pairs["split"] == training_half],
        criterion_columns,
        max_tokens,
        encoder_name=encoder_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
    )
    checkpoint_name = f"pairwise_{training_half}.pt"
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

    Scoring checkpoints to the hub every 10 prompts and resumes from the
    partial artifact, so a run can pick up where a dead session stopped.
    """)


@app.cell
def _():
    score_button = mo.ui.run_button(label="Score preference tensors")
    score_button
    return (score_button,)


@app.cell
def _(score_button):
    mo.stop(not score_button.value)
    mo.stop(
        not artifact_exists("candidates.parquet"),
        mo.md("candidates.parquet is not on the hub yet, generate candidates first."),
    )
    candidates_dataframe = pd.read_parquet(artifact_path("candidates.parquet"))
    # Row order is the canonical evaluation-prompt order
    evaluation_prompts = candidates_dataframe["prompt"].drop_duplicates().tolist()
    scoring_device = default_device()
    # Only the inference half precomputes the full tensor: policies are solved
    # on it. The evaluate-half model scores just the policy support atoms
    # against the anchor, on demand in the experiments notebook
    scoring_model, scoring_columns = load_reward_model(
        model_path("pairwise_inference.pt"), scoring_device
    )
    tensors_file = "preference_tensors_inference.npz"
    tensor_arrays = resume_tensors(tensors_file, evaluation_prompts, scoring_columns)
    for tensor_index, tensor_prompt in enumerate(
        mo.status.progress_bar(evaluation_prompts, title="inference")
    ):
        if f"tensor_{tensor_index}" in tensor_arrays:
            continue
        prompt_candidates = candidates_dataframe[
            candidates_dataframe["prompt"] == tensor_prompt
        ].sort_values("sample_index")  # pyright: ignore[reportCallIssue]
        # The anchor rides at the last sample_index; pool prefixes slice
        # past it
        tensor_arrays[f"tensor_{tensor_index}"] = preference_tensor(
            scoring_model,
            tensor_prompt,
            prompt_candidates["response"].tolist(),
            scoring_device,
            batch_size=16,
        )
        if (tensor_index + 1) % 10 == 0:
            upload_tensors(
                tensors_file, evaluation_prompts, scoring_columns, tensor_arrays
            )
            if scoring_device.startswith("cuda"):
                print(
                    f"prompt {tensor_index + 1}: "
                    f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated"
                )
    upload_tensors(tensors_file, evaluation_prompts, scoring_columns, tensor_arrays)
    scoring_model.to("cpu")
    if scoring_device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    app.run()
