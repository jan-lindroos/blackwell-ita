"""Fine-tune a Bradley-Terry reward model on HelpSteer2 preference pairs.

The pointwise counterpart of ``scripts/train_pairwise_preference.py``. The two
scripts share every training choice — data, splits, graded targets, backbone,
loss, optimiser, schedule, batch order, early stopping and validation metrics —
and differ only in how the pair logit is produced. Here each response is scored
independently as ``r(prompt, response)`` (one scalar per criterion) and the
pair logit is ``r(first) - r(second)``, the (graded) Bradley-Terry likelihood.

Run on a GPU node, monitored by wandb:

    python scripts/train_bt_reward.py --pairs-path data/pairs.parquet
"""

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import train_prefs as tp

# Held out for the downstream ITA experiment; see notebooks/experiments.py
ITA_HOLDOUT_COUNT = 100


def build_tokenizer(encoder_name: str) -> PreTrainedTokenizerBase:
    """Load the encoder's tokenizer, giving it a pad token if it has none.

    Llama-3.1 and friends ship without one, and score() pads every batch to
    its longest sequence.
    """
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def gathered_scorer_state(model, accelerator) -> dict | None:
    """The encoder's full state dict, gathered when the model is sharded.

    Collective, so every rank calls it even though only the main process
    writes the checkpoint. None when there is nothing to gather.
    """
    if accelerator is None:
        return None
    prefix = "scorer."
    return {
        key.removeprefix(prefix): value
        for key, value in accelerator.get_state_dict(model).items()
        if key.startswith(prefix)
    }



def pointwise_text(prompt: str, response: str) -> str:
    """Format a prompt with a single response for pointwise reward scoring."""
    return f"{prompt}\n\n[RESPONSE]\n{response}"


class BradleyTerryRewardModel(torch.nn.Module):
    """Pointwise reward model trained with a graded Bradley-Terry objective.

    Reuses the pairwise model's ``MultiHeadEncoder`` backbone unchanged; only
    the input format (one response per input) and the logit construction
    (difference of two pointwise scores) differ. Implements ``batch_logits``
    and ``compute_loss`` with the ``PairwisePreferenceModel`` signatures so the
    notebook's training loop and metrics work on it unmodified.
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
        self.scorer = tp.MultiHeadEncoder(encoder_name, head_count)
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

    def batch_logits(self, batch: tp.Batch, device: str) -> torch.Tensor:
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
        self, batch: tp.Batch, device: str, *, as_loss: bool = False
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
        return tp.masked_binary_cross_entropy(logits, batch, device)

    def compute_loss(self, batch: tp.Batch, device: str) -> torch.Tensor:
        """Masked binary cross-entropy between the pair logits and targets."""
        return tp.masked_binary_cross_entropy(
            self.batch_logits(batch, device), batch, device
        )


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
            # as a pairwise model by the notebooks' load_reward_model
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


def load_split_pairs(pairs_path: Path | None) -> pd.DataFrame:
    """Load the split preference pairs: local file, hub artifact, or rebuild.

    The rebuild uses the same functions and seed as notebooks/train_prefs.py,
    so it reproduces the hub artifact's pairs and split labels exactly.
    """
    if pairs_path is not None:
        return pd.read_parquet(pairs_path)
    try:
        return pd.read_parquet(tp.artifact_path("pairs.parquet"))
    except Exception as error:  # noqa: BLE001
        print(f"hub pairs unavailable ({error}); rebuilding from source")
    import datasets

    preferences = pd.read_json(
        "hf://datasets/nvidia/HelpSteer2/preference/preference.jsonl.gz", lines=True
    )
    frames = []
    # HelpSteer2 ships train and validation only; its preference file labels
    # the same two as "train" and "val". The validation split is this
    # project's evaluation data, and its prompts are disjoint from train's
    for response_split, preference_split, label in (
        ("train", "train", "train"),
        ("validation", "val", "evaluation"),
    ):
        responses = datasets.load_dataset(
            "nvidia/HelpSteer2", split=response_split
        ).to_pandas()
        split_pairs = tp.helpsteer2_pairs(
            responses, preferences[preferences["split"] == preference_split]
        )
        frames.append(split_pairs.assign(split=label))
    pairs = pd.concat(frames, ignore_index=True)
    # The ITA prompts are carved out of the evaluation split, not the training
    # split: they must never be trained on, and they must carry a decisive
    # overall preference for select_anchors to use them
    evaluation = pairs[pairs["split"] == "evaluation"]
    held_out = tp.ita_holdout_prompts(evaluation, ITA_HOLDOUT_COUNT)
    return pairs.assign(
        split=pairs["split"].mask(pairs["prompt"].isin(set(held_out)), "ita_holdout")
    )


def parse_args() -> argparse.Namespace:
    """Parse the training hyperparameters, defaults matching the notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=None,
        help="local pairs.parquet; defaults to the hub artifact or a rebuild",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="subsample to this many training prompts (smoke tests)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="cap training rounds and report min/round (smoke and timing runs)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="log eval/... metrics every this many steps (0 disables)",
    )
    parser.add_argument(
        "--eval-pairs",
        type=int,
        default=128,
        help="validation pairs scored per periodic eval (caps its cost)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="skip the final test evaluation on the untrained prompts",
    )
    parser.add_argument(
        "--test-prompts",
        type=int,
        default=None,
        help="subsample the test set to this many prompts (smoke tests)",
    )
    parser.add_argument("--encoder", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="post-warmup schedule; cosine decays to zero over --total-rounds"
        " and trains that fixed length (no early stopping)",
    )
    parser.add_argument(
        "--total-rounds",
        type=int,
        default=6,
        help="fixed training length for --lr-schedule cosine, in rounds",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--accumulation",
        type=int,
        default=1,
        help="micro-batches per optimizer step; effective batch ="
        " batch-size x accumulation at one micro-batch's memory",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=8,
        help="batch for validation/eval/test passes (no-grad, memory-cheap),"
        " independent of the training micro-batch",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1810)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="shard the model across processes with accelerate/FSDP; launch"
        " under `accelerate launch` or torchrun. Needed for encoders whose"
        " fp32 optimiser state alone exceeds one GPU",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb-project", default="blackwell-ita-rm-comparison")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    """Train the BT reward model under the shared comparison protocol."""
    args = parse_args()
    criterion_columns = [*tp.HELPSTEER2_ATTRIBUTES, "overall"]
    pairs = load_split_pairs(args.pairs_path)
    # Three disjoint splits, fixed in the pairs artifact rather than derived
    # here: all of HelpSteer2's train half trains, its validation half is the
    # evaluation set, and ITA_HOLDOUT_COUNT of the latter is reserved for the
    # downstream ITA experiment and never scored here
    frame = pairs[pairs["split"] == "train"]
    evaluation_frame = pairs[pairs["split"] == "evaluation"]
    if frame.empty or evaluation_frame.empty:
        raise RuntimeError(
            f"expected 'train' and 'evaluation' splits, found"
            f" {sorted(pairs['split'].unique())}; rebuild the pairs artifact"
        )
    if args.limit_prompts is not None:
        kept_prompts = (
            frame["prompt"]
            .drop_duplicates()
            .sample(
                n=min(args.limit_prompts, frame["prompt"].nunique()),
                random_state=args.seed,
            )
        )
        frame = frame[frame["prompt"].isin(kept_prompts)]
    accelerator = None
    if args.distributed:
        from accelerate import Accelerator

        accelerator = Accelerator()
    device = (
        str(accelerator.device)
        if accelerator is not None
        else (args.device if args.device else tp.default_device())
    )
    on_main = accelerator is None or accelerator.is_main_process
    if device == "cpu" and args.device is None and accelerator is None:
        raise RuntimeError(
            "no GPU is visible to torch (torch.cuda.is_available() is False); "
            "refusing to silently train a 4B model on CPU. "
            "Pass --device cpu to force CPU training."
        )
    if on_main:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "bt-reward",
            config={
                "model_type": "bradley_terry_reward",
                "criterion_columns": criterion_columns,
                "training_pairs": len(frame),
                "evaluation_pairs": len(evaluation_frame),
                "augment_presentation_order": True,
                "device": device,
            }
            | {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        )

    # Seeded before model construction and loader iteration: with the same
    # seed both compared runs share head initialisation and batch order
    torch.manual_seed(args.seed)
    train_loader, validation_loader = tp.build_explicit_loaders(
        frame,
        evaluation_frame,
        criterion_columns,
        batch_size=args.batch_size,
        augment_presentation_order=True,
    )
    model = BradleyTerryRewardModel(
        args.encoder,
        build_tokenizer(args.encoder),
        args.max_tokens,
        len(criterion_columns),
    )
    if accelerator is not None:
        # Onto the device before sharding. A transformer-based auto-wrap policy
        # only wraps the decoder layers, so the embedding, the final norm and
        # the head sit outside every wrapped unit and FSDP leaves them wherever
        # they already are — on the CPU, where the first embedding lookup meets
        # input ids that are not. The full model is resident only until
        # prepare() shards it away.
        model.to(accelerator.device)
        # Prepared here rather than inside the training loop so the caller
        # keeps the sharded handle it has to save from
        model = accelerator.prepare(model)
    # Evaluation passes are no-grad and memory-cheap, so they run at their own
    # batch size rather than the training micro-batch
    validation_loader = DataLoader(
        validation_loader.dataset, batch_size=args.inference_batch_size
    )
    # A fixed slice of the validation set keeps the periodic eval cheap; the
    # per-round full validation still drives early stopping
    eval_dataset = validation_loader.dataset
    if args.eval_pairs < len(eval_dataset):  # pyright: ignore[reportArgumentType]
        eval_dataset = Subset(eval_dataset, range(args.eval_pairs))
    eval_loader = DataLoader(eval_dataset, batch_size=args.inference_batch_size)
    # A step is an optimizer step consuming `accumulation` micro-batches, so a
    # round stays one third of a pass over the training pairs
    steps_per_epoch = max(1, len(train_loader) // (3 * args.accumulation))
    # Cosine trains a fixed length: the decay horizon and the round cap cover
    # the same steps, and patience is set high enough never to fire first
    max_rounds = args.max_rounds
    patience = args.patience
    total_steps = None
    if args.lr_schedule == "cosine":
        max_rounds = args.total_rounds if max_rounds is None else max_rounds
        total_steps = max_rounds * steps_per_epoch
        patience = max_rounds + 1
    start_time = time.monotonic()
    best_validation_loss = tp.train_until_no_improvement(
        cast(tp.PairwisePreferenceModel, model),
        train_loader,
        validation_loader,
        args.learning_rate,
        args.warmup_steps,
        device,
        steps_per_epoch=steps_per_epoch,
        patience=patience,
        max_rounds=max_rounds,
        total_steps=total_steps,
        accumulation_steps=args.accumulation,
        log_metrics=wandb.log,
        eval_every=args.eval_every if args.eval_every > 0 else None,
        eval_loader=eval_loader,
        eval_criteria=criterion_columns,
        accelerator=accelerator,
        # Derived from the arguments rather than created per rank: every rank
        # parses the same output-dir, so every rank names the same directory,
        # which is what save_state's one-shard-per-rank layout requires
        checkpoint_directory=args.output_dir / ".fsdp-best",
    )
    train_minutes = (time.monotonic() - start_time) / 60.0
    if on_main:
        wandb.log({"train_minutes": train_minutes})
    if args.max_rounds is not None and on_main:
        print(
            f"{args.max_rounds} round(s) took {train_minutes:.1f} min "
            f"(~{train_minutes / args.max_rounds:.1f} min/round with validation); "
            "full runs typically stop after roughly 4-8 rounds"
        )
    metrics = tp.per_criterion_metrics(
        cast(tp.PairwisePreferenceModel, model),
        validation_loader,
        criterion_columns,
        device,
    )
    # Final report over the whole evaluation split with the best weights
    # restored, rather than the --eval-pairs subsample the curves are logged
    # on. It drove early stopping, so it is not an untouched test set
    test_metrics = None
    test_frame = evaluation_frame
    if not args.skip_test:
        if args.test_prompts is not None:
            kept_test_prompts = (
                test_frame["prompt"]
                .drop_duplicates()
                .sample(
                    n=min(args.test_prompts, test_frame["prompt"].nunique()),
                    random_state=args.seed,
                )
            )
            test_frame = test_frame[test_frame["prompt"].isin(kept_test_prompts)]
        test_loader = DataLoader(
            tp.PreferencePairDataset(test_frame, criterion_columns, False),
            batch_size=args.inference_batch_size,
        )
        test_metrics = tp.per_criterion_metrics(
            cast(tp.PairwisePreferenceModel, model),
            test_loader,
            criterion_columns,
            device,
        )
    scorer_state = gathered_scorer_state(model, accelerator)
    if accelerator is None:
        model.to("cpu")
    checkpoint_path = args.output_dir / "bt.pt"
    if on_main:
        save_bt_reward_model(
            model, criterion_columns, args.encoder, checkpoint_path, scorer_state
        )
    reported = (
        {"best_validation_loss": best_validation_loss}
        | {
            f"{criterion}/{key}": value
            for criterion, values in metrics.items()
            for key, value in values.items()
        }
        | (
            {
                f"test/{criterion}/{key}": value
                for criterion, values in test_metrics.items()
                for key, value in values.items()
            }
            if test_metrics is not None
            else {}
        )
    )
    # Every rank reaches the barrier, so none leaves before the gather above
    # has finished on all of them
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if not on_main:
        return
    wandb.log(reported)
    print(f"Saved {checkpoint_path} (best validation loss {best_validation_loss:.4f})")
    for criterion, values in metrics.items():
        print(
            f"  {criterion}: loss {values['loss']:.4f}, decisive accuracy "
            f"{values['decisive_accuracy']:.1%} over {values['decisive_count']} pairs"
        )
    if test_metrics is not None:
        print(f"Evaluation metrics ({len(test_frame)} pairs):")
        for criterion, values in test_metrics.items():
            print(
                f"  {criterion}: loss {values['loss']:.4f}, decisive accuracy "
                f"{values['decisive_accuracy']:.1%} over "
                f"{values['decisive_count']} pairs"
            )
    wandb.finish()


if __name__ == "__main__":
    main()
