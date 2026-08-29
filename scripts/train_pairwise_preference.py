"""Fine-tune the pairwise preference model on HelpSteer2 preference pairs.

The joint-scoring counterpart of ``scripts/train_bt_reward.py``. The two
scripts share every training choice — data, splits, graded targets, backbone,
loss, optimiser, schedule, batch order, early stopping and validation metrics —
and differ only in how the pair logit is produced. Here both responses enter
one input (``[RESPONSE 1]`` / ``[RESPONSE 2]``) and the encoder emits the pair
logit per criterion directly, exactly the model the notebooks train.

Run on a GPU node, monitored by wandb:

    python scripts/train_pairwise_preference.py --half inference
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import train_prefs as tp


def load_split_pairs(pairs_path: Path | None) -> pd.DataFrame:
    """Load the split preference pairs: local file, hub artifact, or rebuild.

    The rebuild uses the same functions and seed as notebooks/train_prefs.py,
    so it reproduces the hub artifact's pairs and split labels exactly.
    Kept line-for-line identical to the copy in scripts/train_bt_reward.py.
    """
    if pairs_path is not None:
        return pd.read_parquet(pairs_path)
    try:
        return pd.read_parquet(tp.artifact_path("pairs.parquet"))
    except Exception as error:  # noqa: BLE001
        print(f"hub pairs unavailable ({error}); rebuilding from source")
    import datasets

    responses = datasets.load_dataset("nvidia/HelpSteer2", split="train").to_pandas()
    preferences = pd.read_json(
        "hf://datasets/nvidia/HelpSteer2/preference/preference.jsonl.gz", lines=True
    )
    preferences = preferences[preferences["split"] == "train"]
    pairs = tp.helpsteer2_pairs(responses, preferences)
    held_out, inference, evaluate = tp.prompt_splits(pairs["prompt"])
    labels = (
        {prompt: "evaluation" for prompt in held_out}
        | {prompt: "inference" for prompt in inference}
        | {prompt: "evaluate" for prompt in evaluate}
    )
    return pairs.assign(split=pairs["prompt"].map(labels))


def parse_args() -> argparse.Namespace:
    """Parse the training hyperparameters, defaults matching the notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--half",
        choices=["inference", "evaluate"],
        default="inference",
        help="prompt half to train on; keep identical across compared runs",
    )
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
        default=50,
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
        help="skip the final test evaluation on the untrained prompt half",
    )
    parser.add_argument(
        "--test-prompts",
        type=int,
        default=None,
        help="subsample the test half to this many prompts (smoke tests)",
    )
    parser.add_argument("--encoder", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-tokens", type=int, default=4000)
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
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1810)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb-project", default="blackwell-ita-rm-comparison")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    """Train the pairwise preference model under the shared comparison protocol."""
    args = parse_args()
    criterion_columns = [*tp.HELPSTEER2_ATTRIBUTES, "overall"]
    pairs = load_split_pairs(args.pairs_path)
    frame = pairs[pairs["split"] == args.half]
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
    device = args.device if args.device else tp.default_device()
    if device == "cpu" and args.device is None:
        raise RuntimeError(
            "no GPU is visible to torch (torch.cuda.is_available() is False); "
            "refusing to silently train a 4B model on CPU. "
            "Pass --device cpu to force CPU training."
        )
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"pairwise-preference-{args.half}",
        config={
            "model_type": "pairwise_preference",
            "criterion_columns": criterion_columns,
            "training_pairs": len(frame),
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
    train_loader, validation_loader = tp.build_loaders(
        frame,
        criterion_columns,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        augment_presentation_order=True,
        seed=args.seed,
    )
    model = tp.PairwisePreferenceModel(
        args.encoder,
        AutoTokenizer.from_pretrained(args.encoder),
        args.max_tokens,
        len(criterion_columns),
    )
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
        model,
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
    )
    train_minutes = (time.monotonic() - start_time) / 60.0
    wandb.log({"train_minutes": train_minutes})
    if args.max_rounds is not None:
        print(
            f"{args.max_rounds} round(s) took {train_minutes:.1f} min "
            f"(~{train_minutes / args.max_rounds:.1f} min/round with validation); "
            "full runs typically stop after roughly 4-8 rounds"
        )
    metrics = tp.per_criterion_metrics(
        model, validation_loader, criterion_columns, device
    )
    # Final report on the untrained half: untouched by training and by the
    # early stopping that selected the checkpoint, unlike the validation set
    test_metrics = None
    test_half = "evaluate" if args.half == "inference" else "inference"
    test_frame = pairs[pairs["split"] == test_half]
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
            model, test_loader, criterion_columns, device
        )
    model.to("cpu")
    checkpoint_path = args.output_dir / f"pairwise_{args.half}.pt"
    # The existing save format, so the checkpoint drops straight into the
    # notebooks' load_reward_model and the downstream scoring pipeline
    tp.save_reward_model(model, criterion_columns, args.encoder, checkpoint_path)
    wandb.log(
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
    print(f"Saved {checkpoint_path} (best validation loss {best_validation_loss:.4f})")
    for criterion, values in metrics.items():
        print(
            f"  {criterion}: loss {values['loss']:.4f}, decisive accuracy "
            f"{values['decisive_accuracy']:.1%} over {values['decisive_count']} pairs"
        )
    if test_metrics is not None:
        print(f"Test metrics on the {test_half} half ({len(test_frame)} pairs):")
        for criterion, values in test_metrics.items():
            print(
                f"  {criterion}: loss {values['loss']:.4f}, decisive accuracy "
                f"{values['decisive_accuracy']:.1%} over "
                f"{values['decisive_count']} pairs"
            )
    wandb.finish()


if __name__ == "__main__":
    main()
