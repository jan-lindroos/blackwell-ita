"""Fine-tune the pairwise preference model on HelpSteer2 preference pairs.

The joint-scoring counterpart of ``scripts/train_bt_reward.py``. The two
scripts share every training choice — data, splits, graded targets, backbone,
loss, optimiser, schedule, batch order, early stopping and validation metrics —
and differ only in how the pair logit is produced. Here both responses enter
one input (``[RESPONSE 1]`` / ``[RESPONSE 2]``) and the encoder emits the pair
logit per criterion directly, exactly the model the notebooks train.

Run on a GPU node, monitored by wandb:

    python scripts/train_pairwise_preference.py --pairs-path data/pairs.parquet
"""

import argparse
import sys
import time
from pathlib import Path

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



def load_split_pairs(pairs_path: Path | None) -> pd.DataFrame:
    """Load the split preference pairs: local file, hub artifact, or rebuild.

    The rebuild uses the same functions and seed as notebooks/train_prefs.py,
    so it reproduces the hub artifact's pairs and split labels exactly.
    Kept line-for-line identical to the copy in scripts/train_bt_reward.py.
    """
    if pairs_path is not None:
        return pd.read_parquet(pairs_path)
    try:
        return pd.read_parquet(tp.pairs_path())
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
    """Train the pairwise preference model under the shared comparison protocol."""
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
            name=args.wandb_run_name or "pairwise-preference",
            config={
                "model_type": "pairwise_preference",
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
    model = tp.PairwisePreferenceModel(
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
        model, validation_loader, criterion_columns, device
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
            model, test_loader, criterion_columns, device
        )
    scorer_state = gathered_scorer_state(model, accelerator)
    if accelerator is None:
        model.to("cpu")
    checkpoint_path = args.output_dir / "pairwise.pt"
    # The existing save format, so the checkpoint drops straight into the
    # notebooks' load_reward_model and the downstream scoring pipeline
    if on_main:
        tp.save_reward_model(
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
