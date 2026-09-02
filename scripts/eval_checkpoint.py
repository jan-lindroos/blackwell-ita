"""Score a saved pairwise checkpoint on a split of the pairs artifact.

Reruns the same per-criterion evaluation the training scripts report at the
end, against a checkpoint rather than a live run: for comparing checkpoints
after the fact, and for checking that a dtype conversion left the metrics
alone before the smaller copy replaces the original.

    python scripts/eval_checkpoint.py --checkpoint checkpoints/lr1e-5/pairwise.pt
    python scripts/eval_checkpoint.py --checkpoint checkpoints/lr1e-5/pairwise.pt \
        --dtype bfloat16 --save-converted checkpoints/lr1e-5/pairwise_bf16.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import train_prefs as tp
from train_bt_reward import load_bt_reward_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args() -> argparse.Namespace:
    """Parse the checkpoint, the split to score it on, and the dtype."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs-path", type=Path, default=Path("data/pairs.parquet"))
    parser.add_argument(
        "--split",
        default="evaluation",
        help="split label to score; 'evaluation' is what the training runs report",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float32",
        help="cast the encoder to this before scoring. The forward pass already"
        " runs under bfloat16 autocast on CUDA, so bfloat16 weights change the"
        " stored precision far more than the arithmetic",
    )
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument(
        "--save-converted",
        type=Path,
        default=None,
        help="also write the checkpoint back at --dtype, once it has been scored",
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "pairwise", "bradley_terry"],
        default="auto",
        help="how to rebuild the checkpoint. 'auto' reads its model_type tag,"
        " which only Bradley-Terry checkpoints carry -- so an untagged file"
        " from elsewhere is indistinguishable from a pairwise one, and the two"
        " share their state-dict keys. Force it when the provenance is unclear,"
        " and read the loss: a model rebuilt the wrong way scores near chance",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    """Score one checkpoint and print its per-criterion metrics as JSON."""
    args = parse_args()
    device = args.device if args.device else tp.default_device()
    pairs = pd.read_parquet(args.pairs_path)
    frame = pairs[pairs["split"] == args.split]
    if frame.empty:
        raise SystemExit(
            f"no rows for split {args.split!r}; found "
            f"{sorted(pairs['split'].unique())}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    kind = args.model_type
    if kind == "auto":
        kind = checkpoint.get("model_type") or "pairwise"
    if kind == "bradley_terry":
        model, criterion_columns = load_bt_reward_model(args.checkpoint)
    else:
        model, criterion_columns = tp.load_reward_model(args.checkpoint)
    print(f"rebuilt as: {kind} (tag: {checkpoint.get('model_type') or 'none'})")
    # Encoder and head together: a bfloat16 hidden state hitting an fp32 head
    # is a dtype mismatch anywhere autocast is not covering for it. The metrics
    # stay in fp32 regardless, because forward() casts its logits back
    model.scorer.to(DTYPES[args.dtype])
    model.to(device)

    loader = DataLoader(
        tp.PreferencePairDataset(frame, criterion_columns, False),
        batch_size=args.inference_batch_size,
    )
    metrics = tp.per_criterion_metrics(
        cast(tp.PairwisePreferenceModel, model), loader, criterion_columns, device
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "pairs": len(frame),
                "dtype": args.dtype,
                "model_type": kind,
                "metrics": metrics,
            },
            indent=2,
        )
    )

    if args.save_converted is not None:
        args.save_converted.parent.mkdir(parents=True, exist_ok=True)
        model.to("cpu")
        tp.save_reward_model(
            model, criterion_columns, checkpoint["encoder_name"], args.save_converted
        )
        size = args.save_converted.stat().st_size / 1e9
        print(f"wrote {args.save_converted} ({size:.1f} GB)")


if __name__ == "__main__":
    main()
