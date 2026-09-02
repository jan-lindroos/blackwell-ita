"""Score a candidate pool into per-prompt preference tensors.

The batch counterpart of the tensor cell in ``notebooks/train_prefs.py``. Each
prompt's tensor is the skew-symmetrised win probability of every ordered pair
in its pool, which is what the Blackwell winner's max-min is solved over: the
LP constrains the policy against every pure opponent column, so no entry of
the matrix is spare.

    python scripts/score_candidates.py --model-type pairwise \
        --checkpoint checkpoints/lr5e-6/pairwise.pt

That is n(n-1) forward passes per prompt, so the run is long and the tensors
are written after every prompt: resubmitting resumes from the last one.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import train_prefs as tp
from train_bt_reward import bt_preference_tensor, load_bt_reward_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    """Parse the checkpoint, the pool, and the scoring batch size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidates-path", type=Path, default=Path("data/candidates.parquet")
    )
    parser.add_argument(
        "--tensors-path",
        type=Path,
        default=Path("data/preference_tensors.npz"),
        help="written after every prompt, and read back to resume",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
        help="bfloat16 halves the weights and matches fp32 to ~1e-4 of loss,"
        " because score() already runs its arithmetic under bfloat16 autocast",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="pairs per forward. No gradients are kept, so this can be far"
        " above the training micro-batch",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="score only this many prompts, for timing runs",
    )
    parser.add_argument(
        "--model-type",
        choices=["pairwise", "bradley_terry"],
        required=True,
        help="how to rebuild the checkpoint. Required, and deliberately not"
        " inferred: the two types share their state-dict keys, so the wrong"
        " one loads cleanly and scores with the wrong input format and the"
        " wrong logit. Every invariant a tensor is checked for survives that"
        " -- skew symmetry and the 0.5 diagonal follow from the sigmoid, not"
        " from the weights being right -- so the mistake produces a hundred"
        " plausible tensors and no error. Confirm with eval_checkpoint.py:"
        " a checkpoint rebuilt the wrong way scores near chance",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    """Score every prompt's pool, saving after each so the run can resume."""
    args = parse_args()
    device = args.device if args.device else tp.default_device()
    candidates = pd.read_parquet(args.candidates_path)
    # The canonical order the tensors are keyed on: tensor_{i} belongs to the
    # i-th prompt of this list, so it must not depend on scoring order
    prompts = candidates["prompt"].drop_duplicates().tolist()
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

    # Both model types save the same encoder state under the same keys, so a
    # Bradley-Terry checkpoint loads cleanly into the joint model and then
    # gets used with the wrong input format and the wrong logit -- silently.
    # The type tag is the only thing that separates them
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # The tag is written only by save_bt_reward_model, so its absence says
    # nothing: an untagged file is as likely a pairwise checkpoint as a
    # Bradley-Terry one saved by code that predates the tag. Present, though,
    # it is authoritative -- and contradicting it is a mistake, not a choice
    scorer_kind = args.model_type
    tagged = checkpoint.get("model_type")
    if tagged is not None and tagged != scorer_kind:
        raise SystemExit(
            f"{args.checkpoint} is tagged {tagged!r} but --model-type says"
            f" {scorer_kind!r}; the tag is written at save time and is right"
        )
    if scorer_kind == "bradley_terry":
        model, criteria = load_bt_reward_model(args.checkpoint)
        build_tensor = bt_preference_tensor
    else:
        model, criteria = tp.load_reward_model(args.checkpoint)
        build_tensor = tp.preference_tensor
    print(
        f"scorer: {scorer_kind} (tag: {tagged or 'none'})"
        f" on {checkpoint['encoder_name']}",
        flush=True,
    )
    model.scorer.to(DTYPES[args.dtype])
    model.to(device)
    model.eval()

    tensors: dict[str, np.ndarray] = {}
    if args.tensors_path.exists():
        saved = np.load(args.tensors_path)
        if saved["prompts"].tolist() != prompts[: len(saved["prompts"])]:
            raise SystemExit(
                f"{args.tensors_path} was scored over a different prompt order;"
                " delete it or point --tensors-path somewhere else"
            )
        # Resuming across model types would splice two different scorers'
        # tensors into one artifact, and nothing downstream could tell
        saved_kind = str(saved["scorer"]) if "scorer" in saved.files else "pairwise"
        if saved_kind != scorer_kind:
            raise SystemExit(
                f"{args.tensors_path} was scored by a {saved_kind} model, not"
                f" {scorer_kind}; point --tensors-path somewhere else"
            )
        tensors = {
            f"tensor_{i}": saved[f"tensor_{i}"] for i in range(len(saved["prompts"]))
        }
        print(f"resuming after {len(tensors)} prompts", flush=True)

    args.tensors_path.parent.mkdir(parents=True, exist_ok=True)
    # Halved on any out-of-memory and carried forward, so an overnight run
    # degrades in speed rather than dying: pool sequence lengths vary, and the
    # batch that fits is not known before the first long one arrives
    batch_size = args.batch_size
    for index, prompt in enumerate(prompts):
        if f"tensor_{index}" in tensors:
            continue
        pool = candidates[candidates["prompt"] == prompt].sort_values("sample_index")
        start = time.monotonic()
        while True:
            try:
                tensors[f"tensor_{index}"] = build_tensor(
                    model, prompt, pool["response"].tolist(), device, batch_size
                )
                break
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                torch.cuda.empty_cache()
                batch_size //= 2
                # Kept for the remaining prompts rather than reset: what does
                # not fit once will not fit on the next pool either, and an
                # unattended run should not rediscover that a hundred times
                print(f"  out of memory; retrying at batch {batch_size}", flush=True)
        minutes = (time.monotonic() - start) / 60
        count = len(pool)
        forwards = count if scorer_kind == "bradley_terry" else count * (count - 1)
        print(
            f"prompt {index + 1}/{len(prompts)}: {count}x{count} from {forwards}"
            f" forward(s) in {minutes:.1f} min",
            flush=True,
        )
        # Saved every prompt, not at the end: an n-squared scoring pass will
        # not fit one walltime, and losing hours of it to a timeout is avoidable
        np.savez(
            args.tensors_path,
            prompts=np.array(prompts[: index + 1]),
            criteria=np.array(criteria),
            scorer=np.array(scorer_kind),
            **tensors,
        )

    print(f"{len(tensors)} prompts scored -> {args.tensors_path}")


if __name__ == "__main__":
    main()
