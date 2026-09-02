"""Sample a base-policy candidate pool for the ITA hold-out prompts.

The batch counterpart of the generation cells in
``notebooks/generate_candidates.py``: same anchors, same sampler, same
combination, run on a GPU node instead of a marimo session.

    python scripts/build_candidates.py --pairs-path data/pairs.parquet

Writes data/anchors.parquet and data/candidates.parquet. Generation is
resumed per prompt from an existing candidates file, so a job that runs out
of walltime can be resubmitted and pick up where it stopped.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import generate_candidates as gc
import train_prefs as tp


def parse_args() -> argparse.Namespace:
    """Parse the pool size, the sampler settings and the output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=Path("data/pairs.parquet"))
    parser.add_argument(
        "--split",
        default="ita_holdout",
        help="split whose prompts get a candidate pool",
    )
    parser.add_argument("--base-model", default="RLHFlow/LLaMA3-SFT-v2")
    parser.add_argument("--samples-per-prompt", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=32,
        help="sequences per generate() call; the notebook default of 8 leaves"
        " an 80GB card idle for an 8B policy",
    )
    parser.add_argument("--seed", type=int, default=1810)
    parser.add_argument("--anchors-path", type=Path, default=Path("data/anchors.parquet"))
    parser.add_argument(
        "--candidates-path", type=Path, default=Path("data/candidates.parquet")
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    """Pick the anchors, sample the pool for each prompt, and write both out."""
    args = parse_args()
    pairs = pd.read_parquet(args.pairs_path)
    frame = pairs[pairs["split"] == args.split]
    if frame.empty:
        raise SystemExit(
            f"no rows for split {args.split!r}; found "
            f"{sorted(pairs['split'].unique())}"
        )
    anchors = gc.select_anchors(frame.assign(split=args.split))
    # select_anchors reads the ita_holdout label, and skips pairs whose overall
    # preference is missing or tied. The hold-out was drawn from decisive pairs
    # precisely so this cannot silently shrink the pool
    if len(anchors) != frame["prompt"].nunique():
        raise SystemExit(
            f"{len(anchors)} anchors for {frame['prompt'].nunique()} prompts:"
            " some hold-out pairs carry no decisive overall preference"
        )
    args.anchors_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_parquet(args.anchors_path)
    print(f"{len(anchors)} anchors -> {args.anchors_path}", flush=True)

    done = pd.DataFrame(columns=["prompt", "response"])
    if args.candidates_path.exists():
        existing = pd.read_parquet(args.candidates_path)
        # A prompt counts as done only at the full pool size; a partial one is
        # resampled rather than left short
        counts = existing.groupby("prompt").size()
        complete = set(counts[counts >= args.samples_per_prompt].index)
        done = existing[existing["prompt"].isin(complete)]
        print(f"resuming: {len(complete)} prompts already sampled", flush=True)

    todo = [p for p in anchors["prompt"] if p not in set(done["prompt"])]
    if todo:
        print(f"sampling {args.samples_per_prompt} responses for {len(todo)} prompts",
              flush=True)
        fresh = gc.generate_candidates(
            args.base_model,
            todo,
            args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.generate_batch_size,
            seed=args.seed,
            device=args.device if args.device else tp.default_device(),
        )
        raw = pd.concat([done, fresh], ignore_index=True)
    else:
        raw = done
        print("every prompt already has a full pool", flush=True)

    # Index each prompt's samples 0..N-1 with its anchor appended at N, then
    # count tokens, exactly as the notebook does
    combined = gc.combine_with_anchors(raw, anchors, args.samples_per_prompt)
    combined = combined.assign(
        tokens=gc.token_counts(
            combined["response"].tolist(), AutoTokenizer.from_pretrained(args.base_model)
        )
    )
    combined.to_parquet(args.candidates_path)
    print(
        f"{len(combined)} rows over {combined['prompt'].nunique()} prompts"
        f" -> {args.candidates_path}"
    )


if __name__ == "__main__":
    main()
