"""Compare the winner policies two reward models' tensors solve to.

The alignment result is downstream of whichever reward model scored the pools,
and nothing in the pipeline says how much that choice matters. Solving the
same prompts from two artifacts answers it: policies that agree mean the
result is a property of the method, policies that diverge mean it is partly a
property of the scorer, and either way it belongs in the write-up.

    python scripts/compare_tensors.py \\
        --left data/preference_tensors.npz --right data/preference_tensors_bt.npz

Reports, per pool size: support sizes, how often the two agree on the
highest-weight candidate, Jaccard overlap of the supports, and total variation
distance. Solving is seconds, so this costs nothing next to the scoring runs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import experiments as ex


def parse_args() -> argparse.Namespace:
    """Parse the two artifacts and the pool sizes to solve at."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[4, 16, 64, 128])
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="compare only the first this many prompts",
    )
    return parser.parse_args()


def support_set(policy: np.ndarray) -> set[int]:
    """Indices carrying weight, under the same threshold the judge pass uses."""
    return {index for index, _ in ex.policy_support(policy)}


def main() -> None:
    """Solve both artifacts at every pool size and report their agreement."""
    args = parse_args()
    left, right = np.load(args.left), np.load(args.right)
    prompts = left["prompts"].tolist()
    if right["prompts"].tolist() != prompts:
        raise SystemExit(
            "the two artifacts cover different prompts or a different order;"
            " tensor_{i} is positional, so they are not comparable"
        )
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    print(f"{len(prompts)} prompts | left {args.left.name} | right {args.right.name}")

    for size in args.pool_sizes:
        rows = []
        for index in range(len(prompts)):
            first = left[f"tensor_{index}"][: ex.OVERALL_INDEX, :size, :size]
            second = right[f"tensor_{index}"][: ex.OVERALL_INDEX, :size, :size]
            left_policy = ex.blackwell_winner(first)
            right_policy = ex.blackwell_winner(second)
            left_support, right_support = (
                support_set(left_policy),
                support_set(right_policy),
            )
            union = left_support | right_support
            # Nash is solved on the overall head alone, so it is a second,
            # cheaper reading of whether the two scorers rank the pool alike
            left_nash = ex.best_of_nash(
                left[f"tensor_{index}"][ex.OVERALL_INDEX, :size, :size]
            )
            right_nash = ex.best_of_nash(
                right[f"tensor_{index}"][ex.OVERALL_INDEX, :size, :size]
            )
            rows.append(
                {
                    "left_support": len(left_support),
                    "right_support": len(right_support),
                    "top1_agree": int(left_policy.argmax() == right_policy.argmax()),
                    "jaccard": len(left_support & right_support) / len(union),
                    # Total variation: half the L1 distance, so 0 is identical
                    # and 1 is disjoint
                    "total_variation": 0.5
                    * float(np.abs(left_policy - right_policy).sum()),
                    "nash_top1_agree": int(left_nash.argmax() == right_nash.argmax()),
                }
            )
        table = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        print(
            f"  n={size:4}: support {table['left_support'].mean():4.1f} vs"
            f" {table['right_support'].mean():4.1f}"
            f" | top-1 agree {table['top1_agree'].mean():5.1%}"
            f" | jaccard {table['jaccard'].mean():.3f}"
            f" | TV {table['total_variation'].mean():.3f}"
            f" | nash top-1 {table['nash_top1_agree'].mean():5.1%}"
        )


if __name__ == "__main__":
    main()
