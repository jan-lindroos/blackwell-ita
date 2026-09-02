"""Check a preference-tensor artifact before the winner is solved on it.

Every property here is one the Blackwell LP relies on and none of which the
scoring run asserts for itself: a tensor that is the wrong shape, not
skew-symmetric, or outside [0, 1] still solves to *something*, so the failure
would surface as a quietly wrong policy rather than an error.

    python scripts/check_tensors.py --tensors-path data/preference_tensors.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import experiments as ex


def parse_args() -> argparse.Namespace:
    """Parse the artifact path and the pool sizes to solve at."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensors-path", type=Path, default=Path("data/preference_tensors.npz")
    )
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=[4, 16, 128],
        help="solve the winner at these prefixes of the first prompt's pool",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    """Report the artifact's shape, invariants, and one solved policy."""
    args = parse_args()
    saved = np.load(args.tensors_path)
    keys = [key for key in saved.files if key.startswith("tensor_")]
    prompts, criteria = saved["prompts"], saved["criteria"]
    print(f"tensors: {len(keys)}  prompts: {len(prompts)}  criteria: {len(criteria)}")
    print(f"criteria: {criteria.tolist()}")

    faults: list[str] = []
    for index in range(len(keys)):
        tensor = saved[f"tensor_{index}"]
        head_count, rows, columns = tensor.shape
        if head_count != len(criteria) or rows != columns:
            faults.append(f"tensor_{index}: shape {tensor.shape}")
            continue
        # The scoring pass skew-symmetrises explicitly, so a violation here
        # means the artifact was assembled from mismatched halves
        if not np.allclose(
            tensor + tensor.transpose(0, 2, 1), 1.0, atol=args.tolerance
        ):
            faults.append(f"tensor_{index}: not skew-symmetric")
        elif not np.allclose(
            np.diagonal(tensor, axis1=1, axis2=2), 0.5, atol=args.tolerance
        ):
            faults.append(f"tensor_{index}: diagonal is not 0.5")
        elif tensor.min() < -args.tolerance or tensor.max() > 1 + args.tolerance:
            faults.append(f"tensor_{index}: outside [0, 1]")

    shapes = {saved[f"tensor_{i}"].shape for i in range(len(keys))}
    print(f"shapes: {sorted(shapes)}")
    print(f"faults: {faults[:5] if faults else 'none'}")

    # The LP is what the artifact exists for, so solving it is the real check
    tensor = saved["tensor_0"]
    overall_index = len(criteria) - 1
    for size in args.pool_sizes:
        if size > tensor.shape[1]:
            print(f"  n={size}: skipped, pool is {tensor.shape[1]}")
            continue
        policy = ex.blackwell_winner(tensor[:overall_index, :size, :size])
        support = int((policy > args.tolerance).sum())
        print(
            f"  n={size:4}: support {support:4}/{size}, mass {policy.sum():.6f},"
            f" max weight {policy.max():.4f}"
        )

    if faults:
        raise SystemExit(f"{len(faults)} tensor(s) failed their invariants")


if __name__ == "__main__":
    main()
