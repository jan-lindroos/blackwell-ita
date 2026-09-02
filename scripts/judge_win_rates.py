"""Score the candidate pools against their anchors under a held-out judge.

Stage two of the experiment, as a batch job: solve the winner policies from a
preference-tensor artifact, then ask a judge that did not produce those tensors
how often each policy beats the prompt's anchor. The win rate is an
expectation, weighted by the policy, so a policy spreading mass over weak
candidates scores worse than its best atom alone would suggest.

    python scripts/judge_win_rates.py --judge checkpoints/judge/pairwise.pt \\
        --model-type pairwise --tensors-path data/preference_tensors.npz \\
        --out data/win_rates.parquet

Only distinct support atoms are scored, and the cache is shared across methods
and pool sizes, so the cost is the union of the supports rather than their sum.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))

import experiments as ex
import train_prefs as tp
from train_bt_reward import bt_anchor_preference_rates, load_bt_reward_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    """Parse the judge, the tensors to solve from, and the output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        choices=["pairwise", "bradley_terry"],
        required=True,
        help="how to rebuild the judge. Required for the same reason as in"
        " score_candidates.py: the two types share their state-dict keys, so"
        " the wrong one loads cleanly and judges with the wrong input format",
    )
    parser.add_argument(
        "--tensors-path", type=Path, default=Path("data/preference_tensors.npz")
    )
    parser.add_argument(
        "--candidates-path", type=Path, default=Path("data/candidates.parquet")
    )
    parser.add_argument("--anchors-path", type=Path, default=Path("data/anchors.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/win_rates.parquet"))
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=ex.N_VALUES,
        help="pool prefixes to solve at; the default matches the notebook",
    )
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def solve(tensor: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """The winner policies for one prompt at one pool size, by method."""
    attributes = tensor[: ex.OVERALL_INDEX, :size, :size]
    return {
        "best_of_blackwell": ex.blackwell_winner(attributes),
        "blackwell_no_verbosity": ex.blackwell_winner(
            tensor[ex.NO_VERBOSITY_HEADS, :size, :size]
        ),
        "entropic_blackwell": ex.blackwell_winner(attributes, beta=ex.BETA),
        "best_of_nash": ex.best_of_nash(tensor[ex.OVERALL_INDEX, :size, :size]),
    }


def main() -> None:
    """Solve every policy, score the atoms they use, and write the rates."""
    args = parse_args()
    device = args.device if args.device else tp.default_device()
    candidates = pd.read_parquet(args.candidates_path)
    anchors = dict(
        zip(
            *pd.read_parquet(args.anchors_path)[["prompt", "anchor"]].to_dict("list").values(),
            strict=True,
        )
    )
    saved = np.load(args.tensors_path)
    prompts = saved["prompts"].tolist()
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    pools = {
        prompt: group.sort_values("sample_index")["response"].tolist()
        for prompt, group in candidates.groupby("prompt")
    }

    checkpoint = torch.load(args.judge, map_location="cpu", weights_only=True)
    tagged = checkpoint.get("model_type")
    if tagged is not None and tagged != args.model_type:
        raise SystemExit(
            f"{args.judge} is tagged {tagged!r} but --model-type says"
            f" {args.model_type!r}; the tag is written at save time and is right"
        )
    if args.model_type == "bradley_terry":
        model, criteria = load_bt_reward_model(args.judge)
        rate_atoms = bt_anchor_preference_rates
    else:
        model, criteria = tp.load_reward_model(args.judge)
        rate_atoms = ex.anchor_preference_rates
    print(f"judge: {args.model_type} (tag: {tagged or 'none'}) on {checkpoint['encoder_name']}")
    model.scorer.to(DTYPES[args.dtype])
    model.to(device)
    model.eval()

    rows = []
    for index, prompt in enumerate(prompts):
        policies = {
            (method, size): policy
            for size in args.pool_sizes
            for method, policy in solve(saved[f"tensor_{index}"], size).items()
        }
        policies[("base", 1)] = np.array([1.0])
        supports = {key: ex.policy_support(policy) for key, policy in policies.items()}
        # One score per distinct atom, shared across every method and pool size
        # that puts weight on it
        wanted = sorted({atom for support in supports.values() for atom, _ in support})
        rates = rate_atoms(
            model,
            prompt,
            [pools[prompt][atom] for atom in wanted],
            anchors[prompt],
            device,
            args.batch_size,
        )
        by_atom = dict(zip(wanted, rates, strict=True))
        for (method, size), support in supports.items():
            expected = sum(weight * by_atom[atom] for atom, weight in support)
            rows.extend(
                {
                    "prompt": prompt,
                    "method": method,
                    "n": size,
                    "criterion": criterion,
                    "win_rate": float(rate),
                }
                for criterion, rate in zip(criteria, expected, strict=True)
            )
        print(f"prompt {index + 1}/{len(prompts)}: {len(wanted)} atom(s)", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    print(f"\n{len(frame)} rows -> {args.out}")
    summary = (
        frame[frame["criterion"] != "overall"]
        .groupby(["method", "n"], as_index=False)["win_rate"]
        .min()
        .pivot(index="method", columns="n", values="win_rate")
    )
    print("\nworst-criterion mean win rate against the anchor:")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
