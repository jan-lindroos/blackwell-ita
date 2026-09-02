"""Read an eval_checkpoint report and say whether it clears the loss gate.

Scoring a candidate pool with a checkpoint that was rebuilt the wrong way
produces tensors that pass every structural check, because skew symmetry and
the 0.5 diagonal follow from the sigmoid rather than from the weights. The one
signal that does separate the two is the loss: a model used with the wrong
input format and the wrong logit scores at chance, which for a binary cross
entropy against any target is log 2 = 0.693.

Prints PASS or FAIL and the overall loss, and exits non-zero on FAIL so a
shell can gate on it.

    python scripts/gate_scoring.py --log slurm-44988119.out
"""

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the log to read and the loss the report has to beat."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--criterion",
        default="overall",
        help="the head to gate on; overall is the one the winner is solved for",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=0.68,
        help="chance is log 2 = 0.693, and every correctly rebuilt run so far"
        " has scored between 0.585 and 0.652, so this separates the two"
        " cleanly without assuming the checkpoint is any good",
    )
    return parser.parse_args()


def main() -> None:
    """Extract the report, compare its loss, and exit non-zero if it fails."""
    args = parse_args()
    text = args.log.read_text(errors="replace")
    # The report is the only balanced brace block carrying a metrics key
    reports = []
    depth, start = 0, None
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if "metrics" in parsed:
                        reports.append(parsed)
                start = None
    if not reports:
        raise SystemExit(f"FAIL: no evaluation report in {args.log}")

    report = reports[-1]
    loss = report["metrics"][args.criterion]["loss"]
    accuracy = report["metrics"][args.criterion]["decisive_accuracy"]
    rebuilt = re.search(r"rebuilt as: (\S+)", text)
    print(
        f"rebuilt as {rebuilt.group(1) if rebuilt else '?'},"
        f" dtype {report.get('dtype')}, {report.get('pairs')} pairs"
    )
    print(f"{args.criterion}: loss {loss:.4f}, accuracy {accuracy:.1%}")
    if loss > args.max_loss:
        raise SystemExit(
            f"FAIL: {loss:.4f} is above {args.max_loss}, near the {0.693:.3f} a"
            " model rebuilt the wrong way would score"
        )
    print(f"PASS: {loss:.4f} <= {args.max_loss}")


if __name__ == "__main__":
    main()
