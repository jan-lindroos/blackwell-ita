"""Upload checkpoints to a hub repo, retrying around transient network faults.

A 16 GB file over a link that drops once is a 30-minute upload that has to be
restarted; the hub's chunk deduplication means a retry re-sends only what did
not land, so the loop is worth more than it costs. Files already in the repo
at the same size are skipped, which makes the whole thing resumable.

    python scripts/upload_checkpoints.py --repo jiayinglin/blackwell-ita-rms \
        checkpoints/lr5e-6/pairwise.pt:helpsteer2/pairwise_lr5e-6.pt

Needs HF_TOKEN in the environment (scripts/psc_*.sbatch read ~/.hf_token).
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    """Parse the target repo and the local:remote pairs to upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. user/blackwell-ita-rms")
    parser.add_argument(
        "--private",
        action="store_true",
        help="create the repo private if it does not exist yet",
    )
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument(
        "uploads",
        nargs="+",
        metavar="LOCAL:REMOTE",
        help="local checkpoint path and its path within the repo",
    )
    return parser.parse_args()


def main() -> None:
    """Upload every requested file, skipping what is already there."""
    args = parse_args()
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    existing = {
        sibling.rfilename: sibling.size
        for sibling in api.repo_info(args.repo, files_metadata=True).siblings
    }

    for spec in args.uploads:
        local_text, _, remote = spec.partition(":")
        local = Path(local_text)
        if not remote:
            raise SystemExit(f"{spec!r} is not LOCAL:REMOTE")
        if not local.is_file():
            raise SystemExit(f"missing {local}")
        size = local.stat().st_size
        if existing.get(remote) == size:
            print(f"skip {remote}: already there at {size / 1e9:.1f} GB", flush=True)
            continue

        print(f"{local} -> {remote} ({size / 1e9:.1f} GB)", flush=True)
        for attempt in range(1, args.attempts + 1):
            start = time.monotonic()
            try:
                api.upload_file(
                    path_or_fileobj=local, path_in_repo=remote, repo_id=args.repo
                )
            except Exception as error:  # any fault here is worth a retry
                if attempt == args.attempts:
                    raise
                # Linear rather than exponential: these are minute-scale
                # transfers, and the faults seen are transient CAS timeouts
                pause = 30 * attempt
                print(
                    f"  attempt {attempt}/{args.attempts} failed after "
                    f"{(time.monotonic() - start) / 60:.1f} min "
                    f"({type(error).__name__}); retrying in {pause}s",
                    flush=True,
                )
                time.sleep(pause)
                continue
            minutes = (time.monotonic() - start) / 60
            rate = size / 1e9 / max(minutes, 1e-9) * 60
            print(f"  done in {minutes:.1f} min ({rate:.1f} GB/h)", flush=True)
            break


if __name__ == "__main__":
    main()
