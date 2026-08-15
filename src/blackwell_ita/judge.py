"""Pairwise response judging via the Claude CLI, with order-swap debiasing."""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import marimo as mo

# Pinned full model ID: a floating alias like "sonnet" can silently resolve to
# a different model between judging runs
JUDGE_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You compare two responses to an instruction and pick the better one. "
    "Reply with exactly one word: FIRST or SECOND."
)


def comparison_prompt(instruction: str, first: str, second: str) -> str:
    """Build the judge prompt for an overall comparison."""
    return f"""Which response to the instruction is better?

Instruction:
{instruction}

First response:
{first}

Second response:
{second}

Reply with exactly one word: FIRST or SECOND."""


def claude_pick(prompt: str, model: str, attempts: int = 3) -> str | None:
    """Ask the Claude CLI to pick FIRST or SECOND; None if unparseable."""
    stderr = ""
    for attempt in range(attempts):
        completed = subprocess.run(
            [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--system-prompt",
                SYSTEM_PROMPT,
                "--tools",
                "",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers": {}}',
                "--no-session-persistence",
                "--model",
                model,
                prompt,
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            # A malformed zero-exit reply (e.g. missing "result") crashes
            # loudly here by design: rare, and a loud abort beats silently
            # corrupting a judging run
            reply = json.loads(completed.stdout)["result"].strip().upper()
            if reply.startswith("FIRST"):
                return "FIRST"
            if reply.startswith("SECOND"):
                return "SECOND"
            return None
        stderr = completed.stderr
        # Transient CLI failures must not abort a judging run mid-flight
        if attempt < attempts - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"claude judge failed after {attempts} attempts: {stderr}")


def outcome(
    instruction: str,
    response: str,
    anchor: str,
    model: str = JUDGE_MODEL,
) -> dict:
    """Judge a response against an anchor in both orders; draws score 0.5."""
    forward = claude_pick(comparison_prompt(instruction, response, anchor), model)
    backward = claude_pick(comparison_prompt(instruction, anchor, response), model)
    if forward == "FIRST" and backward == "SECOND":
        score = 1.0
    elif forward == "SECOND" and backward == "FIRST":
        score = 0.0
    else:
        score = 0.5
    # Raw verdicts distinguish genuine order-swap disagreement from parse
    # failures when analysing draw rates
    return {"score": score, "forward": forward, "backward": backward}


def outcomes(
    comparisons: list[dict], model: str = JUDGE_MODEL, workers: int = 8
) -> list[dict]:
    """Judge many comparisons concurrently, preserving input order."""
    with ThreadPoolExecutor(workers) as pool:
        return list(
            mo.status.progress_bar(
                pool.map(
                    lambda comparison: outcome(**comparison, model=model),
                    comparisons,
                ),
                total=len(comparisons),
                title="judging",
            )
        )
