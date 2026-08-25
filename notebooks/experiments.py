# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface-hub",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    import subprocess
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    from scipy.optimize import linprog

    ARTIFACTS_REPO = "jan-lindroos/blackwell-ita-artifacts"
    DATASET = "helpsteer2"
    HEADS = [
        "helpfulness",
        "correctness",
        "coherence",
        "complexity",
        "verbosity",
        "overall",
    ]
    OVERALL_INDEX = HEADS.index("overall")
    SAMPLES_PER_PROMPT = 128
    N_VALUES = [1, 4, 16, 64, 128]
    # Pinned full model ID: a floating alias like "sonnet" can silently resolve
    # to a different model between judging runs
    JUDGE_MODEL = "claude-sonnet-5"
    SYSTEM_PROMPT = (
        "You compare two responses to an instruction and pick the better one. "
        "Reply with exactly one word: FIRST or SECOND."
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Winner policies on HelpSteer2. Candidates, anchors and preference tensors
    download from the hub on open. Press the solve button to compute policies
    and upload selections, then the judge button to spend Claude judge calls
    on the policy support atoms.
    """)
    return


@app.function
def artifact_path(filename: str) -> Path:
    """Download an artifact from the hub, returning its local cache path."""
    return Path(
        hf_hub_download(ARTIFACTS_REPO, f"{DATASET}/{filename}", repo_type="dataset")
    )


@app.function
def upload_dataframe(filename: str, dataframe: pd.DataFrame) -> None:
    """Upload a dataframe to the artifacts repo as a parquet file."""
    api = HfApi()
    api.create_repo(ARTIFACTS_REPO, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / filename
        dataframe.to_parquet(path)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"{DATASET}/{filename}",
            repo_id=ARTIFACTS_REPO,
            repo_type="dataset",
        )


@app.function
def prompt_tensor(tensors, prompt_index: int) -> np.ndarray:
    """Per-prompt preference tensor from an npz, keyed on the anchors row index."""
    return tensors[f"tensor_{prompt_index}"]


@app.function
def blackwell_winner(
    preference_tensor: np.ndarray, thresholds: list[float] | None = None
) -> np.ndarray:
    """Blackwell winner policy minimising the worst per-criterion shortfall."""
    # Orthant target set S = {z : z_j >= tau_j}, tau = 1/2 per head by default:
    # minimise the worst clipped shortfall max_{i,j} (tau_j - P_j(pi, e_i))
    # over pure opponents i and criteria j
    head_count, count, _ = preference_tensor.shape
    if thresholds is None:
        thresholds = [0.5] * head_count
    inequality_matrix = np.vstack(
        [
            np.hstack([-preference_tensor[head].T, -np.ones((count, 1))])
            for head in range(head_count)
        ]
    )
    result = linprog(
        c=[0.0] * count + [1.0],
        A_ub=inequality_matrix,
        b_ub=np.repeat(-np.asarray(thresholds), count),
        A_eq=[[1.0] * count + [0.0]],
        b_eq=[1.0],
        bounds=[(0.0, None)] * count + [(0.0, None)],
    )
    if not result.success:
        raise RuntimeError(f"blackwell_winner linprog failed: {result.message}")
    return result.x[:count]


@app.function
def best_of_nash(preference_matrix: np.ndarray) -> np.ndarray:
    """Von Neumann winner policy for a single preference matrix."""
    # The von Neumann winner is exactly the k=1 Blackwell winner: at tau = 1/2
    # the zero-shortfall policies are precisely the Nash equilibria
    return blackwell_winner(preference_matrix[None])


@app.function
def policy_support(
    policy: np.ndarray, threshold: float = 1e-6
) -> list[tuple[int, float]]:
    """Support atoms of a policy as (index, weight) pairs summing to one."""
    # LP solutions carry numerical dust: tiny negative entries and near-zero
    # atoms that would otherwise trigger spurious judge calls
    weights = np.clip(policy, 0.0, None)
    weights[weights < threshold] = 0.0
    weights = weights / weights.sum()
    return [
        (index, float(weight)) for index, weight in enumerate(weights) if weight > 0.0
    ]


@app.function
def solve_policies(tensors, prompts: list[str]) -> dict:
    """Winner policies over pool prefixes, keyed (prompt, method, n)."""
    policies = {}
    for prompt_index, prompt in enumerate(
        mo.status.progress_bar(prompts, title="solving")
    ):
        tensor = prompt_tensor(tensors, prompt_index)
        policies[(prompt, "base", 1)] = np.array([1.0])
        for n in N_VALUES:
            policies[(prompt, "best_of_blackwell", n)] = blackwell_winner(
                tensor[:OVERALL_INDEX, :n, :n]
            )
            policies[(prompt, "best_of_nash", n)] = best_of_nash(
                tensor[OVERALL_INDEX, :n, :n]
            )
    return policies


@app.function
def selections_frame(policies: dict, pools: dict) -> pd.DataFrame:
    """Support atoms of every policy as one tidy selections dataframe."""
    rows = [
        {
            "prompt": prompt,
            "method": method,
            "n": n,
            "sample_index": index,
            "weight": weight,
            "response": pools[prompt][index],
        }
        for (prompt, method, n), policy in policies.items()
        for index, weight in policy_support(policy)
    ]
    return pd.DataFrame(
        rows, columns=["prompt", "method", "n", "sample_index", "weight", "response"]
    )


@app.function
def anchor_win_rates(tensor: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """Per-head expected win rate of a pool-prefix policy against the anchor."""
    # The anchor sits at the last index of every per-prompt tensor, after the
    # SAMPLES_PER_PROMPT pool candidates
    return tensor[:, : len(policy), -1] @ policy


@app.function
def held_out_win_rates(
    policies: dict, tensors, prompt_indices: dict[str, int]
) -> pd.DataFrame:
    """Mean per-head win rate against the anchor under held-out tensors."""
    rows = [
        {"method": method, "n": n, "criterion": criterion, "win_rate": float(rate)}
        for (prompt, method, n), policy in policies.items()
        for criterion, rate in zip(
            HEADS,
            anchor_win_rates(prompt_tensor(tensors, prompt_indices[prompt]), policy),
            strict=True,
        )
    ]
    return (  # pyright: ignore[reportReturnType]
        pd.DataFrame(rows)
        .groupby(["method", "n", "criterion"], as_index=False)["win_rate"]
        .mean()
    )


@app.function
def expected_scores(
    selections: pd.DataFrame, atom_scores: dict[tuple[str, str], float]
) -> pd.DataFrame:
    """Expected score per (prompt, method, n) policy from its atoms' scores."""
    scored = selections.assign(
        score=selections["weight"].to_numpy()
        * np.array(
            [
                atom_scores[(prompt, response)]
                for prompt, response in zip(
                    selections["prompt"], selections["response"], strict=True
                )
            ]
        )
    )
    grouped: pd.DataFrame = scored.groupby(["prompt", "method", "n"], as_index=False)[  # pyright: ignore[reportAssignmentType]
        ["weight", "score"]
    ].sum()
    grouped["score"] = grouped["score"] / grouped["weight"]
    return grouped.drop(columns=["weight"])


@app.function
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


@app.function
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
            check=False,
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
        if attempt < attempts - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"claude judge failed after {attempts} attempts: {stderr}")


@app.function
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


@app.function
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


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _():
    candidates_dataframe = pd.read_parquet(artifact_path("candidates.parquet"))
    anchors_dataframe = pd.read_parquet(artifact_path("anchors.parquet"))
    inference_tensors = np.load(artifact_path("preference_tensors_inference.npz"))
    evaluate_tensors = np.load(artifact_path("preference_tensors_evaluate.npz"))
    prompts = anchors_dataframe["prompt"].tolist()
    prompt_indices = {prompt: index for index, prompt in enumerate(prompts)}
    anchors = dict(
        zip(anchors_dataframe["prompt"], anchors_dataframe["anchor"], strict=True)
    )
    pools = {
        prompt: group.sort_values("sample_index")["response"].tolist()[
            :SAMPLES_PER_PROMPT
        ]
        for prompt, group in candidates_dataframe.groupby("prompt")
    }
    # Per-prompt tensors index rows of anchors.parquet, order their heads as
    # HEADS and carry the anchor as their last row and column; a prompt-order
    # mismatch would silently score every prompt against another prompt's
    # tensors
    for _tensors in (inference_tensors, evaluate_tensors):
        assert _tensors["criteria"].tolist() == HEADS
        assert _tensors["prompts"].tolist() == prompts
    assert prompt_tensor(inference_tensors, 0).shape == (
        len(HEADS),
        SAMPLES_PER_PROMPT + 1,
        SAMPLES_PER_PROMPT + 1,
    )
    len(prompts)
    return anchors, evaluate_tensors, inference_tensors, pools, prompt_indices, prompts


@app.cell
def _():
    solve_button = mo.ui.run_button(label="Compute policies and upload selections")
    solve_button
    return (solve_button,)


@app.cell
def _(inference_tensors, pools, prompts, solve_button):
    mo.stop(not solve_button.value)
    policies = solve_policies(inference_tensors, prompts)
    selections_dataframe = selections_frame(policies, pools)
    upload_dataframe("selections.parquet", selections_dataframe)
    selections_dataframe
    return policies, selections_dataframe


@app.cell
def _(evaluate_tensors, policies, prompt_indices):
    win_rates_dataframe = held_out_win_rates(policies, evaluate_tensors, prompt_indices)
    win_rates_dataframe
    return (win_rates_dataframe,)


@app.cell
def _(plt, win_rates_dataframe):
    worst_rates = (
        win_rates_dataframe[win_rates_dataframe["criterion"].isin(HEADS[:OVERALL_INDEX])]
        .groupby(["method", "n"], as_index=False)["win_rate"]
        .min()
    )
    metric_figure, metric_axes = plt.subplots(figsize=(5, 3.2))
    metric_axes.axhline(
        worst_rates.loc[worst_rates["method"] == "base", "win_rate"].item(),
        color="grey",
        linestyle=":",
        linewidth=1,
        label="base",
    )
    for line_method in ("best_of_nash", "best_of_blackwell"):
        line_stats = worst_rates[worst_rates["method"] == line_method].sort_values("n")
        metric_axes.plot(
            line_stats["n"], line_stats["win_rate"], marker="o", label=line_method
        )
    metric_axes.set_xscale("log", base=2)
    metric_axes.set_xlabel("N (pool size)")
    metric_axes.set_ylabel("Worst-criterion win rate")
    metric_axes.legend(frameon=False)
    metric_axes.spines[["top", "right"]].set_visible(False)
    metric_figure
    return


@app.cell
def _():
    judge_button = mo.ui.run_button(label="Judge with Claude")
    judge_button
    return (judge_button,)


@app.cell
def _(anchors, judge_button, selections_dataframe):
    mo.stop(not judge_button.value)
    # Expectation scoring: judge each distinct support atom once against the
    # anchor, then average atom scores under each policy's weights, so the
    # atom cache is shared across methods and pool sizes
    atoms = selections_dataframe[["prompt", "response"]].drop_duplicates()
    comparisons = [
        {"instruction": prompt, "response": response, "anchor": anchors[prompt]}
        for prompt, response in zip(atoms["prompt"], atoms["response"], strict=True)
    ]
    atom_scores = {
        (comparison["instruction"], comparison["response"]): judgement["score"]
        for comparison, judgement in zip(comparisons, outcomes(comparisons), strict=True)
    }
    judge_scores_dataframe = expected_scores(selections_dataframe, atom_scores)
    upload_dataframe("judge_scores.parquet", judge_scores_dataframe)
    judge_scores_dataframe
    return (judge_scores_dataframe,)


@app.cell
def _(judge_scores_dataframe):
    judge_summary = judge_scores_dataframe.groupby(["method", "n"], as_index=False)[
        "score"
    ].mean()
    judge_summary.pivot(index="n", columns="method", values="score").round(3)
    return (judge_summary,)


@app.cell
def _(judge_summary, plt):
    judge_figure, judge_axes = plt.subplots(figsize=(5, 3.2))
    judge_axes.axhline(
        judge_summary.loc[judge_summary["method"] == "base", "score"].item(),
        color="grey",
        linestyle=":",
        linewidth=1,
        label="base",
    )
    for curve_method in ("best_of_nash", "best_of_blackwell"):
        curve_stats = judge_summary[judge_summary["method"] == curve_method].sort_values(
            "n"
        )
        judge_axes.plot(
            curve_stats["n"], curve_stats["score"], marker="o", label=curve_method
        )
    judge_axes.set_xscale("log", base=2)
    judge_axes.set_xlabel("N (pool size)")
    judge_axes.set_ylabel("Expected overall win rate vs anchor")
    judge_axes.legend(frameon=False)
    judge_axes.spines[["top", "right"]].set_visible(False)
    judge_figure
    return


if __name__ == "__main__":
    app.run()
