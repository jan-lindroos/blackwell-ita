# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cvxpy",
#     "huggingface-hub",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
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
    from collections.abc import Callable
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from huggingface_hub import HfApi, file_exists, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    ARTIFACTS_REPO = "blackwell-ita/artifacts"
    DATASET = "helpsteer2"
    DEFAULT_BASE_MODEL = "RLHFlow/LLaMA3-SFT-v2"
    BACKBONES = [
        DEFAULT_BASE_MODEL,
        "google/gemma-2b-it",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    HEADS = [
        "helpfulness",
        "correctness",
        "coherence",
        "complexity",
        "verbosity",
        "overall",
    ]
    OVERALL_INDEX = HEADS.index("overall")
    NO_VERBOSITY_HEADS = [
        head for head in range(OVERALL_INDEX) if HEADS[head] != "verbosity"
    ]
    # Scored by the Phi-4-mini evaluation model; the welfare metrics read from
    # these so no arm is graded by the tensors its LP optimised against
    EVALUATION_TENSORS = "preference_tensors_eval.npz"
    SAMPLES_PER_PROMPT = 128
    N_VALUES = [1, 4, 16, 64, 128]
    BETA = 0.05
    # The main Blackwell arm: the four quality attributes plus overall as a
    # fifth criterion, all at tau = 1/2. Any tau above 1/2 on overall reduces
    # to best-of-Nash on 97+ of 100 prompts (the shared shortfall relaxes the
    # other heads by the same amount), so there is no sweep
    QUALITY_HEADS = NO_VERBOSITY_HEADS + [OVERALL_INDEX]
    # Scalarised baseline weights: best-of-Nash on (1 - w) overall + w length
    WEIGHTS = [0.1, 0.25, 0.5]
    # Scorer suffix -> (line style, label) so BT sits dashed on the same axes
    SCORERS = {"": ("-", "pairwise"), "_bt": ("--", "BT")}
    # Verbosity is a descriptive rating, not a quality criterion (it agrees
    # with helpfulness on 54% of decisive pairs, a coin flip), so the reported
    # welfare and worst-criterion metrics run over the four quality heads;
    # the 5-head arm stays as the demonstration that chasing it fails
    WELFARE_HEADS = [HEADS[head] for head in NO_VERBOSITY_HEADS]
    # Chart palette (validated categorical slots) and the neutral for base/Nash
    BLUE, ORANGE, YELLOW, MAGENTA, GREY = (
        "#2a78d6",
        "#eb6834",
        "#eda100",
        "#e87ba4",
        "#8a8983",
    )
    # Pinned full model ID: a floating alias like "sonnet" can silently resolve
    # to a different model between judging runs
    JUDGE_MODEL = "claude-sonnet-5"
    ATOM_COLUMNS = ["instruction", "response", "model", "score", "forward", "backward"]
    SYSTEM_PROMPT = (
        "You compare two responses to an instruction and pick the better one. "
        "Reply with exactly one word: FIRST or SECOND."
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Winner policies on HelpSteer2. Candidates, anchors and the preference
    tensors for the selected backbone download from the hub on open. Press
    the solve button to compute policies and upload selections, then the
    judge button to spend Claude judge calls on the policy support atoms.
    Win rates against the anchor come straight from the tensors.
    """)
    return


@app.function
def pool_prefix(model_name: str) -> str:
    """Artifact path prefix for a backbone's candidate pools.

    The default backbone keeps the original flat helpsteer2/ layout the hub
    artifacts already use; other backbones get their own subfolder so runs
    cannot clobber each other.
    """
    if model_name == DEFAULT_BASE_MODEL:
        return "helpsteer2"
    return f"helpsteer2/{model_name.split('/')[-1].lower()}"


@app.function
def artifact_path(filename: str, prefix: str = DATASET) -> Path:
    """Download an artifact from the hub, returning its local cache path."""
    return Path(
        hf_hub_download(ARTIFACTS_REPO, f"{prefix}/{filename}", repo_type="dataset")
    )


@app.function
def upload_dataframe(filename: str, dataframe: pd.DataFrame, prefix: str) -> None:
    """Upload a dataframe to the artifacts repo as a parquet file."""
    api = HfApi()
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / filename
        dataframe.to_parquet(path)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"{prefix}/{filename}",
            repo_id=ARTIFACTS_REPO,
            repo_type="dataset",
        )


@app.function
def prompt_tensor(tensors, prompt_index: int) -> np.ndarray:
    """Per-prompt preference tensor from an npz, keyed on the anchors row index."""
    return tensors[f"tensor_{prompt_index}"]


@app.function
def blackwell_winner(
    preference_tensor: np.ndarray,
    thresholds: list[float] | None = None,
    beta: float = 0.0,
) -> np.ndarray:
    """Blackwell winner policy minimising the worst per-criterion shortfall.

    beta > 0 adds the entropic regularisation beta * KL(pi || uniform),
    making the objective strictly convex with a unique full-support
    minimiser; beta = 0 is the exact linear programme.
    """
    # Orthant target set S = {z : z_j >= tau_j}, tau = 1/2 per head by default:
    # minimise the worst clipped shortfall max_{i,j} (tau_j - P_j(pi, e_i))
    # over pure opponents i and criteria j, in epigraph form. The t >= 0
    # bound implements the clipping; the entropy term (an exponential cone)
    # differs from KL(pi || uniform) by the constant beta * log(count)
    head_count, count, _ = preference_tensor.shape
    if thresholds is None:
        thresholds = [0.5] * head_count
    policy = cp.Variable(count, nonneg=True)
    shortfall = cp.Variable(nonneg=True)
    objective = shortfall - beta * cp.sum(cp.entr(policy)) if beta > 0 else shortfall
    constraints = [cp.sum(policy) == 1] + [
        preference_tensor[head].T @ policy + shortfall >= thresholds[head]  # pyright: ignore[reportOptionalSubscript]
        for head in range(head_count)
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)  # pyright: ignore[reportArgumentType]
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.SolverError as error:
        if beta == 0:
            raise RuntimeError(f"blackwell_winner solve failed: {error}") from error
        # Clarabel breaks down on a few n = 128 entropic instances; SCS solves
        # them to well within the support threshold
        try:
            problem.solve(solver=cp.SCS)
        except cp.SolverError as scs_error:
            raise RuntimeError(
                f"blackwell_winner solve failed: {scs_error}"
            ) from scs_error
    # Clarabel reports optimal_inaccurate on some entropic solves depending
    # on build; the tolerance slack is far below the 1e-6 support threshold
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or policy.value is None:
        raise RuntimeError(f"blackwell_winner solve failed: {problem.status}")
    return np.asarray(policy.value)


@app.function
def length_preference(tokens: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Pairwise "shorter is better" head: sigmoid of the token difference.

    Entry (a, b) is the probability a beats b, so it is skew-symmetric with
    1/2 on the diagonal like the reward-model heads. A Bradley-Terry
    preference with reward -tokens / scale; scale defaults to the pool's
    mean length, so a gap of one mean length gives sigmoid(1) ~ 0.73. A hard
    0/1 head would be unattainable against the pool's shortest candidate
    and its shortfall would swamp the criteria; the sigmoid keeps it bounded
    """
    lengths = np.asarray(tokens, dtype=float)
    if scale is None:
        scale = float(lengths.mean())
    difference = (lengths[None, :] - lengths[:, None]) / scale
    return 1.0 / (1.0 + np.exp(-difference))


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
def solve_policies(tensors, prompts: list[str], pool_tokens: dict) -> dict:
    """Winner policies over pool prefixes, keyed (prompt, method, n).

    The length arm stacks the length head (scale = the pool's mean length)
    onto the quality criteria as one more head at threshold 1/2; nash_length
    arms are the scalarised baseline, the von Neumann winner of
    (1 - w) overall + w length, with w after the @.
    """
    policies = {}
    for prompt_index, prompt in enumerate(
        mo.status.progress_bar(prompts, title="solving")
    ):
        tensor = prompt_tensor(tensors, prompt_index)
        tokens = np.asarray(pool_tokens[prompt], dtype=float)
        policies[(prompt, "base", 1)] = np.array([1.0])
        for n in N_VALUES:
            mean_length = float(tokens[:n].mean())
            overall = tensor[OVERALL_INDEX, :n, :n]
            quality = tensor[QUALITY_HEADS, :n, :n]
            policies[(prompt, "best_of_nash", n)] = best_of_nash(overall)
            policies[(prompt, "blackwell_all_heads", n)] = blackwell_winner(
                tensor[:, :n, :n]
            )
            policies[(prompt, "blackwell_quality", n)] = blackwell_winner(quality)
            policies[(prompt, "blackwell_attributes", n)] = blackwell_winner(
                tensor[NO_VERBOSITY_HEADS, :n, :n]
            )
            policies[(prompt, "entropic_blackwell", n)] = blackwell_winner(
                quality, beta=BETA
            )
            length = length_preference(tokens[:n], mean_length)
            policies[(prompt, "blackwell_quality_length", n)] = blackwell_winner(
                np.concatenate([quality, length[None]])
            )
            for weight in WEIGHTS:
                policies[(prompt, f"nash_length@{weight:g}", n)] = best_of_nash(
                    (1.0 - weight) * overall + weight * length
                )
    return policies


@app.function
def prompt_results(
    policies: dict, tensors, prompts: list[str], pool_tokens: dict
) -> pd.DataFrame:
    """Per (prompt, method, n): expected per-head win rate vs the anchor and tokens.

    The tensors carry the anchor as their last row and column, so a support
    atom's per-head rate is read straight off its prompt's tensor.
    """
    prompt_indices = {prompt: index for index, prompt in enumerate(prompts)}
    rows = []
    for (prompt, method, n), policy in policies.items():
        support = policy_support(policy)
        tensor = prompt_tensor(tensors, prompt_indices[prompt])
        rates = sum(weight * tensor[:, index, -1] for index, weight in support)
        tokens = sum(weight * pool_tokens[prompt][index] for index, weight in support)
        rows.append(
            {
                "prompt": prompt,
                "method": method,
                "n": n,
                **dict(zip(HEADS, np.asarray(rates, dtype=float), strict=True)),
                "tokens": float(tokens),
            }
        )
    return pd.DataFrame(rows)


@app.function
def anchor_win_rates(policies: dict, tensors, prompts: list[str]) -> pd.DataFrame:
    """Mean per-head win rate of each policy against the anchor."""
    pool_tokens = {prompt: np.zeros(SAMPLES_PER_PROMPT + 1) for prompt in prompts}
    results = prompt_results(policies, tensors, prompts, pool_tokens)
    return (  # pyright: ignore[reportReturnType]
        results.melt(
            id_vars=["method", "n"],
            value_vars=HEADS,
            var_name="criterion",
            value_name="win_rate",
        )
        .groupby(["method", "n", "criterion"], as_index=False)["win_rate"]
        .mean()
    )


@app.function
def summarise(
    results: pd.DataFrame, draws: int = 2000, seed: int = 1810
) -> pd.DataFrame:
    """Mean and 95% band over prompts of each metric per (method, n).

    Metrics: judged overall (score), each of HEADS' win rate against the
    anchor, Rawlsian and Nash welfare over WELFARE_HEADS, and expected
    tokens. Bands are paired:
    the same multinomial prompt weights resample every arm, and each band
    is the arm's mean plus the bootstrap spread of its difference from base,
    so prompt difficulty cancels and the base arm's band has zero width.
    """
    prompts = sorted(results["prompt"].unique())
    count = len(prompts)
    weights = (
        np.random.default_rng(seed).multinomial(
            count, np.full(count, 1.0 / count), size=draws
        )
        / count
    )
    columns = ["score", *HEADS, "tokens"]
    welfare_positions = [1 + HEADS.index(head) for head in WELFARE_HEADS]

    def statistics(means: np.ndarray) -> dict[str, np.ndarray]:
        heads = means[..., welfare_positions]
        return {
            "overall": means[..., 0],
            # Per-criterion win rate against the anchor, named "rm_<head>" so
            # it cannot collide with the judged "overall"
            **{
                f"rm_{head}": means[..., 1 + index]
                for index, head in enumerate(HEADS)
            },
            "rawlsian": heads.min(axis=-1),
            "nash_welfare": np.exp(np.log(heads).mean(axis=-1)),
            "tokens": means[..., -1],
        }

    def matrix_for(group: pd.DataFrame) -> np.ndarray:
        return group.set_index("prompt").reindex(prompts)[columns].to_numpy(dtype=float)

    base = matrix_for(results[results["method"] == "base"])  # pyright: ignore[reportArgumentType]
    base_draws = statistics(weights @ base)
    rows = []
    for (method, n), group in results.groupby(["method", "n"]):  # pyright: ignore[reportGeneralTypeIssues]
        matrix = matrix_for(group)  # pyright: ignore[reportArgumentType]
        point = statistics(matrix.mean(axis=0))
        draws_ = statistics(weights @ matrix)
        for metric, value in point.items():
            if np.isnan(value):
                continue
            paired = draws_[metric] - base_draws[metric]
            centred = paired - paired.mean()
            rows.append(
                {
                    "method": method,
                    "n": n,
                    "metric": metric,
                    "mean": float(value),
                    "lo": float(value + np.percentile(centred, 2.5)),
                    "hi": float(value + np.percentile(centred, 97.5)),
                }
            )
    return pd.DataFrame(rows, columns=["method", "n", "metric", "mean", "lo", "hi"])


@app.function
def cyclic_triad_fraction(
    matrix: np.ndarray, samples: int = 2000, seed: int = 1810
) -> float:
    """Fraction of sampled candidate triads that form a preference cycle."""
    count = matrix.shape[0]
    if count < 3:
        return 0.0
    rng = np.random.default_rng(seed)
    triads = rng.integers(0, count, size=(samples, 3))
    distinct = (
        (triads[:, 0] != triads[:, 1])
        & (triads[:, 1] != triads[:, 2])
        & (triads[:, 0] != triads[:, 2])
    )
    a, b, c = triads[distinct].T
    forward = (matrix[a, b] > 0.5) & (matrix[b, c] > 0.5) & (matrix[c, a] > 0.5)
    backward = (matrix[b, a] > 0.5) & (matrix[c, b] > 0.5) & (matrix[a, c] > 0.5)
    return float((forward | backward).mean())


@app.function
def triad_fractions(tensors, prompts: list[str]) -> pd.DataFrame:
    """Mean cyclic-triad fraction of the overall head per pool size."""
    rows = [
        {
            "n": n,
            "fraction": float(
                np.mean(
                    [
                        cyclic_triad_fraction(
                            prompt_tensor(tensors, index)[OVERALL_INDEX, :n, :n]
                        )
                        for index in range(len(prompts))
                    ]
                )
            ),
        }
        for n in N_VALUES
    ]
    return pd.DataFrame(rows)


@app.function
def policies_from_selections(selections: pd.DataFrame) -> dict:
    """Rebuild policies, keyed (prompt, method, n), from a selections table.

    Selections hold every support atom with its weight, so this reproduces
    the solved policies up to the support threshold and a hub download
    stands in for a fresh solve.
    """
    policies = {}
    for (prompt, method, n), group in selections.groupby(["prompt", "method", "n"]):  # pyright: ignore[reportGeneralTypeIssues]
        policy = np.zeros(int(n))
        policy[group["sample_index"].to_numpy()] = group["weight"].to_numpy()
        policies[(prompt, method, int(n))] = policy
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
def expected_token_counts(policies: dict, pool_tokens: dict) -> pd.DataFrame:
    """Mean expected response token count per (method, n) policy."""
    rows = [
        {
            "method": method,
            "n": n,
            "tokens": float(
                np.asarray(pool_tokens[prompt][: len(policy)]) @ policy
            ),
        }
        for (prompt, method, n), policy in policies.items()
    ]
    return (  # pyright: ignore[reportReturnType]
        pd.DataFrame(rows).groupby(["method", "n"], as_index=False)["tokens"].mean()
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
def claude_pick(prompt: str, model: str, attempts: int = 5) -> str | None:
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


@app.function
def load_atom_judgements(prefix: str) -> pd.DataFrame:
    """A backbone's atom judgement cache from the hub, empty if none exists."""
    try:
        return pd.read_parquet(artifact_path("atom_judgements.parquet", prefix))
    except EntryNotFoundError:
        return pd.DataFrame(columns=ATOM_COLUMNS)


@app.function
def judge_atoms(
    comparisons: list[dict],
    cache: pd.DataFrame,
    model: str = JUDGE_MODEL,
    chunk: int = 64,
    checkpoint: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Judge the comparisons the cache lacks for this model; return the merged cache.

    Atoms are keyed on (instruction, response, model): the anchor is fixed
    per instruction, so a judgement carries over between scorers and runs.
    Judging goes in chunks and ``checkpoint`` receives the merged cache after
    each, so a judge outage part-way loses at most one chunk.
    """
    seen = set(zip(cache["instruction"], cache["response"], cache["model"], strict=True))
    fresh = [
        comparison
        for comparison in comparisons
        if (comparison["instruction"], comparison["response"], model) not in seen
    ]
    for start in range(0, len(fresh), chunk):
        batch = fresh[start : start + chunk]
        rows = [
            {
                "instruction": comparison["instruction"],
                "response": comparison["response"],
                "model": model,
                **judgement,
            }
            for comparison, judgement in zip(
                batch, outcomes(batch, model=model), strict=True
            )
        ]
        cache = pd.concat(
            [cache, pd.DataFrame(rows, columns=ATOM_COLUMNS)], ignore_index=True
        )
        if checkpoint is not None:
            checkpoint(cache)
    return cache


@app.function
def draw_metric(axes, summary: pd.DataFrame, metric: str, arms: list[tuple]) -> None:
    """Lines with 95% bands over N for the given (method, colour, style, label, alpha) arms."""
    for method, colour, style, label, alpha in arms:
        stats = summary[(summary["method"] == method) & (summary["metric"] == metric)]
        stats = stats.sort_values("n")  # pyright: ignore[reportCallIssue]
        if stats.empty:
            continue
        axes.plot(
            stats["n"],
            stats["mean"],
            color=colour,
            linestyle=style,
            linewidth=2,
            marker="o",
            markersize=4,
            alpha=alpha,
            label=label,
        )
        axes.fill_between(
            stats["n"], stats["lo"], stats["hi"], color=colour, alpha=0.08 * alpha, linewidth=0
        )
    base = summary[(summary["method"] == "base") & (summary["metric"] == metric)]["mean"]
    if not base.empty:  # pyright: ignore[reportAttributeAccessIssue]
        axes.axhline(base.item(), color=GREY, linestyle=":", linewidth=1, label="base")
    axes.set_xscale("log", base=2)
    axes.set_xticks(N_VALUES)
    axes.set_xticklabels([str(n) for n in N_VALUES])
    axes.set_xlabel("N (pool size)")
    axes.spines[["top", "right"]].set_visible(False)


@app.function
def hub_results(backbones: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Every (backbone, scorer suffix) per-prompt results table on the hub."""
    tables = {}
    for backbone in backbones:
        prefix = pool_prefix(backbone)
        for suffix in SCORERS:
            filename = f"results{suffix}.parquet"
            if file_exists(ARTIFACTS_REPO, f"{prefix}/{filename}", repo_type="dataset"):
                tables[(backbone, suffix)] = pd.read_parquet(artifact_path(filename, prefix))
    return tables


@app.function
def grader_of(results: pd.DataFrame) -> str:
    """Which model graded a results table's welfare columns."""
    if "grader" in results.columns and len(results):
        return str(results["grader"].iloc[0])
    return "selector (self-graded)"


@app.function
def backbone_rows(summaries: dict[tuple[str, str], pd.DataFrame]) -> list[str]:
    """Backbones with at least one scorer's summary, in BACKBONES order."""
    return [
        backbone for backbone in BACKBONES if any(key[0] == backbone for key in summaries)
    ]


@app.function
def styled(arms: list[tuple], suffix: str) -> list[tuple]:
    """Give (method, colour, label, alpha) arms the scorer's line style and label."""
    style, scorer = SCORERS[suffix]
    return [
        (method, colour, style, f"{label} ({scorer})", alpha)
        for method, colour, label, alpha in arms
    ]


@app.function
def chart_grid(rows: int, panels: int):
    """A rows x panels figure whose axes are always a 2-D array."""
    figure, axes = plt.subplots(
        rows, panels, figsize=(4.7 * panels, 3.6 * rows), constrained_layout=True, squeeze=False
    )
    return figure, axes


@app.function
def legend_outside(sources: list, target) -> None:
    """Deduplicated legend to the right of target from the sources' handles."""
    entries = {}
    for axes in sources:
        for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    target.legend(
        entries.values(),
        entries.keys(),
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )


@app.function
def draw_rows(summaries: dict[tuple[str, str], pd.DataFrame], metrics: list[tuple[str, str]], arms) -> tuple:
    """One row per backbone, one panel per metric, pairwise solid and BT dashed.

    ``arms`` is a list of (method, colour, label, alpha) or a callable taking
    a summary and returning one, for arms that depend on the summary.
    """
    rows = backbone_rows(summaries)
    figure, axes = chart_grid(len(rows), len(metrics))
    for row, backbone in enumerate(rows):
        for suffix in SCORERS:
            summary = summaries.get((backbone, suffix))
            if summary is None:
                continue
            row_arms = styled(arms(summary) if callable(arms) else arms, suffix)  # pyright: ignore[reportArgumentType]
            for column, (metric, _) in enumerate(metrics):
                draw_metric(axes[row, column], summary, metric, row_arms)
        for column, (_, label) in enumerate(metrics):
            axes[row, column].set_ylabel(label)
        axes[row, 0].set_title(backbone.split("/")[-1], loc="left", fontsize=9)
    legend_outside([axes[0, -1]], axes[0, -1])
    return figure, axes


@app.function
def chart_verbosity(summaries: dict[tuple[str, str], pd.DataFrame]):
    """Chart 1: the verbosity head makes the max-min policy chase length."""
    arms = [
        ("blackwell_all_heads", BLUE, "Blackwell, all 6 heads (with verbosity)", 1.0),
        ("blackwell_quality", ORANGE, "Blackwell, 5 quality heads", 1.0),
        ("best_of_nash", GREY, "best-of-Nash", 1.0),
    ]
    metrics = [
        ("overall", "Judged overall win rate vs anchor"),
        ("tokens", "Expected response tokens"),
    ]
    figure, _ = draw_rows(summaries, metrics, arms)
    return figure


@app.function
def chart_quality(summaries: dict[tuple[str, str], pd.DataFrame], grader: str):
    """Chart 2: on quality criteria Blackwell matches Nash and guards the worst one."""
    arms = [
        ("blackwell_quality", ORANGE, "Blackwell, 5 quality heads", 1.0),
        ("blackwell_attributes", ORANGE, "Blackwell, 4 attributes (no overall)", 0.5),
        ("best_of_nash", GREY, "best-of-Nash", 1.0),
    ]
    metrics = [
        ("overall", "Judged overall win rate vs anchor"),
        ("rawlsian", "Rawlsian welfare (worst attribute)"),
        ("nash_welfare", "Nash welfare (geometric mean)"),
    ]
    figure, _ = draw_rows(summaries, metrics, arms)
    figure.suptitle(f"Welfare graded by the {grader}", fontsize=10)
    return figure


@app.function
def matched_weight(summary: pd.DataFrame) -> str:
    """The nash_length arm whose tokens at the largest N are closest to the length arm's."""
    at_max = summary[(summary["n"] == N_VALUES[-1]) & (summary["metric"] == "tokens")]
    target = at_max.loc[at_max["method"] == "blackwell_quality_length", "mean"].item()
    baselines = at_max[at_max["method"].str.startswith("nash_length@")]  # pyright: ignore[reportAttributeAccessIssue]
    return baselines.loc[(baselines["mean"] - target).abs().idxmin(), "method"]  # pyright: ignore[reportAttributeAccessIssue]


@app.function
def chart_length(summaries: dict[tuple[str, str], pd.DataFrame]):
    """Chart 3: conciseness as a criterion, against Nash and a scalarised baseline."""

    def arms(summary: pd.DataFrame) -> list[tuple]:
        baseline = matched_weight(summary)
        return [
            ("best_of_nash", GREY, "best-of-Nash", 1.0),
            ("blackwell_quality", ORANGE, "Blackwell, 5 quality heads", 1.0),
            ("blackwell_quality_length", YELLOW, "Blackwell + length head", 1.0),
            (baseline, MAGENTA, f"scalarised Nash, {baseline.split('@')[1]} length", 1.0),
        ]

    metrics = [
        ("overall", "Judged overall win rate vs anchor"),
        ("tokens", "Expected response tokens"),
    ]
    figure, _ = draw_rows(summaries, metrics, arms)
    return figure


@app.function
def chart_criteria(summaries: dict[tuple[str, str], pd.DataFrame], grader: str, backbone: str):
    """One backbone's per-criterion win rates against the anchor, 2 x 3 panels.

    The judged "overall" panel of the other charts is Claude's verdict; these
    are the evaluation model's six heads, verbosity included as the
    descriptive attribute it is.
    """
    arms = [
        ("blackwell_all_heads", BLUE, "Blackwell, all 6 heads (with verbosity)", 1.0),
        ("blackwell_quality", ORANGE, "Blackwell, 5 quality heads", 1.0),
        ("blackwell_quality_length", YELLOW, "Blackwell + length head", 1.0),
        ("best_of_nash", GREY, "best-of-Nash", 1.0),
    ]
    figure, axes = chart_grid(2, 3)
    for index, head in enumerate(HEADS):
        panel = axes[index // 3, index % 3]
        for suffix in SCORERS:
            summary = summaries.get((backbone, suffix))
            if summary is not None:
                draw_metric(panel, summary, f"rm_{head}", styled(arms, suffix))
        panel.set_ylabel(f"{head.capitalize()} win rate vs anchor")
    figure.suptitle(
        f"{backbone.split('/')[-1]}: per-criterion win rates from the {grader}",
        fontsize=10,
    )
    legend_outside([axes[0, 2]], axes[0, 2])
    return figure


@app.function
def chart_triads(triads: dict[str, pd.DataFrame]):
    """Chart 4: cyclic-triad fraction of the pairwise overall head per backbone.

    BT is transitive by construction, so the pairwise model's near-zero
    cycle rate is why the two scorers pick alike in charts 1 to 3.
    """
    figure, axes = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    for (backbone, frame), colour in zip(triads.items(), (BLUE, ORANGE, YELLOW), strict=False):
        axes.plot(
            frame["n"], frame["fraction"], color=colour, marker="o", linewidth=2, label=backbone.split("/")[-1]
        )
    axes.axhline(0.0, color=GREY, linestyle="--", linewidth=1, label="BT (transitive)")
    axes.set_xscale("log", base=2)
    axes.set_xticks(N_VALUES)
    axes.set_xticklabels([str(n) for n in N_VALUES])
    axes.set_xlabel("N (pool size)")
    axes.set_ylabel("Cyclic triads in the pairwise overall head")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return figure


@app.cell
def _():
    base_model_dropdown = mo.ui.dropdown(
        options=BACKBONES,
        value=DEFAULT_BASE_MODEL,
        label="base model",
    )
    base_model_dropdown
    return (base_model_dropdown,)


@app.cell
def _():
    # The value is the filename suffix shared by the scorer's tensors and the
    # selections and judge scores derived from them
    scorer_dropdown = mo.ui.dropdown(
        options={"pairwise": "", "bradley-terry": "_bt"},
        value="pairwise",
        label="scorer",
    )
    scorer_dropdown
    return (scorer_dropdown,)


@app.cell
def _(base_model_dropdown, scorer_dropdown):
    prefix = pool_prefix(base_model_dropdown.value)
    suffix = scorer_dropdown.value
    candidates_dataframe = pd.read_parquet(artifact_path("candidates.parquet", prefix))
    anchors_dataframe = pd.read_parquet(artifact_path("anchors.parquet", prefix))
    preference_tensors = np.load(
        artifact_path(f"preference_tensors{suffix}.npz", prefix)
    )
    has_evaluation = file_exists(
        ARTIFACTS_REPO, f"{prefix}/{EVALUATION_TENSORS}", repo_type="dataset"
    )
    # Until the evaluation model has scored this backbone's pool the selector's
    # own tensors stand in, and the metrics are self-graded
    evaluation_tensors = (
        np.load(artifact_path(EVALUATION_TENSORS, prefix))
        if has_evaluation
        else preference_tensors
    )
    evaluation_label = (
        "Phi-4-mini evaluation model" if has_evaluation else "selector (self-graded)"
    )
    prompts = anchors_dataframe["prompt"].tolist()
    anchors = dict(
        zip(anchors_dataframe["prompt"], anchors_dataframe["anchor"], strict=True)
    )
    pools = {
        prompt: group.sort_values("sample_index")["response"].tolist()[
            :SAMPLES_PER_PROMPT
        ]
        for prompt, group in candidates_dataframe.groupby("prompt")
    }
    pool_tokens = {
        prompt: group.sort_values("sample_index")["tokens"].tolist()[
            :SAMPLES_PER_PROMPT
        ]
        for prompt, group in candidates_dataframe.groupby("prompt")
    }
    # Per-prompt tensors index rows of anchors.parquet, order their heads as
    # HEADS and carry the anchor as their last row and column; a prompt-order
    # mismatch would silently score every prompt against another prompt's
    # tensors
    for tensors in (preference_tensors, evaluation_tensors):
        assert tensors["criteria"].tolist() == HEADS
        assert tensors["prompts"].tolist() == prompts
        assert prompt_tensor(tensors, 0).shape == (
            len(HEADS),
            SAMPLES_PER_PROMPT + 1,
            SAMPLES_PER_PROMPT + 1,
        )
    mo.md(f"{len(prompts)} prompts, metrics graded by the {evaluation_label}")
    return (
        anchors,
        evaluation_label,
        evaluation_tensors,
        pool_tokens,
        pools,
        preference_tensors,
        prefix,
        prompts,
        suffix,
    )


@app.cell
def _():
    solve_button = mo.ui.run_button(label="Recompute policies and upload selections")
    solve_button
    return (solve_button,)


@app.cell
def _(
    pool_tokens,
    pools,
    preference_tensors,
    prefix,
    prompts,
    solve_button,
    suffix,
):
    # Hub selections reproduce the solved policies exactly, so opening the
    # notebook shows results without the solve; the button forces a re-solve
    selections_file = f"selections{suffix}.parquet"
    on_hub = file_exists(ARTIFACTS_REPO, f"{prefix}/{selections_file}", repo_type="dataset")
    if solve_button.value or not on_hub:
        mo.stop(
            not solve_button.value,
            mo.md("No selections on the hub for this backbone and scorer yet; press the button."),
        )
        policies = solve_policies(preference_tensors, prompts, pool_tokens)
        selections_dataframe = selections_frame(policies, pools)
        upload_dataframe(selections_file, selections_dataframe, prefix)
    else:
        selections_dataframe = pd.read_parquet(artifact_path(selections_file, prefix))
        policies = policies_from_selections(selections_dataframe)
    selections_dataframe
    return policies, selections_dataframe


@app.cell
def _(evaluation_tensors, policies, prompts):
    win_rates_dataframe = anchor_win_rates(policies, evaluation_tensors, prompts)
    win_rates_dataframe.pivot(index=["method", "n"], columns="criterion", values="win_rate").round(3)
    return


@app.cell
def _():
    judge_button = mo.ui.run_button(label="Re-judge with Claude")
    judge_button
    return (judge_button,)


@app.cell
def _(anchors, judge_button, prefix, selections_dataframe, suffix):
    # Hub judge scores stand in unless the button forces a re-judge; scores
    # for arms missing from a stale file simply leave those arms unjudged
    judge_file = f"judge_scores{suffix}.parquet"
    judged_on_hub = file_exists(ARTIFACTS_REPO, f"{prefix}/{judge_file}", repo_type="dataset")
    if not judge_button.value and judged_on_hub:
        judge_scores_dataframe = pd.read_parquet(artifact_path(judge_file, prefix))
        mo.stop(True, judge_scores_dataframe)
    mo.stop(
        not judge_button.value,
        mo.md("No judge scores on the hub for this backbone and scorer yet; press the button."),
    )
    # Expectation scoring: judge each distinct support atom once against the
    # anchor, then average atom scores under each policy's weights. The hub's
    # atom cache is upserted after every chunk, so atoms already judged by
    # this model in an earlier run or under the other scorer are not
    # re-judged and an outage loses at most one chunk. The entropic policy
    # has full support (~n calls per prompt) and is scored on the tensor
    # metrics only
    judged_selections = selections_dataframe[
        selections_dataframe["method"] != "entropic_blackwell"
    ]
    atoms = judged_selections[["prompt", "response"]].drop_duplicates()
    comparisons = [
        {"instruction": prompt, "response": response, "anchor": anchors[prompt]}
        for prompt, response in zip(atoms["prompt"], atoms["response"], strict=True)
    ]
    atom_cache = judge_atoms(
        comparisons,
        load_atom_judgements(prefix),
        checkpoint=lambda cache: upload_dataframe("atom_judgements.parquet", cache, prefix),
    )
    judged_atoms = atom_cache[atom_cache["model"] == JUDGE_MODEL]
    atom_scores = dict(
        zip(
            zip(judged_atoms["instruction"], judged_atoms["response"], strict=True),
            judged_atoms["score"],
            strict=True,
        )
    )
    judge_scores_dataframe = expected_scores(judged_selections, atom_scores)
    upload_dataframe(judge_file, judge_scores_dataframe, prefix)
    judge_scores_dataframe
    return (judge_scores_dataframe,)


@app.cell
def _(
    evaluation_label,
    evaluation_tensors,
    judge_scores_dataframe,
    policies,
    pool_tokens,
    prefix,
    prompts,
    suffix,
):
    # Per-prompt table behind every chart: per-head win rates vs the anchor
    # (evaluation tensors), expected tokens and the judged score
    results_dataframe = (
        prompt_results(policies, evaluation_tensors, prompts, pool_tokens)
        .merge(judge_scores_dataframe, on=["prompt", "method", "n"], how="left")
        .assign(grader=evaluation_label)
    )
    upload_dataframe(f"results{suffix}.parquet", results_dataframe, prefix)
    summary_dataframe = summarise(results_dataframe)
    summary_dataframe.pivot(index=["method", "n"], columns="metric", values="mean").round(3)
    return


@app.cell
def _():
    charts_button = mo.ui.run_button(label="Show charts")
    charts_button
    return (charts_button,)


@app.cell
def _(charts_button):
    mo.stop(not charts_button.value)
    # The story charts read every results table on the hub, one row per
    # backbone with pairwise solid and BT dashed, so they do not depend on
    # the dropdowns; the summary above is what this backbone and scorer
    # contribute
    hub_tables = hub_results(BACKBONES)
    hub_summaries = {key: summarise(table) for key, table in hub_tables.items()}
    hub_grader = grader_of(next(iter(hub_tables.values()))) if hub_tables else ""
    list(hub_summaries)
    return hub_grader, hub_summaries


@app.cell
def _(hub_summaries):
    chart_verbosity(hub_summaries)
    return


@app.cell
def _(hub_grader, hub_summaries):
    chart_quality(hub_summaries, hub_grader)
    return


@app.cell
def _(hub_summaries):
    chart_length(hub_summaries)
    return


@app.cell
def _(hub_grader, hub_summaries):
    mo.vstack(
        [
            chart_criteria(hub_summaries, hub_grader, criteria_backbone)
            for criteria_backbone in backbone_rows(hub_summaries)
        ]
    )
    return


if __name__ == "__main__":
    app.run()
