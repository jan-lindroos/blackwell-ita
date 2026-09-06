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
    TAUS = [round(0.5 + 0.05 * step, 2) for step in range(10)]
    # Length-head scales as multiples of the pool's mean length: small is a
    # strong preference for concision, large is near indifference
    SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
    # Verbosity is a descriptive rating, not a quality criterion (it agrees
    # with helpfulness on 54% of decisive pairs, a coin flip), so the reported
    # welfare and worst-criterion metrics run over the four quality heads;
    # the 5-head arm stays as the demonstration that chasing it fails
    WELFARE_HEADS = [HEADS[head] for head in NO_VERBOSITY_HEADS]
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

    Tau-sweep arms add the overall head to the four no-verbosity criteria and
    carry its threshold after an @ in the method name; tokens arms stack the
    length head onto the criteria as one more head at threshold 1/2 and carry
    its scale (a multiple of the pool's mean length) after the @.
    """
    overall_heads = NO_VERBOSITY_HEADS + [OVERALL_INDEX]
    policies = {}
    for prompt_index, prompt in enumerate(
        mo.status.progress_bar(prompts, title="solving")
    ):
        tensor = prompt_tensor(tensors, prompt_index)
        tokens = np.asarray(pool_tokens[prompt], dtype=float)
        policies[(prompt, "base", 1)] = np.array([1.0])
        for n in N_VALUES:
            mean_length = float(tokens[:n].mean())
            length = length_preference(tokens[:n], mean_length)[None]
            policies[(prompt, "best_of_blackwell", n)] = blackwell_winner(
                tensor[:OVERALL_INDEX, :n, :n]
            )
            policies[(prompt, "blackwell_no_verbosity", n)] = blackwell_winner(
                tensor[NO_VERBOSITY_HEADS, :n, :n]
            )
            policies[(prompt, "entropic_blackwell", n)] = blackwell_winner(
                tensor[:OVERALL_INDEX, :n, :n], beta=BETA
            )
            policies[(prompt, "best_of_nash", n)] = best_of_nash(
                tensor[OVERALL_INDEX, :n, :n]
            )
            for scale in SCALES:
                policies[(prompt, f"blackwell_no_verbosity_tokens@{scale:g}", n)] = (
                    blackwell_winner(
                        np.concatenate(
                            [
                                tensor[NO_VERBOSITY_HEADS, :n, :n],
                                length_preference(tokens[:n], scale * mean_length)[None],
                            ]
                        )
                    )
                )
            for tau in TAUS:
                thresholds = [0.5] * len(NO_VERBOSITY_HEADS) + [tau]
                policies[(prompt, f"blackwell_no_verbosity_overall@{tau:.2f}", n)] = (
                    blackwell_winner(tensor[overall_heads, :n, :n], thresholds)
                )
                policies[
                    (prompt, f"blackwell_no_verbosity_overall_tokens@{tau:.2f}", n)
                ] = blackwell_winner(
                    np.concatenate([tensor[overall_heads, :n, :n], length]),
                    thresholds + [0.5],
                )
    return policies


@app.function
def frontier_methods(win_rates: pd.DataFrame) -> list[str]:
    """Methods on the frontier chart.

    The fixed arms, the whole length-scale sweep, each tau-sweep family at
    tau = 0.5, and each tau family's best tau by the tensor's own overall win
    rate against the anchor at the largest N, so no judge calls are spent
    selecting tau.
    """
    methods = win_rates["method"]
    scales = sorted(
        set(methods[methods.str.startswith("blackwell_no_verbosity_tokens@")]),
        key=lambda method: float(method.split("@")[1]),
    )
    sweep = win_rates[
        methods.str.contains("overall@")
        & (win_rates["criterion"] == "overall")
        & (win_rates["n"] == N_VALUES[-1])
    ]
    families = sweep["method"].str.split("@").str[0]  # pyright: ignore[reportAttributeAccessIssue]
    best = sweep.loc[sweep.groupby(families)["win_rate"].idxmax(), "method"]
    return list(
        dict.fromkeys(
            [
                "base",
                "best_of_nash",
                "best_of_blackwell",
                "blackwell_no_verbosity",
                *scales,
                "blackwell_no_verbosity_overall@0.50",
                "blackwell_no_verbosity_overall_tokens@0.50",
                *best.tolist(),
            ]
        )
    )


@app.function
def welfare_frame(win_rates: pd.DataFrame) -> pd.DataFrame:
    """Rawlsian (minimum) and Nash (geometric mean) welfare per (method, n).

    Both aggregate the mean criterion win rates against the anchor over the
    four quality heads (WELFARE_HEADS); verbosity and overall are left out.
    """
    criteria = win_rates[win_rates["criterion"].isin(WELFARE_HEADS)]
    grouped = criteria.groupby(["method", "n"])["win_rate"]
    return pd.DataFrame(
        {
            "rawlsian": grouped.min(),
            "nash": grouped.agg(lambda rates: float(np.exp(np.log(rates).mean()))),
        }
    ).reset_index()


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
def anchor_win_rates(policies: dict, tensors, prompts: list[str]) -> pd.DataFrame:
    """Mean per-head win rate of each policy against the anchor.

    The tensors carry the anchor as their last row and column, so a support
    atom's per-head rate is read straight off its prompt's tensor rather
    than re-scored with the model that produced it.
    """
    prompt_indices = {prompt: index for index, prompt in enumerate(prompts)}
    rows = [
        {"method": method, "n": n, "criterion": criterion, "win_rate": float(rate)}
        for (prompt, method, n), policy in policies.items()
        for criterion, rate in zip(
            HEADS,
            # A support is never empty, so sum() cannot fall through to its
            # integer start value
            sum(
                weight * prompt_tensor(tensors, prompt_indices[prompt])[:, index, -1]
                for index, weight in policy_support(policy)
            ),  # pyright: ignore[reportArgumentType]
            strict=True,
        )
    ]
    return (  # pyright: ignore[reportReturnType]
        pd.DataFrame(rows)
        .groupby(["method", "n", "criterion"], as_index=False)["win_rate"]
        .mean()
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


@app.function
def load_atom_judgements(prefix: str) -> pd.DataFrame:
    """A backbone's atom judgement cache from the hub, empty if none exists."""
    try:
        return pd.read_parquet(artifact_path("atom_judgements.parquet", prefix))
    except EntryNotFoundError:
        return pd.DataFrame(columns=ATOM_COLUMNS)


@app.function
def judge_atoms(
    comparisons: list[dict], cache: pd.DataFrame, model: str = JUDGE_MODEL
) -> pd.DataFrame:
    """Judge the comparisons the cache lacks for this model; return the merged cache.

    Atoms are keyed on (instruction, response, model): the anchor is fixed
    per instruction, so a judgement carries over between scorers and runs.
    """
    seen = set(zip(cache["instruction"], cache["response"], cache["model"], strict=True))
    fresh = [
        comparison
        for comparison in comparisons
        if (comparison["instruction"], comparison["response"], model) not in seen
    ]
    rows = [
        {
            "instruction": comparison["instruction"],
            "response": comparison["response"],
            "model": model,
            **judgement,
        }
        for comparison, judgement in zip(
            fresh, outcomes(fresh, model=model), strict=True
        )
    ]
    return pd.concat(
        [cache, pd.DataFrame(rows, columns=ATOM_COLUMNS)], ignore_index=True
    )


@app.cell
def _():
    base_model_dropdown = mo.ui.dropdown(
        options=[
            DEFAULT_BASE_MODEL,
            "google/gemma-2b-it",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
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
    solve_button = mo.ui.run_button(label="Compute policies and upload selections")
    solve_button
    return (solve_button,)


@app.cell
def _(
    pool_tokens, preference_tensors, pools, prefix, prompts, solve_button, suffix
):
    mo.stop(not solve_button.value)
    policies = solve_policies(preference_tensors, prompts, pool_tokens)
    selections_dataframe = selections_frame(policies, pools)
    upload_dataframe(f"selections{suffix}.parquet", selections_dataframe, prefix)
    selections_dataframe
    return policies, selections_dataframe


@app.cell
def _(evaluation_tensors, policies, prompts):
    win_rates_dataframe = anchor_win_rates(policies, evaluation_tensors, prompts)
    win_rates_dataframe
    return (win_rates_dataframe,)


@app.cell
def _(policies, preference_tensors, prompts):
    # Tau is chosen on the selector's own tensors, so the evaluation model
    # never touches selection
    frontier_arms = frontier_methods(
        anchor_win_rates(policies, preference_tensors, prompts)
    )
    frontier_arms
    return (frontier_arms,)


@app.cell
def _(win_rates_dataframe):
    worst_rates = (
        win_rates_dataframe[win_rates_dataframe["criterion"].isin(WELFARE_HEADS)]
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
    for line_method in (
        "best_of_nash",
        "best_of_blackwell",
        "blackwell_no_verbosity",
        "entropic_blackwell",
    ):
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
    return (worst_rates,)


@app.cell
def _(policies, pool_tokens, worst_rates):
    # Token efficiency: exact counts from the base tokenizer, no annotation.
    # A method whose expected tokens climb with N while its win rate does not
    # is chasing the raw verbosity head rather than quality
    efficiency_dataframe = worst_rates.merge(
        expected_token_counts(policies, pool_tokens), on=["method", "n"]
    )
    efficiency_dataframe["win_rate_per_ktoken"] = efficiency_dataframe["win_rate"] / (
        efficiency_dataframe["tokens"] / 1000.0
    )
    efficiency_dataframe
    return (efficiency_dataframe,)


@app.cell
def _(efficiency_dataframe):
    tokens_figure, tokens_axes = plt.subplots(figsize=(5, 3.2))
    tokens_axes.axhline(
        efficiency_dataframe.loc[
            efficiency_dataframe["method"] == "base", "tokens"
        ].item(),
        color="grey",
        linestyle=":",
        linewidth=1,
        label="base",
    )
    for tokens_method in (
        "best_of_nash",
        "best_of_blackwell",
        "blackwell_no_verbosity",
        "entropic_blackwell",
    ):
        tokens_stats = efficiency_dataframe[
            efficiency_dataframe["method"] == tokens_method
        ].sort_values("n")
        tokens_axes.plot(
            tokens_stats["n"], tokens_stats["tokens"], marker="o", label=tokens_method
        )
    tokens_axes.set_xscale("log", base=2)
    tokens_axes.set_xlabel("N (pool size)")
    tokens_axes.set_ylabel("Expected response tokens")
    tokens_axes.legend(frameon=False)
    tokens_axes.spines[["top", "right"]].set_visible(False)
    tokens_figure
    return


@app.cell
def _():
    judge_button = mo.ui.run_button(label="Judge with Claude")
    judge_button
    return (judge_button,)


@app.cell
def _(anchors, frontier_arms, judge_button, prefix, selections_dataframe, suffix):
    mo.stop(not judge_button.value)
    # Expectation scoring: judge each distinct support atom once against the
    # anchor, then average atom scores under each policy's weights. The hub's
    # atom cache is upserted, so atoms already judged by this model in an
    # earlier run or under the other scorer are not re-judged. Only the
    # frontier arms are judged: the entropic policy has full support, so
    # judging it would cost ~n calls per prompt, and the off-chart tau
    # variants would multiply the atom count; both are scored on the tensor
    # metric only
    judged_selections = selections_dataframe[
        selections_dataframe["method"].isin(frontier_arms)
    ]
    atoms = judged_selections[["prompt", "response"]].drop_duplicates()
    comparisons = [
        {"instruction": prompt, "response": response, "anchor": anchors[prompt]}
        for prompt, response in zip(atoms["prompt"], atoms["response"], strict=True)
    ]
    atom_cache = judge_atoms(comparisons, load_atom_judgements(prefix))
    upload_dataframe("atom_judgements.parquet", atom_cache, prefix)
    judged_atoms = atom_cache[atom_cache["model"] == JUDGE_MODEL]
    atom_scores = dict(
        zip(
            zip(judged_atoms["instruction"], judged_atoms["response"], strict=True),
            judged_atoms["score"],
            strict=True,
        )
    )
    judge_scores_dataframe = expected_scores(judged_selections, atom_scores)
    upload_dataframe(f"judge_scores{suffix}.parquet", judge_scores_dataframe, prefix)
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
def _(judge_summary):
    judge_figure, judge_axes = plt.subplots(figsize=(5, 3.2))
    judge_axes.axhline(
        judge_summary.loc[judge_summary["method"] == "base", "score"].item(),
        color="grey",
        linestyle=":",
        linewidth=1,
        label="base",
    )
    for curve_method in ("best_of_nash", "best_of_blackwell", "blackwell_no_verbosity"):
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


@app.function
def frontier_figure(frontier: pd.DataFrame, arms: list[str], grader: str):
    """Judged overall against Rawlsian welfare, Nash welfare and expected tokens.

    One colour per criterion ablation, traced over N with markers growing
    with N; a tau family's best tau is dashed next to its solid tau = 0.5
    line; the length-scale sweep draws only scale 1 over N and the other
    scales as a dotted trace at the largest N.
    """
    colours = {
        "best_of_blackwell": "#2a78d6",
        "blackwell_no_verbosity": "#eb6834",
        "blackwell_no_verbosity_overall": "#1baf7a",
        "blackwell_no_verbosity_tokens": "#eda100",
        "blackwell_no_verbosity_overall_tokens": "#e87ba4",
        "best_of_nash": "#8a8983",
        "base": "#8a8983",
    }
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True
    )
    panels = (
        ("rawlsian", "Rawlsian welfare (worst quality criterion)"),
        ("nash", "Nash welfare (geometric mean of quality criteria)"),
        ("tokens", "Expected response tokens"),
    )
    scale_arms = [
        arm
        for arm in arms
        if arm.startswith("blackwell_no_verbosity_tokens@")
        and float(arm.split("@")[1]) != 1.0
    ]
    for panel_axes, (column, label) in zip(axes, panels, strict=True):
        for arm in arms:
            if arm in scale_arms:
                continue
            stats = frontier[frontier["method"] == arm].sort_values("n")  # pyright: ignore[reportCallIssue]
            family, _, suffix = arm.partition("@")
            dashed = "overall" in family and suffix not in ("", "0.50")
            panel_axes.plot(
                stats[column],
                stats["overall"],
                color=colours[family],
                linestyle="--" if dashed else "-",
                linewidth=2,
                label=arm,
            )
            panel_axes.scatter(
                stats[column],
                stats["overall"],
                s=20 + 12 * np.log2(stats["n"]),
                color=colours[family],
                edgecolor="white",
                linewidth=1,
                zorder=3,
            )
        if scale_arms:
            trace = frontier[
                frontier["method"].isin(scale_arms + ["blackwell_no_verbosity_tokens@1"])
                & (frontier["n"] == N_VALUES[-1])
            ].copy()
            trace["scale"] = trace["method"].str.split("@").str[1].astype(float)  # pyright: ignore[reportAttributeAccessIssue]
            trace = trace.sort_values("scale")  # pyright: ignore[reportCallIssue]
            panel_axes.plot(
                trace[column],
                trace["overall"],
                color=colours["blackwell_no_verbosity_tokens"],
                linestyle=":",
                linewidth=1.5,
                marker="s",
                markersize=5,
                label=f"tokens scale sweep at N = {N_VALUES[-1]}",
            )
        panel_axes.set_xlabel(label)
        panel_axes.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Expected overall win rate vs anchor")
    figure.suptitle(f"Welfare graded by the {grader}", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return figure


@app.cell
def _(judge_summary, policies, pool_tokens, win_rates_dataframe):
    frontier_dataframe = (
        welfare_frame(win_rates_dataframe)
        .merge(expected_token_counts(policies, pool_tokens), on=["method", "n"])
        .merge(judge_summary.rename(columns={"score": "overall"}), on=["method", "n"])
    )
    frontier_dataframe.round(3)
    return (frontier_dataframe,)


@app.cell
def _(evaluation_label, frontier_arms, frontier_dataframe):
    frontier_figure(frontier_dataframe, frontier_arms, evaluation_label)
    return


if __name__ == "__main__":
    app.run()
