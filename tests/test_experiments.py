"""Tests for the experiments notebook logic."""

import json
from types import SimpleNamespace

import experiments
import numpy as np
import pandas as pd
import pytest
import torch
from experiments import (
    HEADS,
    anchor_preference_rates,
    best_of_nash,
    blackwell_winner,
    comparison_prompt,
    expected_scores,
    expected_token_counts,
    held_out_win_rates,
    pairwise_text,
    policy_support,
    prompt_tensor,
)
from scipy.optimize import linprog

CYCLE = np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
DOMINANT = np.array([[0.5, 0.9, 0.9], [0.1, 0.5, 0.5], [0.1, 0.5, 0.5]])


def worst_case_win(policy: np.ndarray, preference_matrix: np.ndarray) -> float:
    """Worst-case win probability of a policy against pure opponents."""
    return (preference_matrix.T @ policy).min()


def test_blackwell_skew_symmetric_matrix_has_zero_shortfall():
    """On a skew-symmetric game the value is 1/2, so the shortfall is zero."""
    rng = np.random.default_rng(1810)
    raw = rng.uniform(0.0, 1.0, size=(5, 5))
    matrix = (raw + 1.0 - raw.T) / 2.0
    policy = blackwell_winner(matrix[None])
    assert worst_case_win(policy, matrix) >= 0.5 - 1e-8


def test_blackwell_cycle_yields_uniform_mixture():
    """A rock-paper-scissors cycle has the uniform policy as its winner."""
    np.testing.assert_allclose(
        blackwell_winner(CYCLE[None]), np.full(3, 1 / 3), atol=1e-6
    )


def test_blackwell_mixture_beats_pure_candidates():
    """The winner's worst case beats every pure candidate's worst case."""
    best_pure = max(
        worst_case_win(np.eye(3)[index], CYCLE) for index in range(3)
    )
    assert worst_case_win(blackwell_winner(CYCLE[None]), CYCLE) > best_pure


def test_blackwell_unique_minimiser_pins_transposes_and_clip():
    """A unique known minimiser guards against transpose and clipping bugs."""
    tensor = np.stack(
        [
            np.array([[0.5, 0.9], [0.1, 0.5]]),
            np.array([[0.5, 0.2], [0.8, 0.5]]),
        ]
    )
    np.testing.assert_allclose(blackwell_winner(tensor), [4 / 7, 3 / 7], atol=1e-6)


def test_blackwell_thresholds_tilt_the_balanced_policy():
    """Raising one head's threshold moves the policy towards that head."""
    tensor = np.stack(
        [
            np.array([[0.5, 1.0], [0.0, 0.5]]),
            np.array([[0.5, 0.0], [1.0, 0.5]]),
        ]
    )
    np.testing.assert_allclose(blackwell_winner(tensor), [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(
        blackwell_winner(tensor, thresholds=[0.5, 1.0]), [0.0, 1.0], atol=1e-6
    )


def test_best_of_nash_is_single_head_blackwell():
    """best_of_nash coincides with the Blackwell winner on one overall head."""
    for matrix in (CYCLE, DOMINANT):
        np.testing.assert_allclose(best_of_nash(matrix), blackwell_winner(matrix[None]))
    np.testing.assert_allclose(best_of_nash(DOMINANT), [1.0, 0.0, 0.0], atol=1e-6)


def blackwell_objective(policy: np.ndarray, tensor: np.ndarray) -> float:
    """Worst clipped shortfall of a policy, straight from the definition."""
    shortfalls = 0.5 - np.einsum("m,jmi->ji", policy, tensor)
    return float(np.clip(shortfalls, 0.0, None).max())


def independent_von_neumann(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Independent max-min LP returning the policy and game value."""
    # The classic max-min LP, implemented here so the equivalence test shares
    # no code with the Blackwell solver under test
    count = matrix.shape[0]
    result = linprog(
        c=[0.0] * count + [-1.0],
        A_ub=np.hstack([-matrix.T, np.ones((count, 1))]),
        b_ub=np.zeros(count),
        A_eq=[[1.0] * count + [0.0]],
        b_eq=[1.0],
        bounds=[(0.0, None)] * count + [(None, None)],
    )
    assert result.success
    return result.x[:count], result.x[count]


def test_blackwell_single_criterion_equals_von_neumann_winner():
    """The k=1 Blackwell and von Neumann solution sets coincide."""
    # Equilibria need not be unique, so set membership — not identical
    # vertices — is checked both ways on random skew-symmetric games
    rng = np.random.default_rng(1810)
    for _ in range(20):
        raw = rng.uniform(0.0, 1.0, size=(5, 5))
        matrix = (raw + 1.0 - raw.T) / 2.0
        blackwell_policy = blackwell_winner(matrix[None])
        von_neumann_policy, game_value = independent_von_neumann(matrix)
        # Interior-point solutions land within ~1e-8 of the optimum, unlike
        # the exact simplex vertices the 1e-9 tolerances here once assumed
        np.testing.assert_allclose(game_value, 0.5, atol=1e-6)
        assert worst_case_win(blackwell_policy, matrix) >= 0.5 - 1e-6
        assert blackwell_objective(von_neumann_policy, matrix[None]) <= 1e-6


def test_blackwell_winner_raises_on_solver_failure():
    """A solver failure surfaces as RuntimeError rather than returning None."""
    # Overflow-scale values make the solve fail rather than return a policy
    with pytest.raises(RuntimeError, match="solve failed"):
        blackwell_winner(np.full((1, 2, 2), 1e300))


def test_blackwell_matches_definition_by_grid_search():
    """The solver's objective value matches a fine grid search."""
    rng = np.random.default_rng(1810)
    raw = rng.uniform(0.0, 1.0, size=(3, 2, 2))
    tensor = (raw + 1.0 - raw.transpose(0, 2, 1)) / 2.0
    solver_value = blackwell_objective(blackwell_winner(tensor), tensor)
    grid_values = [
        blackwell_objective(np.array([weight, 1.0 - weight]), tensor)
        for weight in np.linspace(0.0, 1.0, 2001)
    ]
    np.testing.assert_allclose(solver_value, min(grid_values), atol=1e-3)


def test_entropic_blackwell_keeps_cycle_uniform():
    """The uniform policy stays optimal on a cycle at every regularisation."""
    for beta in (0.01, 0.05, 0.5):
        np.testing.assert_allclose(
            blackwell_winner(CYCLE[None], beta=beta), np.full(3, 1 / 3), atol=1e-6
        )


def test_entropic_blackwell_has_full_support_and_bounded_excess():
    """The entropic policy has full support and value within beta log n."""
    rng = np.random.default_rng(1810)
    raw = rng.uniform(0.0, 1.0, size=(2, 6, 6))
    tensor = (raw + 1.0 - raw.transpose(0, 2, 1)) / 2.0
    beta = 0.05
    entropic = blackwell_winner(tensor, beta=beta)
    assert entropic.min() > 0.0
    exact_value = blackwell_objective(blackwell_winner(tensor), tensor)
    assert blackwell_objective(entropic, tensor) <= (
        exact_value + beta * np.log(6) + 1e-6
    )


def test_entropic_blackwell_gives_identical_candidates_equal_mass():
    """Strict convexity forces equal mass on duplicated candidates."""
    policy = blackwell_winner(DOMINANT[None], beta=0.05)
    assert policy[1] == pytest.approx(policy[2], abs=1e-6)


def test_entropic_blackwell_interpolates_between_lp_and_uniform():
    """Small beta approaches the LP vertex; large beta approaches uniform."""
    assert blackwell_winner(DOMINANT[None], beta=0.01)[0] > 0.9
    np.testing.assert_allclose(
        blackwell_winner(DOMINANT[None], beta=5.0), np.full(3, 1 / 3), atol=0.05
    )


def test_policy_support_drops_dust_and_renormalises():
    """Numerical dust is dropped and the remaining weights sum to one."""
    atoms = policy_support(np.array([0.5, -1e-9, 3e-7, 0.5]))
    assert [index for index, _ in atoms] == [0, 3]
    np.testing.assert_allclose([weight for _, weight in atoms], [0.5, 0.5])
    assert policy_support(np.array([0.0, 1.0])) == [(1, 1.0)]


def test_expected_scores_weight_averages_atoms_per_group():
    """Scores weight-average the atom scores within each (prompt, method, n)."""
    selections = pd.DataFrame(
        {
            "prompt": ["p", "p", "p", "p", "p", "q"],
            "method": ["blackwell", "blackwell", "base", "blackwell", "blackwell", "base"],
            "n": [4, 4, 4, 8, 8, 4],
            "weight": [0.25, 0.75, 1.0, 0.5, 0.5, 1.0],
            "response": ["r1", "r2", "r1", "r1", "r2", "r3"],
        }
    )
    atom_scores = {("p", "r1"): 1.0, ("p", "r2"): 0.0, ("q", "r3"): 0.5}
    scores = expected_scores(selections, atom_scores)
    assert list(scores.columns) == ["prompt", "method", "n", "score"]
    actual = {
        (row["prompt"], row["method"], row["n"]): row["score"]
        for _, row in scores.iterrows()
    }
    expected = {
        ("p", "blackwell", 4): 0.25,
        ("p", "base", 4): 1.0,
        ("p", "blackwell", 8): 0.5,
        ("q", "base", 4): 0.5,
    }
    assert len(actual) == len(expected)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value)


def test_expected_token_counts_weights_pool_prefix_and_averages_prompts():
    """Expected tokens weight the pool prefix and average over prompts."""
    policies = {
        ("p", "blackwell", 2): np.array([0.25, 0.75]),
        ("q", "blackwell", 2): np.array([1.0, 0.0]),
        ("p", "base", 1): np.array([1.0]),
    }
    pool_tokens = {"p": [100, 200, 999], "q": [40, 999, 999]}
    counts = expected_token_counts(policies, pool_tokens)
    actual = {
        (row["method"], row["n"]): row["tokens"] for _, row in counts.iterrows()
    }
    assert actual[("blackwell", 2)] == pytest.approx((175.0 + 40.0) / 2)
    assert actual[("base", 1)] == pytest.approx(100.0)


def test_prompt_tensor_reads_the_key_scheme():
    """The loader reads tensor_{i} keys and fails loudly on a missing prompt."""
    array = np.ones((6, 2, 2))
    np.testing.assert_array_equal(prompt_tensor({"tensor_3": array}, 3), array)
    with pytest.raises(KeyError):
        prompt_tensor({"tensor_1": array}, 0)


def stub_scorer(logits_by_text: dict[str, list[float]], scored_texts: list[str]):
    """Stand-in model returning fixed per-head logits for known pairwise texts."""

    def score(texts: list[str], device: str) -> torch.Tensor:
        scored_texts.extend(texts)
        return torch.tensor([logits_by_text[text] for text in texts])

    return SimpleNamespace(score=score)


def test_anchor_preference_rates_skew_symmetrises_both_orders():
    """Rates average the forward and complemented backward pass per head."""
    logit = float(np.log(3.0))
    logits_by_text = {
        pairwise_text("p", "r", "a"): [0.0, logit],
        pairwise_text("p", "a", "r"): [0.0, -logit],
    }
    rates = anchor_preference_rates(
        stub_scorer(logits_by_text, []), "p", ["r"], "a", "cpu"
    )
    np.testing.assert_allclose(rates, [[0.5, 0.75]], atol=1e-6)


def test_held_out_win_rates_scores_atoms_once_and_averages():
    """Distinct atoms score once; win rates weight supports and average prompts."""
    atom_rates = {("p", "p0"): 0.8, ("p", "p1"): 0.4, ("q", "q0"): 0.6}
    anchors = {"p": "ap", "q": "aq"}
    logits_by_text = {}
    for (prompt, response), rate in atom_rates.items():
        logit = float(np.log(rate / (1.0 - rate)))
        logits_by_text[pairwise_text(prompt, response, anchors[prompt])] = [
            logit
        ] * len(HEADS)
        logits_by_text[pairwise_text(prompt, anchors[prompt], response)] = [
            -logit
        ] * len(HEADS)
    policies = {
        ("p", "base", 1): np.array([1.0]),
        ("p", "blackwell", 2): np.array([0.5, 0.5]),
        ("q", "base", 1): np.array([1.0]),
    }
    scored_texts: list[str] = []
    frame = held_out_win_rates(
        policies,
        stub_scorer(logits_by_text, scored_texts),
        {"p": ["p0", "p1"], "q": ["q0"]},
        anchors,
        "cpu",
    )
    # Both p policies share atom p0, so only 3 atoms score, in both orders
    assert len(scored_texts) == 6
    assert len(scored_texts) == len(set(scored_texts))
    assert list(frame.columns) == ["method", "n", "criterion", "win_rate"]
    rates = {
        (row["method"], row["n"], row["criterion"]): row["win_rate"]
        for _, row in frame.iterrows()
    }
    for criterion in HEADS:
        assert rates[("base", 1, criterion)] == pytest.approx((0.8 + 0.6) / 2)
        assert rates[("blackwell", 2, criterion)] == pytest.approx(0.6)


def order_sensitive_pick(preferred: str):
    """Stand-in for claude_pick preferring whichever slot holds ``preferred``."""

    def pick(prompt: str, model: str) -> str:
        first_slot = prompt.index("First response:")
        second_slot = prompt.index("Second response:")
        position = prompt.index(preferred)
        return "FIRST" if first_slot < position < second_slot else "SECOND"

    return pick


def test_outcome_swaps_orders_and_maps_scores(monkeypatch):
    """A consistent winner scores 1 or 0 through the order-swapped prompts."""
    monkeypatch.setattr(
        experiments, "claude_pick", order_sensitive_pick("the response")
    )
    assert experiments.outcome("instruction", "the response", "the anchor")["score"] == 1.0
    monkeypatch.setattr(experiments, "claude_pick", order_sensitive_pick("the anchor"))
    assert experiments.outcome("instruction", "the response", "the anchor")["score"] == 0.0


def fixed_picks(replies):
    """Stand-in for claude_pick returning scripted replies in order."""
    iterator = iter(replies)
    return lambda prompt, model: next(iterator)


def test_outcome_draws_on_disagreement_or_parse_failure(monkeypatch):
    """Order-swap disagreement and unparseable verdicts both draw at 0.5."""
    monkeypatch.setattr(experiments, "claude_pick", fixed_picks(["FIRST", "FIRST"]))
    assert experiments.outcome("instruction", "response", "anchor")["score"] == 0.5
    monkeypatch.setattr(experiments, "claude_pick", fixed_picks([None, "SECOND"]))
    judgement = experiments.outcome("instruction", "response", "anchor")
    assert judgement["score"] == 0.5
    assert judgement["forward"] is None
    assert judgement["backward"] == "SECOND"


def fixed_run(result: str, returncode: int = 0):
    """Stand-in for subprocess.run returning a fixed CLI reply."""
    return lambda command, **kwargs: SimpleNamespace(
        stdout=json.dumps({"result": result}), stderr="", returncode=returncode
    )


def test_claude_pick_invokes_cli_and_parses_reply(monkeypatch):
    """The CLI is called with the prompt and model, and the reply is parsed."""
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return fixed_run(" first choice")(command, **kwargs)

    monkeypatch.setattr(experiments.subprocess, "run", fake_run)
    assert experiments.claude_pick("the prompt", "claude-sonnet-5") == "FIRST"
    assert captured["command"][0] == "claude"
    assert captured["command"][-1] == "the prompt"
    assert "claude-sonnet-5" in captured["command"]


def test_claude_pick_handles_second_and_unparseable_replies(monkeypatch):
    """SECOND replies parse; unparseable replies return None."""
    monkeypatch.setattr(experiments.subprocess, "run", fixed_run("Second."))
    assert experiments.claude_pick("prompt", "claude-sonnet-5") == "SECOND"
    monkeypatch.setattr(experiments.subprocess, "run", fixed_run("I cannot decide"))
    assert experiments.claude_pick("prompt", "claude-sonnet-5") is None


def test_claude_pick_retries_transient_failures(monkeypatch):
    """A failed CLI call is retried and the retry's reply is used."""
    calls = {"count": 0}

    def flaky_run(command, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(stdout="", stderr="boom", returncode=1)
        return fixed_run("FIRST")(command, **kwargs)

    monkeypatch.setattr(experiments.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(experiments.subprocess, "run", flaky_run)
    assert experiments.claude_pick("prompt", "claude-sonnet-5") == "FIRST"
    assert calls["count"] == 2


def test_claude_pick_raises_after_exhausting_attempts(monkeypatch):
    """Persistent CLI failures raise once the attempts run out."""
    calls = {"count": 0}

    def failing_run(command, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(experiments.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(experiments.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match=r"3 attempts.*boom"):
        experiments.claude_pick("prompt", "claude-sonnet-5")
    assert calls["count"] == 3


def test_outcomes_preserves_order_and_forwards_arguments(monkeypatch):
    """Concurrent judging keeps input order and forwards keyword arguments."""

    def fake_outcome(instruction, response, anchor, model=None):
        assert model == "haiku"
        return {"score": float(len(response))}

    monkeypatch.setattr(experiments, "outcome", fake_outcome)
    comparisons = [
        {"instruction": "i", "response": "r" * n, "anchor": "a"} for n in (3, 1, 2)
    ]
    scores = [
        judgement["score"]
        for judgement in experiments.outcomes(comparisons, model="haiku", workers=2)
    ]
    assert scores == [3.0, 1.0, 2.0]


def test_comparison_prompt_orders_sections():
    """The prompt presents instruction, first and second responses in order."""
    prompt = comparison_prompt("the instruction", "response A", "response B")
    assert (
        prompt.index("the instruction")
        < prompt.index("First response:")
        < prompt.index("response A")
        < prompt.index("Second response:")
        < prompt.index("response B")
    )
    assert "FIRST or SECOND" in prompt
