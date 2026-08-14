"""Tests for the winner policy solvers."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import linprog

from blackwell_ita.winners import (
    best_of_nash,
    blackwell_winner,
    bt_best_of_n,
    bt_preference_tensor,
    expected_scores,
    mean_criterion_best_of_n,
    mean_win_rates,
    policy_support,
    preference_tensor,
    reward_scores,
    worst_criterion_best_of_n,
)

CYCLE = np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
DOMINANT = np.array([[0.5, 0.9, 0.9], [0.1, 0.5, 0.5], [0.1, 0.5, 0.5]])


def worst_case_win(policy: np.ndarray, preference_matrix: np.ndarray) -> float:
    """Worst-case win probability of a policy against pure opponents."""
    return (preference_matrix.T @ policy).min()


def test_best_of_nash_cycle_is_uniform():
    """A three-cycle has the uniform policy as its equilibrium."""
    np.testing.assert_allclose(best_of_nash(CYCLE), np.full(3, 1 / 3), atol=1e-6)


def test_best_of_nash_dominant_object():
    """A dominant object receives all the probability mass."""
    np.testing.assert_allclose(best_of_nash(DOMINANT), [1.0, 0.0, 0.0], atol=1e-6)


def test_best_of_nash_non_uniform_mixture_with_dominated_object():
    """A known non-uniform equilibrium pins the payoff orientation."""
    # Weighted rock-paper-scissors plus a dominated fourth object: the unique
    # equilibrium is (1/3, 1/2, 1/6, 0); a transposed payoff matrix instead
    # concentrates on the dominated object, so this pins the orientation
    skew = np.array([[0.0, 0.1, -0.3], [-0.1, 0.0, 0.2], [0.3, -0.2, 0.0]])
    matrix = np.full((4, 4), 0.5)
    matrix[:3, :3] += skew
    matrix[:3, 3] = 0.85
    matrix[3, :3] = 0.15
    np.testing.assert_allclose(
        best_of_nash(matrix), [1 / 3, 1 / 2, 1 / 6, 0.0], atol=1e-6
    )


def test_blackwell_single_criterion_is_a_nash_equilibrium():
    """With one criterion the Blackwell winner matches the Nash worst case."""
    for matrix in [CYCLE, DOMINANT]:
        policy = blackwell_winner(matrix[None])
        np.testing.assert_allclose(
            worst_case_win(policy, matrix),
            worst_case_win(best_of_nash(matrix), matrix),
            atol=1e-6,
        )


def test_blackwell_balances_conflicting_criteria():
    """Two opposed criteria yield the balanced 50/50 policy."""
    first_criterion = np.array([[0.5, 1.0], [0.0, 0.5]])
    second_criterion = np.array([[0.5, 0.0], [1.0, 0.5]])
    policy = blackwell_winner(np.stack([first_criterion, second_criterion]))
    np.testing.assert_allclose(policy, [0.5, 0.5], atol=1e-6)


def blackwell_objective(policy: np.ndarray, tensor: np.ndarray) -> float:
    """Worst clipped shortfall of a policy, straight from the definition."""
    shortfalls = 0.5 - np.einsum("m,jmi->ji", policy, tensor)
    return float(np.clip(shortfalls, 0.0, None).max())


def test_blackwell_unique_minimiser_pins_transposes_and_clip():
    """A unique known minimiser guards against transpose and clipping bugs."""
    # Unique minimiser (4/7, 3/7) with value 12/70; a transposed constraint
    # matrix or an unclipped objective lands elsewhere
    tensor = np.stack(
        [
            np.array([[0.5, 0.9], [0.1, 0.5]]),
            np.array([[0.5, 0.2], [0.8, 0.5]]),
        ]
    )
    policy = blackwell_winner(tensor)
    np.testing.assert_allclose(policy, [4 / 7, 3 / 7], atol=1e-6)
    np.testing.assert_allclose(blackwell_objective(policy, tensor), 12 / 70, atol=1e-6)


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
    # Equal solution sets, checked both ways on random skew-symmetric games:
    # the k=1 Blackwell winner beats or ties every pure opponent (the defining
    # von Neumann property), and every von Neumann winner has zero Blackwell
    # shortfall. Equilibria need not be unique, so set membership — not
    # identical vertices — is the exact statement of "100% the same"
    rng = np.random.default_rng(1810)
    for _ in range(20):
        raw = rng.uniform(0.0, 1.0, size=(5, 5))
        matrix = (raw + 1.0 - raw.T) / 2.0
        blackwell_policy = blackwell_winner(matrix[None])
        von_neumann_policy, game_value = independent_von_neumann(matrix)
        np.testing.assert_allclose(game_value, 0.5, atol=1e-9)
        assert worst_case_win(blackwell_policy, matrix) >= 0.5 - 1e-9
        assert blackwell_objective(von_neumann_policy, matrix[None]) <= 1e-9


def test_blackwell_winner_raises_on_lp_failure():
    """A solver failure surfaces as RuntimeError rather than returning None."""
    # NaN inputs are rejected earlier by scipy itself; overflow-scale values
    # reach the solver and make it report failure
    with pytest.raises(RuntimeError, match="linprog failed"):
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


def test_policy_support_drops_dust_and_renormalises():
    """Numerical dust is dropped and the remaining weights sum to one."""
    atoms = policy_support(np.array([0.5, -1e-9, 3e-7, 0.5]))
    assert [index for index, _ in atoms] == [0, 3]
    np.testing.assert_allclose([weight for _, weight in atoms], [0.5, 0.5])
    assert policy_support(np.array([0.0, 1.0])) == [(1, 1.0)]


def test_bt_best_of_n_is_argmax_one_hot():
    """Best-of-n puts all mass on the highest reward."""
    np.testing.assert_array_equal(
        bt_best_of_n(np.array([0.1, 0.7, 0.3])), [0.0, 1.0, 0.0]
    )


# Response 0 dominates criterion 0 but collapses on criterion 1 (mean win
# rates 0.9 and 0.2), response 1 is balanced (0.45 and 0.5), response 2
# mirrors response 0 (0.15 and 0.8) — so mean- and worst-criterion
# aggregation provably pick different responses on the same tensor
CONFLICT = np.stack(
    [
        np.array([[0.5, 0.85, 0.95], [0.15, 0.5, 0.75], [0.05, 0.25, 0.5]]),
        np.array([[0.5, 0.3, 0.1], [0.7, 0.5, 0.3], [0.9, 0.7, 0.5]]),
    ]
)


def test_mean_win_rates_exclude_self_comparison():
    """Row means come from off-diagonal entries only."""
    np.testing.assert_allclose(
        mean_win_rates(CONFLICT),
        [[0.9, 0.45, 0.15], [0.2, 0.5, 0.8]],
    )


def test_mean_and_worst_criterion_best_of_n_disagree_under_conflict():
    """Conflicting criteria split the mean and worst-criterion argmax."""
    np.testing.assert_array_equal(mean_criterion_best_of_n(CONFLICT), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(worst_criterion_best_of_n(CONFLICT), [0.0, 1.0, 0.0])


def test_all_methods_coincide_on_single_candidate():
    """Every method returns the one-hot [1.0] on a lone candidate."""
    single = np.full((2, 1, 1), 0.5)
    np.testing.assert_allclose(blackwell_winner(single), [1.0], atol=1e-6)
    np.testing.assert_allclose(best_of_nash(single[0]), [1.0], atol=1e-6)
    np.testing.assert_array_equal(bt_best_of_n(np.array([0.7])), [1.0])
    np.testing.assert_array_equal(mean_criterion_best_of_n(single), [1.0])
    np.testing.assert_array_equal(worst_criterion_best_of_n(single), [1.0])


class StubPairwiseModel:
    """Stand-in pairwise model scoring texts deterministically by length."""

    def __init__(self, head_count: int) -> None:
        """Store the number of criterion heads."""
        self.head_count = head_count

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        """Return one length-derived logit per head for each text."""
        return torch.stack(
            [
                torch.tensor([float(len(text) % 7) - 3.0] * self.head_count)
                for text in texts
            ]
        )


def test_preference_tensor_is_skew_symmetric():
    """The preference tensor is skew-symmetric with 0.5 diagonals."""
    tensor = preference_tensor(
        StubPairwiseModel(head_count=2), "prompt", ["aa", "bbb", "c"], device="cpu"
    )
    assert tensor.shape == (2, 3, 3)
    np.testing.assert_allclose(tensor + tensor.transpose(0, 2, 1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.diagonal(tensor, axis1=1, axis2=2), 0.5)


def test_preference_tensor_batching_matches_single_batch():
    """Batched scoring gives the same tensor as one big batch."""
    model = StubPairwiseModel(head_count=2)
    responses = ["aa", "bbb", "c", "dddd"]
    batched = preference_tensor(model, "prompt", responses, "cpu", batch_size=2)
    single = preference_tensor(model, "prompt", responses, "cpu", batch_size=64)
    np.testing.assert_allclose(batched, single)


def test_bt_preference_tensor_is_skew_symmetric():
    """The BT-implied tensor is skew-symmetric with 0.5 diagonals."""
    rng = np.random.default_rng(1810)
    rewards = rng.uniform(-3.0, 3.0, size=(4, 2))
    tensor = bt_preference_tensor(rewards)
    assert tensor.shape == (2, 4, 4)
    np.testing.assert_allclose(tensor + tensor.transpose(0, 2, 1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.diagonal(tensor, axis1=1, axis2=2), 0.5)


def test_bt_preference_tensor_is_invariant_to_per_head_offsets():
    """Shifting one head's rewards by a constant leaves its tensor unchanged."""
    rng = np.random.default_rng(1810)
    rewards = rng.uniform(-3.0, 3.0, size=(4, 2))
    shifted = rewards.copy()
    shifted[:, 1] += 17.0
    np.testing.assert_allclose(
        bt_preference_tensor(shifted), bt_preference_tensor(rewards), atol=1e-12
    )


def test_bt_preference_tensor_preserves_single_head_argmax():
    """The mean-win-rate argmax matches the raw reward argmax per head."""
    rng = np.random.default_rng(1810)
    rewards = rng.uniform(-3.0, 3.0, size=(5, 1))
    win_rates = mean_win_rates(bt_preference_tensor(rewards))
    assert int(np.argmax(win_rates[0])) == int(np.argmax(rewards[:, 0]))


class StubRewardModel:
    """Stand-in reward model scoring texts deterministically by length."""

    def score(self, texts: list[str], device: str) -> torch.Tensor:
        """Return two length-derived rewards per text."""
        return torch.tensor([[float(len(text)), -float(len(text))] for text in texts])


def test_reward_scores_batches_all_responses():
    """Reward scoring covers every response across batches."""
    responses = ["a", "bb", "ccc", "dddd", "eeeee"]
    rewards = reward_scores(StubRewardModel(), "prompt", responses, "cpu", batch_size=2)
    assert rewards.shape == (5, 2)
    expected_first_head = [len(f"prompt\n\n{response}") for response in responses]
    np.testing.assert_allclose(rewards[:, 0], expected_first_head)
    np.testing.assert_allclose(rewards[:, 1], -rewards[:, 0])


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
    assert len(actual) == 4
    expected = {
        ("p", "blackwell", 4): 0.25,
        ("p", "base", 4): 1.0,
        ("p", "blackwell", 8): 0.5,
        ("q", "base", 4): 0.5,
    }
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value)
