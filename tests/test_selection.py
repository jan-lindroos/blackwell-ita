import warnings

import cvxpy as cp
import numpy as np
import pytest

from blackwell_ita.selection import (
    blackwell_winner,
    budgeted_nash_winner,
    clean_policy,
    max_mean_win_rate,
    maximin,
    nash_shortfall,
    nash_target_winner,
    paired_bootstrap_interval,
    robust_nash_winner,
    scalarised_nash,
    target_set_winner,
    von_neumann_winner,
)


def skew_symmetric(
    upper_entries: dict[tuple[int, int], float], size: int
) -> np.ndarray:
    matrix = np.full((size, size), 0.5)
    for (row, column), value in upper_entries.items():
        matrix[row, column] = value
        matrix[column, row] = 1.0 - value
    return matrix


def random_preference_tensor(seed: int, head_count: int, size: int) -> np.ndarray:
    upper = np.triu(
        np.random.default_rng(seed).uniform(size=(head_count, size, size)), 1
    )
    lower = np.transpose(np.triu(1.0 - upper, 1), (0, 2, 1))
    return upper + lower + 0.5 * np.eye(size)


def column_values(policy: np.ndarray, preference_tensor: np.ndarray) -> np.ndarray:
    """Per-head win rates of ``policy`` against each pure opponent, (head, opponent)."""
    return np.einsum("i,kij->kj", policy, preference_tensor)


def orthant_distance(policy: np.ndarray, preference_tensor: np.ndarray) -> float:
    return max(0.0, float((0.5 - column_values(policy, preference_tensor)).max()))


def nash_distance(policy: np.ndarray, preference_tensor: np.ndarray) -> float:
    return float(nash_shortfall(column_values(policy, preference_tensor), 0.5).max())


def mean_distance(policy: np.ndarray, preference_tensor: np.ndarray) -> float:
    return max(
        0.0, float((0.5 - column_values(policy, preference_tensor).mean(axis=0)).max())
    )


def test_von_neumann_winner_of_rock_paper_scissors_is_uniform():
    rock_paper_scissors = skew_symmetric({(0, 1): 0.0, (0, 2): 1.0, (1, 2): 0.0}, 3)
    assert von_neumann_winner(rock_paper_scissors) == pytest.approx(
        np.full(3, 1 / 3), abs=1e-5
    )


def test_von_neumann_winner_picks_a_condorcet_winner():
    condorcet = skew_symmetric({(0, 1): 0.8, (0, 2): 0.7, (1, 2): 0.6}, 3)
    assert von_neumann_winner(condorcet) == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)


def test_blackwell_protects_the_minority_head_that_averaging_overrides():
    # Two heads prefer candidate 0 strongly, the third hates it; candidate 1
    # is everyone's acceptable compromise
    majority = skew_symmetric({(0, 1): 0.9, (0, 2): 0.9, (1, 2): 0.9}, 3)
    minority = skew_symmetric({(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.9}, 3)
    preference_tensor = np.stack([majority, majority, minority])
    assert scalarised_nash(preference_tensor).argmax() == 0
    blackwell_policy = blackwell_winner(preference_tensor)
    assert blackwell_policy.argmax() == 1
    assert (
        column_values(blackwell_policy, preference_tensor).mean(axis=1).min()
        > column_values(scalarised_nash(preference_tensor), preference_tensor)
        .mean(axis=1)
        .min()
    )


def test_max_mean_win_rate_and_maximin_pick_one_candidate_each():
    majority = skew_symmetric({(0, 1): 0.9, (0, 2): 0.9, (1, 2): 0.9}, 3)
    minority = skew_symmetric({(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.9}, 3)
    preference_tensor = np.stack([majority, majority, majority, minority])
    assert max_mean_win_rate(preference_tensor).tolist() == [1.0, 0.0, 0.0]
    assert maximin(preference_tensor).tolist() == [0.0, 1.0, 0.0]


def test_clean_policy_drops_dust_and_renormalises():
    assert clean_policy(np.array([0.5, 1e-8, -1e-9, 0.5])).tolist() == [
        0.5,
        0.0,
        0.0,
        0.5,
    ]


def test_paired_bootstrap_interval_brackets_the_mean():
    differences = np.random.default_rng(0).normal(0.1, 0.05, size=200)
    low, high = paired_bootstrap_interval(differences)
    assert low < differences.mean() < high
    assert low > 0


def test_nash_shortfall_matches_closed_forms():
    # One head: the plain shortfall below the threshold
    assert nash_shortfall(np.array([[0.2, 0.5, 0.9]]), 0.5) == pytest.approx(
        [0.3, 0.0, 0.0]
    )
    # Two heads: the root of (a + t)(b + t) = c^2
    a, b, c = 0.1, 0.6, 0.5
    root = (-(a + b) + np.sqrt((a - b) ** 2 + 4 * c**2)) / 2
    assert nash_shortfall(np.array([[a], [b]]), c) == pytest.approx([root])
    # A zero entry makes the geometric mean zero, so a shortfall remains
    assert nash_shortfall(np.array([[0.0], [1.0]]), 0.5)[0] > 0
    # Already inside the set, and the threshold-zero set is everything
    assert nash_shortfall(np.array([[0.4], [0.9]]), 0.5) == pytest.approx([0.0])
    assert nash_shortfall(np.array([[0.0], [0.0]]), 0.0) == pytest.approx([0.0])


def test_nash_target_winner_with_one_head_is_the_von_neumann_winner():
    rock_paper_scissors = skew_symmetric({(0, 1): 0.0, (0, 2): 1.0, (1, 2): 0.0}, 3)
    assert nash_target_winner(rock_paper_scissors[None]) == pytest.approx(
        np.full(3, 1 / 3), abs=1e-5
    )
    condorcet = skew_symmetric({(0, 1): 0.8, (0, 2): 0.7, (1, 2): 0.6}, 3)
    assert nash_target_winner(condorcet[None]) == pytest.approx(
        [1.0, 0.0, 0.0], abs=1e-5
    )
    for seed in range(5):
        preference_tensor = random_preference_tensor(seed, 1, 6)
        assert nash_distance(
            nash_target_winner(preference_tensor), preference_tensor
        ) == pytest.approx(
            orthant_distance(
                von_neumann_winner(preference_tensor[0]), preference_tensor
            ),
            abs=1e-5,
        )


def test_nash_target_winner_with_identical_heads_is_the_von_neumann_winner():
    preference_matrix = random_preference_tensor(7, 1, 6)[0]
    preference_tensor = np.stack([preference_matrix] * 4)
    assert nash_distance(
        nash_target_winner(preference_tensor), preference_tensor
    ) == pytest.approx(
        orthant_distance(
            von_neumann_winner(preference_matrix), preference_matrix[None]
        ),
        abs=1e-5,
    )


@pytest.mark.parametrize(
    ("seed", "head_count", "size"), [(0, 2, 3), (1, 3, 5), (2, 4, 8), (3, 4, 16)]
)
def test_nash_target_winner_beats_every_other_policy_on_its_objective(
    seed, head_count, size
):
    preference_tensor = random_preference_tensor(seed, head_count, size)
    policy = nash_target_winner(preference_tensor)
    assert (policy >= 0).all()
    assert policy.sum() == pytest.approx(1.0)
    optimum = nash_distance(policy, preference_tensor)
    rivals = [
        *np.eye(size),
        *np.random.default_rng(seed).dirichlet(np.ones(size), size=2000),
        *np.random.default_rng(seed).dirichlet(np.full(size, 0.1), size=2000),
        blackwell_winner(preference_tensor),
        scalarised_nash(preference_tensor),
        max_mean_win_rate(preference_tensor),
        maximin(preference_tensor),
    ]
    assert (
        optimum
        <= min(nash_distance(rival, preference_tensor) for rival in rivals) + 1e-6
    )


@pytest.mark.parametrize("seed", range(5))
def test_nash_target_sits_between_the_orthant_and_the_mean_half_space(seed):
    # The orthant lies inside the Nash set, which lies inside the half-space
    # mean(z) >= 1/2, so the minimax distances must be ordered the same way
    preference_tensor = random_preference_tensor(seed, 3, 6)
    orthant_value = orthant_distance(
        blackwell_winner(preference_tensor), preference_tensor
    )
    nash_value = nash_distance(nash_target_winner(preference_tensor), preference_tensor)
    mean_value = mean_distance(
        target_set_winner(preference_tensor, np.full((1, 3), 1 / 3), np.array([0.5])),
        preference_tensor,
    )
    assert orthant_value >= nash_value - 1e-6
    assert nash_value >= mean_value - 1e-6


def test_nash_target_and_orthant_winners_trade_off_their_objectives():
    preference_tensor = random_preference_tensor(0, 3, 6)
    orthant_policy = blackwell_winner(preference_tensor)
    nash_policy = nash_target_winner(preference_tensor)
    assert (
        nash_distance(nash_policy, preference_tensor)
        < nash_distance(orthant_policy, preference_tensor) - 0.01
    )
    assert (
        orthant_distance(orthant_policy, preference_tensor)
        < orthant_distance(nash_policy, preference_tensor) - 0.01
    )


def test_nash_target_winner_is_equivariant_to_relabelling():
    preference_tensor = random_preference_tensor(4, 3, 6)
    candidate_order = np.array([3, 0, 5, 1, 4, 2])
    relabelled = preference_tensor[[2, 0, 1]][:, candidate_order][:, :, candidate_order]
    assert nash_distance(nash_target_winner(relabelled), relabelled) == pytest.approx(
        nash_distance(nash_target_winner(preference_tensor), preference_tensor),
        abs=1e-6,
    )
    assert nash_target_winner(relabelled) == pytest.approx(
        nash_target_winner(preference_tensor)[candidate_order], abs=1e-4
    )


def test_nash_target_winner_trivial_threshold_reaches_the_set():
    preference_tensor = random_preference_tensor(5, 3, 4)
    policy = nash_target_winner(preference_tensor, threshold=0.0)
    assert nash_shortfall(column_values(policy, preference_tensor), 0.0).max() == 0.0


@pytest.mark.parametrize(
    ("preference_tensor", "threshold"),
    [
        (np.full((3, 3), 0.5), 0.5),
        (np.full((2, 3, 4), 0.5), 0.5),
        (np.full((2, 0, 0), 0.5), 0.5),
        (np.full((2, 3, 3), 1.5), 0.5),
        (np.full((2, 3, 3), np.nan), 0.5),
        (np.full((2, 3, 3), 0.5), 1.5),
        (np.full((2, 3, 3), 0.5), -0.1),
    ],
)
def test_nash_target_winner_rejects_invalid_inputs(preference_tensor, threshold):
    with pytest.raises(ValueError):
        nash_target_winner(preference_tensor, threshold)


def test_nash_target_winner_solves_hard_preferences_without_warnings():
    # Five participants with 0/1 preferences, where exact power cones failed
    majority = skew_symmetric({(0, 1): 0.0}, 2)
    preference_tensor = np.stack([majority] * 4 + [1.0 - majority])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        policy = nash_target_winner(preference_tensor)
    grid = np.stack([np.linspace(0, 1, 10001), 1 - np.linspace(0, 1, 10001)], axis=1)
    assert (
        nash_distance(policy, preference_tensor)
        <= min(nash_distance(rival, preference_tensor) for rival in grid) + 1e-9
    )


def reference_geometric_mean(
    policy: np.ndarray, preference_tensor: np.ndarray
) -> float:
    row_count = preference_tensor.shape[1] // 2
    reference_win_rates = preference_tensor[:, :row_count, row_count:].mean(axis=2)
    return float(np.exp(np.log(reference_win_rates @ policy[:row_count]).mean()))


def nash_welfare_floor(policy: np.ndarray, preference_tensor: np.ndarray) -> float:
    """The worst pure opponent's geometric mean of head win rates."""
    return float(
        np.exp(np.log(column_values(policy, preference_tensor)).mean(axis=0)).min()
    )


def test_robust_nash_winner_with_one_head_guarantees_the_game_value():
    preference_tensor = random_preference_tensor(3, 1, 5)
    policy = robust_nash_winner(preference_tensor)
    assert column_values(policy, preference_tensor).min() == pytest.approx(
        0.5, abs=1e-5
    )


@pytest.mark.parametrize("seed", range(3))
def test_robust_nash_winner_beats_every_other_policy_on_its_objective(seed):
    preference_tensor = random_preference_tensor(seed, 4, 6)
    policy = robust_nash_winner(preference_tensor)
    floor = nash_welfare_floor(policy, preference_tensor)
    random_generator = np.random.default_rng(seed)
    for _ in range(50):
        other = random_generator.dirichlet(np.ones(6))
        assert nash_welfare_floor(other, preference_tensor) <= floor + 1e-5
    # The Nash target at this level is reached, so the target winner has no shortfall
    target_policy = nash_target_winner(preference_tensor, floor)
    assert nash_shortfall(
        column_values(target_policy, preference_tensor), floor
    ).max() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "preference_tensor",
    [np.zeros((2, 3)), np.full((2, 3, 3), 1.5), np.full((2, 2, 3), 0.5)],
)
def test_robust_nash_winner_rejects_invalid_inputs(preference_tensor):
    with pytest.raises(ValueError):
        robust_nash_winner(preference_tensor)


def budgeted_nash_value(
    policy: np.ndarray, preference_tensor: np.ndarray, error_budget: float
) -> float:
    """The paper's robust objective by its primal column problems, uniform reference weights."""
    head_count, size, _ = preference_tensor.shape
    worst = np.inf
    for opponent in range(size):
        others = [candidate for candidate in range(size) if candidate != opponent]
        floors = []
        for head in range(head_count):
            moved = cp.Variable(size - 1)
            problem = cp.Problem(
                cp.Minimize(policy[others] @ moved),
                [
                    moved >= 0,
                    moved <= 1,
                    cp.sum_squares(moved - preference_tensor[head, others, opponent])
                    * 2
                    / size**2
                    <= 2.25 * error_budget**2,
                ],
            )
            problem.solve(solver=cp.CLARABEL)
            floors.append(policy[opponent] / 2 + float(problem.value))  # pyright: ignore[reportArgumentType]
        worst = min(worst, float(np.prod(floors) ** (1 / head_count)))
    return worst


def test_budgeted_nash_winner_with_no_budget_is_the_level_free_winner():
    preference_tensor = random_preference_tensor(4, 3, 5)
    assert nash_welfare_floor(
        budgeted_nash_winner(preference_tensor, 0.0), preference_tensor
    ) == pytest.approx(
        nash_welfare_floor(robust_nash_winner(preference_tensor), preference_tensor),
        abs=1e-5,
    )


@pytest.mark.parametrize("seed", range(2))
def test_budgeted_nash_winner_maximises_the_primal_robust_objective(seed):
    preference_tensor = random_preference_tensor(seed, 2, 5)
    policy = budgeted_nash_winner(preference_tensor, 0.05)
    value = budgeted_nash_value(policy, preference_tensor, 0.05)
    assert value < nash_welfare_floor(policy, preference_tensor)
    random_generator = np.random.default_rng(seed)
    for _ in range(10):
        other = random_generator.dirichlet(np.ones(5))
        assert budgeted_nash_value(other, preference_tensor, 0.05) <= value + 1e-4


def test_budgeted_nash_winner_hedges_towards_uniform_as_the_budget_grows():
    preference_tensor = random_preference_tensor(5, 3, 6)
    assert budgeted_nash_winner(preference_tensor, 10.0) == pytest.approx(
        np.full(6, 1 / 6), abs=1e-3
    )


@pytest.mark.parametrize(
    "preference_tensor, error_budget",
    [
        (np.zeros((2, 3)), 0.1),
        (np.full((2, 3, 3), 0.5), -0.1),
        (np.full((2, 3, 3), 0.5), np.inf),
    ],
)
def test_budgeted_nash_winner_rejects_invalid_inputs(preference_tensor, error_budget):
    with pytest.raises(ValueError):
        budgeted_nash_winner(preference_tensor, error_budget)
