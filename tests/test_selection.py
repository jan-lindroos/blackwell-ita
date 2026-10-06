import warnings

import numpy as np
import pandas as pd
import pytest

from blackwell_ita.selection import (
    blackwell_winner,
    clean_policy,
    max_mean_win_rate,
    maximin,
    nash_shortfall,
    nash_target_winner,
    paired_bootstrap_interval,
    scalarised_nash,
    summarise_methods,
    target_set_winner,
    von_neumann_winner,
    welfare,
    win_rates,
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
        win_rates(blackwell_policy, preference_tensor, np.full(3, 1 / 3)).min()
        > win_rates(
            scalarised_nash(preference_tensor), preference_tensor, np.full(3, 1 / 3)
        ).min()
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


def test_win_rates_against_a_pure_opponent_read_one_column():
    preference_tensor = np.stack(
        [skew_symmetric({(0, 1): 0.7}, 2), skew_symmetric({(0, 1): 0.2}, 2)]
    )
    assert win_rates(
        np.array([1.0, 0.0]), preference_tensor, np.array([0.0, 1.0])
    ) == pytest.approx([0.7, 0.2])


def test_welfare_minimum_geometric_and_arithmetic_means():
    assert welfare(np.array([0.25, 1.0])) == pytest.approx(
        {"egalitarian_welfare": 0.25, "nash_welfare": 0.5, "utilitarian_welfare": 0.625}
    )
    assert welfare(np.array([0.0, 1.0]))["nash_welfare"] == 0.0


def test_paired_bootstrap_interval_brackets_the_mean():
    differences = np.random.default_rng(0).normal(0.1, 0.05, size=200)
    low, high = paired_bootstrap_interval(differences)
    assert low < differences.mean() < high
    assert low > 0


def test_summarise_methods_pairs_blackwell_and_nash_per_instance():
    results = pd.DataFrame(
        {
            "selector": "model",
            "instance": [0, 0, 0, 1, 1, 1],
            "method": ["uniform", "scalarised_nash", "blackwell"] * 2,
            "score": [0.5, 0.6, 0.7, 0.5, 0.4, 0.6],
        }
    )
    summary_row = summarise_methods(results, ["selector"], "instance", ["score"]).iloc[
        0
    ]
    assert summary_row["instances"] == 2
    assert summary_row["blackwell"] == pytest.approx(0.65)
    assert summary_row["blackwell_minus_nash"] == pytest.approx(0.15)


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
