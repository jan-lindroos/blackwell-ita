import numpy as np
import pandas as pd
import pytest

from blackwell_ita.selection import (
    best_of_n,
    blackwell_winner,
    clean_policy,
    maximin,
    paired_bootstrap_interval,
    scalarised_nash,
    summarise_methods,
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


def test_best_of_n_and_maximin_pick_one_candidate_each():
    majority = skew_symmetric({(0, 1): 0.9, (0, 2): 0.9, (1, 2): 0.9}, 3)
    minority = skew_symmetric({(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.9}, 3)
    preference_tensor = np.stack([majority, majority, majority, minority])
    assert best_of_n(preference_tensor).tolist() == [1.0, 0.0, 0.0]
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
        {"rawlsian_welfare": 0.25, "nash_welfare": 0.5, "utilitarian_welfare": 0.625}
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
