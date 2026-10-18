import numpy as np
import pytest

from blackwell_ita.frontier import (
    averaged_frontier,
    frontier_values,
    pool_win_rates,
)

# Two participants: X and Y each favour one of them, Z is balanced, W is dominated
TRADE_OFF = np.array([[0.9, 0.2, 0.5, 0.3], [0.2, 0.9, 0.5, 0.3]])


def test_pool_win_rates_average_over_the_opponent_candidate():
    preference_tensor = np.array([[[0.5, 1.0], [0.0, 0.5]]])
    assert pool_win_rates(preference_tensor).tolist() == [[0.75, 0.25]]


def test_worst_served_end_mixes_the_two_specialists():
    assert frontier_values(TRADE_OFF, 0.999, "average") == pytest.approx(
        [0.55, 0.55], abs=1e-6
    )
    assert frontier_values(TRADE_OFF, 0.999, "best") == pytest.approx(
        [0.55, 0.55], abs=1e-6
    )


def test_best_served_end_picks_a_specialist():
    assert sorted(frontier_values(TRADE_OFF, 0.001, "best")) == pytest.approx(
        [0.2, 0.9]
    )


@pytest.mark.parametrize("axis", ["average", "best"])
def test_no_mixture_beats_the_frontier_on_its_weighted_objective(axis):
    random_generator = np.random.default_rng(0)
    win_rates = random_generator.uniform(size=(3, 8))
    other = {"average": np.mean, "best": np.max}[axis]
    mixtures = random_generator.dirichlet(np.full(8, 0.3), size=3000)
    for worst_weight in [0.1, 0.5, 0.9]:
        values = frontier_values(win_rates, worst_weight, axis)
        optimum = worst_weight * values.min() + (1 - worst_weight) * other(values)
        rival_values = mixtures @ win_rates.T
        rivals = worst_weight * rival_values.min(axis=1) + (1 - worst_weight) * other(
            rival_values, axis=1
        )
        assert optimum >= rivals.max() - 1e-9


def test_averaged_frontier_averages_each_instance_and_runs_monotone():
    frontier = averaged_frontier([TRADE_OFF, TRADE_OFF[::-1]], "best", weight_count=5)
    assert frontier.columns.tolist() == ["worst_weight", "best", "worst"]
    assert frontier["worst"].iloc[-1] == pytest.approx(0.55, abs=1e-6)
    assert frontier["best"].iloc[0] == pytest.approx(0.9)
    assert frontier["worst"].is_monotonic_increasing
    assert frontier["best"].is_monotonic_decreasing
