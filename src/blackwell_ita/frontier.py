import numpy as np
import pandas as pd
from scipy.optimize import linprog

FRONTIER_AXES = {"average": np.mean, "best": np.max}


def pool_win_rates(preference_tensor: np.ndarray) -> np.ndarray:
    """Each candidate's win rate against a uniformly drawn pool member, shape (head, candidate)."""
    return preference_tensor.mean(axis=2)


def weighted_optimum(
    win_rates: np.ndarray, worst_weight: float, other_head: int | None
) -> np.ndarray:
    """Per-head win rates of the mixture maximising a weighted worst-served objective.

    The objective is ``worst_weight`` times the worst-served head's win rate
    plus the rest times the head average, or ``other_head``'s win rate.
    """
    head_count, candidate_count = win_rates.shape
    other_coefficients = (
        win_rates.mean(axis=0) if other_head is None else win_rates[other_head]
    )
    # The variables are the mixture and then the worst-served value, which
    # cannot exceed any head's win rate
    solution = linprog(
        c=-np.append((1 - worst_weight) * other_coefficients, worst_weight),
        A_ub=np.hstack([-win_rates, np.ones((head_count, 1))]),
        b_ub=np.zeros(head_count),
        A_eq=np.append(np.ones(candidate_count), 0.0)[None],
        b_eq=[1.0],
        bounds=[(0, None)] * candidate_count + [(None, None)],
        method="highs",
    )
    if solution.status != 0:
        raise RuntimeError(f"Frontier linear programme failed: {solution.message}")
    return win_rates @ solution.x[:candidate_count]


def frontier_values(
    win_rates: np.ndarray, worst_weight: float, axis: str
) -> np.ndarray:
    """Per-head win rates of the frontier mixture for worst-served against ``axis``.

    The best-served head is not linear in the mixture, so that frontier takes
    one linear programme per head and keeps the best.
    """
    if axis == "average":
        return weighted_optimum(win_rates, worst_weight, None)
    return max(
        (
            weighted_optimum(win_rates, worst_weight, head)
            for head in range(len(win_rates))
        ),
        key=lambda values: (
            worst_weight * values.min() + (1 - worst_weight) * values.max()
        ),
    )


def averaged_frontier(
    win_rate_matrices: list[np.ndarray], axis: str, weight_count: int = 21
) -> pd.DataFrame:
    """Worst-served against ``axis``, averaged over instances, one row per weight.

    Each weight traces one point of the averaged frontier. Weights stay
    strictly inside 0 and 1 so every point is Pareto optimal.
    """
    frontier_rows = []
    for worst_weight in np.linspace(1e-3, 1 - 1e-3, weight_count):
        values = [
            frontier_values(win_rates, worst_weight, axis)
            for win_rates in win_rate_matrices
        ]
        frontier_rows.append(
            {
                "worst_weight": worst_weight,
                axis: float(np.mean([FRONTIER_AXES[axis](value) for value in values])),
                "worst": float(np.mean([value.min() for value in values])),
            }
        )
    return pd.DataFrame(frontier_rows)
