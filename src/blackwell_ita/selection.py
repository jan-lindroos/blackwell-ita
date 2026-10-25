import warnings

import cvxpy as cp
import numpy as np


def blackwell_winner(
    preference_tensor: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Policy minimising the worst shortfall below ``threshold`` over heads.

    The orthant target set {z : z_k >= threshold}: minimise the largest
    max(0, threshold - P_k(policy, e_i)) over pure opponents i and heads k,
    as a linear programme in epigraph form.
    """
    return target_set_winner(
        preference_tensor,
        np.eye(preference_tensor.shape[0]),
        np.full(preference_tensor.shape[0], threshold),
    )


def target_set_winner(
    preference_tensor: np.ndarray, normals: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Minimax L-infinity distance to {z in R^k: normals @ z >= thresholds}.

    Nonnegative, unit-sum rows make the distance the largest positive constraint
    shortfall. The target is upward closed, without an upper box constraint.
    """
    if (
        preference_tensor.ndim != 3
        or preference_tensor.shape[1] != preference_tensor.shape[2]
        or preference_tensor.shape[1] == 0
        or not np.isfinite(preference_tensor).all()
        or np.any((preference_tensor < 0) | (preference_tensor > 1))
        or normals.ndim != 2
        or normals.shape[1] != preference_tensor.shape[0]
        or normals.shape[0] == 0
        or not np.isfinite(normals).all()
        or (normals < 0).any()
        or not np.allclose(normals.sum(axis=1), 1)
        or thresholds.shape != (len(normals),)
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError("Invalid preference tensor or monotone target set")
    transformed = np.einsum("rk,kij->rij", normals, preference_tensor)
    candidate_count = preference_tensor.shape[1]
    policy = cp.Variable(candidate_count, nonneg=True)
    shortfall = cp.Variable(nonneg=True)
    constraints = [cp.sum(policy) == 1] + [
        transformed[row].T @ policy + shortfall >= thresholds[row]
        for row in range(len(normals))
    ]
    problem = cp.Problem(cp.Minimize(shortfall), constraints)  # pyright: ignore[reportArgumentType]
    problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
    if (
        problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or policy.value is None
    ):
        raise RuntimeError(f"blackwell_winner solve failed: {problem.status}")
    result = clean_policy(np.asarray(policy.value))
    attained = max(
        0.0,
        float(
            (thresholds[:, None] - np.einsum("i,rij->rj", result, transformed)).max()
        ),
    )
    if abs(attained - float(np.asarray(problem.value))) > 1e-5:
        raise RuntimeError("Policy cleanup changed the target-set optimum")
    return result


def nash_target_winner(
    preference_tensor: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Minimax L-infinity distance to {z : geometric mean of z >= threshold}.

    The target is convex and upward closed, so the distance from z is the
    smallest t >= 0 with geo_mean(z + t) >= threshold. Each pure opponent adds
    one geometric-mean cone constraint.
    """
    if (
        preference_tensor.ndim != 3
        or preference_tensor.shape[1] != preference_tensor.shape[2]
        or preference_tensor.shape[1] == 0
        or not np.isfinite(preference_tensor).all()
        or np.any((preference_tensor < 0) | (preference_tensor > 1))
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid preference tensor or Nash target threshold")
    candidate_count = preference_tensor.shape[1]
    policy = cp.Variable(candidate_count, nonneg=True)
    shortfall = cp.Variable(nonneg=True)
    constraints = [cp.sum(policy) == 1] + [
        cp.geo_mean(preference_tensor[:, :, opponent] @ policy + shortfall) >= threshold
        for opponent in range(candidate_count)
    ]
    problem = cp.Problem(cp.Minimize(shortfall), constraints)  # pyright: ignore[reportArgumentType]
    # Equal weights make the second-order cone form exact, and Clarabel's
    # power cones fail on hard 0/1 preferences
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="geo_mean is being approximated")
        problem.solve(solver=cp.CLARABEL)
    if (
        problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or policy.value is None
    ):
        raise RuntimeError(f"nash_target_winner solve failed: {problem.status}")
    result = clean_policy(np.asarray(policy.value))
    attained = nash_shortfall(
        np.einsum("i,kij->kj", result, preference_tensor), threshold
    ).max()
    if abs(attained - float(np.asarray(problem.value))) > 1e-5:
        raise RuntimeError("Policy cleanup changed the target-set optimum")
    return result


def robust_nash_winner(preference_tensor: np.ndarray) -> np.ndarray:
    """Policy maximising the worst comparator's geometric mean of head win rates.

    The level-free Nash target: at its own optimum level the target set is
    reached exactly, so no level has to be chosen in advance.
    """
    if (
        preference_tensor.ndim != 3
        or preference_tensor.shape[1] != preference_tensor.shape[2]
        or preference_tensor.shape[1] == 0
        or not np.isfinite(preference_tensor).all()
        or np.any((preference_tensor < 0) | (preference_tensor > 1))
    ):
        raise ValueError("Invalid preference tensor for the robust Nash winner")
    candidate_count = preference_tensor.shape[1]
    policy = cp.Variable(candidate_count, nonneg=True)
    level = cp.Variable()
    constraints = [cp.sum(policy) == 1] + [
        cp.geo_mean(preference_tensor[:, :, opponent] @ policy) >= level
        for opponent in range(candidate_count)
    ]
    problem = cp.Problem(cp.Maximize(level), constraints)  # pyright: ignore[reportArgumentType]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="geo_mean is being approximated")
        problem.solve(solver=cp.CLARABEL)
    if (
        problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or policy.value is None
    ):
        raise RuntimeError(f"robust_nash_winner solve failed: {problem.status}")
    return clean_policy(np.asarray(policy.value))


def budgeted_nash_winner(
    preference_tensor: np.ndarray, error_budget: float
) -> np.ndarray:
    """Robust Nash alignment with an error budget, on a pool of distinct candidates.

    Maximises the worst Nash welfare over every skew-symmetric kernel whose root
    mean square deviation from the tensor, under uniform reference weights, is
    at most 3/2 of the budget in every head. Each head's win rate against a
    pure opponent is its dual column problem: the opponent's own half plus the
    largest penalised score of the remaining mass.
    """
    if (
        preference_tensor.ndim != 3
        or preference_tensor.shape[1] != preference_tensor.shape[2]
        or preference_tensor.shape[1] == 0
        or not np.isfinite(preference_tensor).all()
        or np.any((preference_tensor < 0) | (preference_tensor > 1))
        or not np.isfinite(error_budget)
        or error_budget < 0
    ):
        raise ValueError("Invalid preference tensor or error budget")
    head_count, candidate_count = preference_tensor.shape[:2]
    policy = cp.Variable(candidate_count, nonneg=True)
    level = cp.Variable()
    floors = cp.Variable((head_count, candidate_count), nonneg=True)
    mass_by_row = cp.reshape(policy, (candidate_count, 1), order="F") @ np.ones(
        (1, candidate_count)
    )
    # Every ordered pair has reference weight 1 / n^2 and its reverse moves with it
    penalty = 1.5 * error_budget * candidate_count / np.sqrt(2)
    constraints = [cp.sum(policy) == 1]
    for head in range(head_count):
        scores = cp.Variable((candidate_count, candidate_count), nonneg=True)
        constraints += [
            scores <= mass_by_row,
            cp.diag(scores) == 0,
            floors[head] + penalty * cp.norm(scores, 2, axis=0)
            <= policy / 2
            + cp.sum(cp.multiply(preference_tensor[head], scores), axis=0),
        ]
    constraints += [
        cp.geo_mean(floors[:, opponent]) >= level for opponent in range(candidate_count)
    ]
    problem = cp.Problem(cp.Maximize(level), constraints)  # pyright: ignore[reportArgumentType]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="geo_mean is being approximated")
        problem.solve(solver=cp.CLARABEL)
    if (
        problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or policy.value is None
    ):
        raise RuntimeError(f"budgeted_nash_winner solve failed: {problem.status}")
    return clean_policy(np.asarray(policy.value))


def nash_shortfall(values: np.ndarray, threshold: float) -> np.ndarray:
    """Per column, the smallest t >= 0 with geometric mean of values + t >= threshold.

    Values are (head, column) in [0, 1]. Bisection on [0, threshold], since
    adding the threshold itself always reaches the target.
    """
    low = np.zeros(values.shape[1])
    high = np.full(values.shape[1], float(threshold))
    with np.errstate(divide="ignore"):
        for _ in range(60):
            middle = (low + high) / 2
            reached = np.exp(np.log(values + middle).mean(axis=0)) >= threshold
            high = np.where(reached, middle, high)
            low = np.where(reached, low, middle)
        return np.where(np.exp(np.log(values).mean(axis=0)) >= threshold, 0.0, high)


def von_neumann_winner(preference_matrix: np.ndarray) -> np.ndarray:
    """Nash equilibrium of the symmetric game: the one-head Blackwell winner."""
    return blackwell_winner(preference_matrix[None])


def scalarised_nash(preference_tensor: np.ndarray) -> np.ndarray:
    """Von Neumann winner of the head-averaged preference matrix."""
    return von_neumann_winner(preference_tensor.mean(axis=0))


def clean_policy(policy: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Clip the LP's numerical dust and renormalise."""
    weights = np.clip(policy, 0.0, None)
    weights[weights < threshold] = 0.0
    return weights / weights.sum()


def uniform_policy(candidate_count: int) -> np.ndarray:
    """Uniform random choice over the candidates."""
    return np.full(candidate_count, 1.0 / candidate_count)


def max_mean_win_rate(preference_tensor: np.ndarray) -> np.ndarray:
    """The candidate with the highest head-averaged win rate against a uniform opponent."""
    scores = preference_tensor.mean(axis=(0, 2))
    return np.eye(len(scores))[scores.argmax()]


def best_of_n(rewards: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pick the largest weighted scalar reward; rewards are (head, candidate).

    These are raw pointwise rewards, not averages of pairwise win probabilities.
    Ties choose the first candidate, as in max_mean_win_rate.
    """
    if rewards.ndim != 2 or 0 in rewards.shape or not np.isfinite(rewards).all():
        raise ValueError("Expected finite scalar rewards, shape (head, candidate)")
    if weights is None:
        weights = np.full(rewards.shape[0], 1 / rewards.shape[0])
    if (
        weights.shape != (rewards.shape[0],)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Scalar weights must be nonnegative and sum to one")
    return np.eye(rewards.shape[1])[(weights @ rewards).argmax()]


def maximin(preference_tensor: np.ndarray) -> np.ndarray:
    """The candidate whose worst-off head has the highest win rate against a uniform opponent."""
    scores = preference_tensor.mean(axis=2).min(axis=0)
    return np.eye(len(scores))[scores.argmax()]


def paired_bootstrap_interval(
    differences: np.ndarray, sample_count: int = 10000, seed: int = 1810
) -> tuple[float, float]:
    """95% interval of the mean paired difference, resampling instances."""
    random_generator = np.random.default_rng(seed)
    resampling_weights = random_generator.multinomial(
        len(differences),
        np.full(len(differences), 1.0 / len(differences)),
        size=sample_count,
    )
    resampled_means = resampling_weights @ differences / len(differences)
    low, high = np.quantile(resampled_means, [0.025, 0.975])
    return float(low), float(high)
