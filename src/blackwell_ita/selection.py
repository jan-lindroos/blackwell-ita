import cvxpy as cp
import numpy as np
import pandas as pd


def blackwell_winner(
    preference_tensor: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Policy minimising the worst shortfall below ``threshold`` over heads.

    The orthant target set {z : z_k >= threshold}: minimise the largest
    max(0, threshold - P_k(policy, e_i)) over pure opponents i and heads k,
    as a linear programme in epigraph form.
    """
    head_count, candidate_count, _ = preference_tensor.shape
    policy = cp.Variable(candidate_count, nonneg=True)
    shortfall = cp.Variable(nonneg=True)
    constraints = [cp.sum(policy) == 1] + [
        preference_tensor[head].T @ policy + shortfall >= threshold
        for head in range(head_count)
    ]
    problem = cp.Problem(cp.Minimize(shortfall), constraints)  # pyright: ignore[reportArgumentType]
    problem.solve(solver=cp.CLARABEL)
    if (
        problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or policy.value is None
    ):
        raise RuntimeError(f"blackwell_winner solve failed: {problem.status}")
    return clean_policy(np.asarray(policy.value))


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


SELECTION_METHODS = {
    "uniform": lambda preference_tensor: uniform_policy(preference_tensor.shape[1]),
    "scalarised_nash": scalarised_nash,
    "blackwell": blackwell_winner,
}


def win_rates(
    policy: np.ndarray, preference_tensor: np.ndarray, opponent: np.ndarray
) -> np.ndarray:
    """Per-head probability that ``policy`` beats the ``opponent`` distribution."""
    return np.einsum("i,kij,j->k", policy, preference_tensor, opponent)


def welfare(per_head_values: np.ndarray) -> dict[str, float]:
    """Rawlsian (minimum), Nash (geometric mean) and utilitarian (mean) welfare."""
    with np.errstate(divide="ignore"):
        log_values = np.log(per_head_values)
    return {
        "rawlsian_welfare": float(per_head_values.min()),
        "nash_welfare": float(np.exp(log_values.mean())),
        "utilitarian_welfare": float(per_head_values.mean()),
    }


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


def summarise_methods(
    results: pd.DataFrame,
    group_columns: list[str],
    instance_column: str,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Mean of each metric per method, plus Blackwell minus scalarised Nash.

    The difference is paired over instances with a bootstrap 95% interval.
    """
    summary_rows = []
    for group_values, group_results in results.groupby(group_columns, sort=False):
        for metric_column in metric_columns:
            metric_by_method = group_results.pivot_table(
                index=instance_column, columns="method", values=metric_column
            ).dropna()
            differences = (
                metric_by_method["blackwell"] - metric_by_method["scalarised_nash"]
            ).to_numpy()
            low, high = paired_bootstrap_interval(differences)
            summary_rows.append(
                dict(zip(group_columns, group_values, strict=True))  # pyright: ignore[reportCallIssue, reportArgumentType]
                | {"metric": metric_column, "instances": len(metric_by_method)}
                | metric_by_method.mean().to_dict()
                | {
                    "blackwell_minus_nash": differences.mean(),
                    "interval_low": low,
                    "interval_high": high,
                }
            )
    return pd.DataFrame(summary_rows)
