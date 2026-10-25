"""Robust welfare and Blackwell regret of selection methods on a candidate pool.

Preference tensors have shape (group, candidate, candidate), entry [g, i, j]
being group g's probability of preferring candidate i to candidate j. The pool
is the response space, the reference policy is uniform over it and every pure
comparator is a pool candidate, so universal coverage is the pool size.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from blackwell_ita.scoring import preferences_to_rewards
from blackwell_ita.selection import (
    best_of_n,
    blackwell_winner,
    budgeted_nash_winner,
    max_mean_win_rate,
    maximin,
    nash_shortfall,
    nash_target_winner,
    paired_bootstrap_interval,
    robust_nash_winner,
    scalarised_nash,
    uniform_policy,
)

WELFARES = ["egalitarian", "utilitarian", "nash"]
# Each Blackwell target and the welfare index whose upper contour set it is
TARGET_WELFARES = {
    "orthant": "egalitarian",
    "half_space": "utilitarian",
    "nash": "nash",
}
METHOD_TARGETS = {
    "blackwell": "orthant",
    "scalarised_nash": "half_space",
    "blackwell_nash": "nash",
}
METHOD_WELFARES = {
    method_name: TARGET_WELFARES[target]
    for method_name, target in METHOD_TARGETS.items()
} | {"robust_nash": "nash", "budgeted_nash": "nash"}
# The Blackwell method whose target is each welfare's upper contour set
WELFARE_METHODS = {
    welfare: method_name
    for method_name, welfare in METHOD_WELFARES.items()
    if method_name in METHOD_TARGETS
}
BASELINES = ["uniform", "max_mean_win_rate", "maximin", "best_of_n"]
MAIN_METHODS = BASELINES + list(METHOD_TARGETS)
TARGET_SELECTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "orthant": blackwell_winner,
    "half_space": scalarised_nash,
}


def comparator_profiles(
    policy: np.ndarray, preference_tensor: np.ndarray
) -> np.ndarray:
    """Group win-rate vectors against each pure comparator, shape (comparator, group)."""
    return np.einsum("i,gij->jg", policy, preference_tensor)


def welfare_values(profiles: np.ndarray) -> dict[str, np.ndarray]:
    """The three welfare indices of each comparator's profile."""
    with np.errstate(divide="ignore"):
        geometric_means = np.exp(np.log(profiles).mean(axis=1))
    return {
        "egalitarian": profiles.min(axis=1),
        "utilitarian": profiles.mean(axis=1),
        "nash": geometric_means,
    }


def robust_welfare(
    policy: np.ndarray, preference_tensor: np.ndarray
) -> dict[str, float]:
    """Each welfare index against the comparator that minimises it."""
    values = welfare_values(comparator_profiles(policy, preference_tensor))
    return {name: float(index_values.min()) for name, index_values in values.items()}


def welfare_optima(preference_tensor: np.ndarray) -> dict[str, float]:
    """The best robust welfare any policy attains, one Blackwell winner per index."""
    return {
        "egalitarian": robust_welfare(
            blackwell_winner(preference_tensor), preference_tensor
        )["egalitarian"],
        "utilitarian": robust_welfare(
            scalarised_nash(preference_tensor), preference_tensor
        )["utilitarian"],
        "nash": robust_welfare(
            robust_nash_winner(preference_tensor), preference_tensor
        )["nash"],
    }


def optimal_losses(optima: dict[str, float]) -> dict[str, float]:
    """The optimal Blackwell loss of each target. The Nash target sits at its own optimum."""
    return {
        "orthant": 0.5 - optima["egalitarian"],
        "half_space": 0.5 - optima["utilitarian"],
        "nash": 0.0,
    }


def blackwell_loss(
    policy: np.ndarray, preference_tensor: np.ndarray, target: str, nash_level: float
) -> float:
    """Worst-case l-infinity distance of the policy's profiles to the target set.

    The orthant and half-space sit at level 1/2, the Nash target at ``nash_level``.
    """
    profiles = comparator_profiles(policy, preference_tensor)
    if target == "nash":
        return float(nash_shortfall(profiles.T, nash_level).max())
    return max(
        0.0, float(0.5 - welfare_values(profiles)[TARGET_WELFARES[target]].min())
    )


def selection_policies(
    selector_tensor: np.ndarray,
    bradley_terry: bool,
    nash_level: float,
    error_budget: float = 0.0,
    method_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Each method's mixture over the pool, selecting on the selector's tensor.

    ``nash_level`` is the level of the Nash target and ``error_budget`` the
    budget of the robust Nash algorithm. Best-of-N runs only for a
    Bradley-Terry selector, on the rewards its tensor encodes.
    ``method_names`` restricts the methods that run.
    """
    selectors = {
        "uniform": lambda: uniform_policy(selector_tensor.shape[1]),
        "max_mean_win_rate": lambda: max_mean_win_rate(selector_tensor),
        "maximin": lambda: maximin(selector_tensor),
        "blackwell": lambda: blackwell_winner(selector_tensor),
        "scalarised_nash": lambda: scalarised_nash(selector_tensor),
        "blackwell_nash": lambda: nash_target_winner(selector_tensor, nash_level),
        "robust_nash": lambda: robust_nash_winner(selector_tensor),
        "budgeted_nash": lambda: budgeted_nash_winner(selector_tensor, error_budget),
    }
    if bradley_terry:
        selectors["best_of_n"] = lambda: best_of_n(
            preferences_to_rewards(selector_tensor)
        )
    return {
        method_name: select()
        for method_name, select in selectors.items()
        if method_names is None or method_name in method_names
    }


def welfare_columns(
    policy: np.ndarray, gold_tensor: np.ndarray, optima: dict[str, float]
) -> dict[str, float]:
    """The policy's robust welfare, the optimum and the regret for each index."""
    achieved = robust_welfare(policy, gold_tensor)
    return (
        {f"{name}_welfare": achieved[name] for name in WELFARES}
        | {f"{name}_optimum": optima[name] for name in WELFARES}
        | {f"{name}_regret": optima[name] - achieved[name] for name in WELFARES}
    )


def subgroup_label(subgroup: tuple[int, ...]) -> str:
    """One-based participant positions, such as "1,3"."""
    return ",".join(str(position + 1) for position in subgroup)


def method_results(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    gold_tensors: dict[str, np.ndarray],
    selector_name: str,
    subgroups: list[tuple[int, ...]],
    bradley_terry: bool = False,
    method_names: list[str] | None = None,
) -> pd.DataFrame:
    """Welfare regret of every method per instance, selecting on the selector and grading on the gold.

    An instance is a question and a subgroup of its participants, and selection
    and grading see only the subgroup. The Nash target's level is the gold
    optimum and the error budget is the selector's model error against the
    gold. Methods with a target also record their Blackwell regret.
    """
    result_rows = []
    for question_id in groups["question_id"]:
        for subgroup in subgroups:
            selector_tensor = selector_tensors[question_id][list(subgroup)]
            gold_tensor = gold_tensors[question_id][list(subgroup)]
            optima = welfare_optima(gold_tensor)
            losses = optimal_losses(optima)
            policies = selection_policies(
                selector_tensor,
                bradley_terry,
                optima["nash"],
                float(model_error(selector_tensor, gold_tensor).max()),
                method_names,
            )
            for method_name, policy in policies.items():
                target = METHOD_TARGETS.get(method_name)
                result_rows.append(
                    {
                        "selector": selector_name,
                        "subgroup": subgroup_label(subgroup),
                        "group_size": len(subgroup),
                        "question_id": question_id,
                        "method": method_name,
                    }
                    | welfare_columns(policy, gold_tensor, optima)
                    | {
                        "blackwell_regret": (
                            blackwell_loss(policy, gold_tensor, target, optima["nash"])
                            - losses[target]
                        )
                        if target
                        else np.nan
                    }
                )
    return pd.DataFrame(result_rows)


def model_error(selector_tensor: np.ndarray, gold_tensor: np.ndarray) -> np.ndarray:
    """Each group's root mean square error over uniformly drawn ordered pairs."""
    return np.sqrt(((selector_tensor - gold_tensor) ** 2).mean(axis=(1, 2)))


def amplified_error(
    gold_tensor: np.ndarray, selector_tensor: np.ndarray, factor: float
) -> np.ndarray:
    """The gold moved ``factor`` times the selector's deviation, kept a valid preference tensor."""
    moved = np.clip(gold_tensor + factor * (selector_tensor - gold_tensor), 0.0, 1.0)
    return (moved + 1.0 - moved.swapaxes(1, 2)) / 2


def error_sweep(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    gold_tensors: dict[str, np.ndarray],
    factors: list[float],
    bradley_terry: bool = False,
) -> pd.DataFrame:
    """Whole-group method results at each error factor, with the worst group's model error."""
    frames = []
    for factor in factors:
        amplified = {
            question_id: amplified_error(
                gold_tensors[question_id], selector_tensors[question_id], factor
            )
            for question_id in groups["question_id"]
        }
        errors = {
            question_id: float(model_error(tensor, gold_tensors[question_id]).max())
            for question_id, tensor in amplified.items()
        }
        group_count = next(iter(gold_tensors.values())).shape[0]
        results = method_results(
            groups,
            amplified,
            gold_tensors,
            f"factor {factor:g}",
            [tuple(range(group_count))],
            bradley_terry,
        )
        frames.append(
            results.assign(
                factor=factor,
                model_error=results["question_id"].map(errors),  # pyright: ignore[reportArgumentType]
            )
        )
    return pd.concat(frames, ignore_index=True)


def budget_sweep(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    gold_tensors: dict[str, np.ndarray],
    budget_factors: list[float],
) -> pd.DataFrame:
    """Nash regret of the budgeted algorithm per instance, whole group, at each multiple of the true error."""
    result_rows = []
    for question_id in groups["question_id"]:
        selector_tensor = selector_tensors[question_id]
        gold_tensor = gold_tensors[question_id]
        optimum = welfare_optima(gold_tensor)["nash"]
        true_error = float(model_error(selector_tensor, gold_tensor).max())
        for factor in budget_factors:
            policy = budgeted_nash_winner(selector_tensor, factor * true_error)
            result_rows.append(
                {
                    "question_id": question_id,
                    "budget_factor": factor,
                    "model_error": true_error,
                    "nash_regret": optimum
                    - robust_welfare(policy, gold_tensor)["nash"],
                }
            )
    return pd.DataFrame(result_rows)


def deployed_policy(
    preference_tensor: np.ndarray,
    pool_size: int,
    draw_count: int,
    select: Callable[[np.ndarray], np.ndarray],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """The average over pools of the selector's mixture, as a distribution over the candidates.

    Each pool is ``pool_size`` candidates drawn with replacement from the
    uniform reference policy.
    """
    candidate_count = preference_tensor.shape[1]
    deployed = np.zeros(candidate_count)
    for _ in range(draw_count):
        drawn = random_generator.integers(candidate_count, size=pool_size)
        np.add.at(deployed, drawn, select(preference_tensor[:, drawn][:, :, drawn]))
    return deployed / draw_count


def sampling_results(
    groups: pd.DataFrame,
    gold_tensors: dict[str, np.ndarray],
    pool_sizes: list[int],
    draw_count: int,
    seed: int = 1810,
) -> pd.DataFrame:
    """Blackwell regret of the deployed policy per instance, pool size and target, the gold selecting.

    With the gold selecting there is no model error, so the regret is the
    sampling term alone. The Nash level is the gold optimum of the whole pool,
    fixed before any pool is drawn.
    """
    random_generator = np.random.default_rng(seed)
    result_rows = []
    for question_id in groups["question_id"]:
        gold_tensor = gold_tensors[question_id]
        optima = welfare_optima(gold_tensor)
        losses = optimal_losses(optima)
        nash_level = optima["nash"]
        selectors = TARGET_SELECTORS | {
            "nash": lambda tensor, level=nash_level: nash_target_winner(tensor, level)
        }
        for pool_size in pool_sizes:
            for target, select in selectors.items():
                deployed = deployed_policy(
                    gold_tensor, pool_size, draw_count, select, random_generator
                )
                result_rows.append(
                    {
                        "question_id": question_id,
                        "pool_size": pool_size,
                        "target": target,
                        "regret": blackwell_loss(
                            deployed, gold_tensor, target, optima["nash"]
                        )
                        - losses[target],
                    }
                )
    return pd.DataFrame(result_rows)


def kind_subsets(
    candidate_kinds: np.ndarray,
    kind_counts: dict[str, int],
    subset_count: int,
    random_generator: np.random.Generator,
) -> list[np.ndarray]:
    """Sorted candidate positions holding ``kind_counts[kind]`` candidates of each kind.

    Each subset draws without replacement within a kind. A composition that
    takes every candidate of each kind it names has a single subset.
    """
    positions = {kind: np.flatnonzero(candidate_kinds == kind) for kind in kind_counts}
    if all(len(positions[kind]) == count for kind, count in kind_counts.items()):
        return [np.sort(np.concatenate(list(positions.values())))]
    return [
        np.sort(
            np.concatenate(
                [
                    random_generator.choice(positions[kind], size=count, replace=False)
                    for kind, count in kind_counts.items()
                ]
            )
        )
        for _ in range(subset_count)
    ]


def pool_size_results(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    gold_tensors: dict[str, np.ndarray],
    candidate_kinds: dict[str, np.ndarray],
    pool_sizes: list[int],
    subset_count: int,
    bradley_terry: bool = True,
    seed: int = 1810,
) -> pd.DataFrame:
    """Robust welfare of each method's deployed policy per group and pool size.

    The whole pool is the response space and every candidate a comparator. A
    pool of size k holds equally many candidates of each kind, drawn without
    replacement, and the deployed policy averages the method's mixture over
    ``subset_count`` pools. The Nash target's level is the gold optimum of the
    whole pool.
    """
    random_generator = np.random.default_rng(seed)
    result_rows = []
    for question_id in groups["question_id"]:
        selector_tensor = selector_tensors[question_id]
        gold_tensor = gold_tensors[question_id]
        kinds = candidate_kinds[question_id]
        kind_names = sorted(set(kinds))
        optima = welfare_optima(gold_tensor)
        for pool_size in pool_sizes:
            subsets = kind_subsets(
                kinds,
                {kind: pool_size // len(kind_names) for kind in kind_names},
                subset_count,
                random_generator,
            )
            deployed: dict[str, np.ndarray] = {}
            for subset in subsets:
                policies = selection_policies(
                    selector_tensor[:, subset][:, :, subset],
                    bradley_terry,
                    optima["nash"],
                    method_names=MAIN_METHODS,
                )
                for method_name, policy in policies.items():
                    deployed.setdefault(method_name, np.zeros(len(kinds)))
                    deployed[method_name][subset] += policy / len(subsets)
            result_rows.extend(
                {"question_id": question_id, "pool_size": pool_size, "method": name}
                | welfare_columns(policy, gold_tensor, optima)
                for name, policy in deployed.items()
            )
    return pd.DataFrame(result_rows)


def targeting_results(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    gold_tensors: dict[str, np.ndarray],
    candidate_kinds: dict[str, np.ndarray],
    levels: dict[str, dict[str, int]],
    subset_count: int,
    bradley_terry: bool = True,
    seed: int = 1810,
) -> pd.DataFrame:
    """Robust welfare of each method per group and targeting level, averaged over subsets.

    A level gives the candidate count of each kind. Each of its subsets is a
    response space of its own, its candidates the only comparators, and the
    Nash target's level is the subset's gold optimum. The orthant's optimal
    loss is one half minus the egalitarian optimum.
    """
    random_generator = np.random.default_rng(seed)
    result_rows = []
    for question_id in groups["question_id"]:
        for level_name, kind_counts in levels.items():
            subset_rows = []
            for subset in kind_subsets(
                candidate_kinds[question_id],
                kind_counts,
                subset_count,
                random_generator,
            ):
                gold_tensor = gold_tensors[question_id][:, subset][:, :, subset]
                optima = welfare_optima(gold_tensor)
                policies = selection_policies(
                    selector_tensors[question_id][:, subset][:, :, subset],
                    bradley_terry,
                    optima["nash"],
                    method_names=MAIN_METHODS,
                )
                subset_rows.extend(
                    {"method": method_name}
                    | welfare_columns(policy, gold_tensor, optima)
                    | {"orthant_optimal_loss": 0.5 - optima["egalitarian"]}
                    for method_name, policy in policies.items()
                )
            averaged = (
                pd.DataFrame(subset_rows).groupby("method", sort=False).mean()
            ).reset_index()  # pyright: ignore[reportAttributeAccessIssue]
            result_rows.extend(
                averaged.assign(question_id=question_id, level=level_name).to_dict(
                    "records"
                )
            )
    return pd.DataFrame(result_rows)


def decisive_accuracy(selector_tensor: np.ndarray, gold_tensor: np.ndarray) -> float:
    """Share of the gold's decisive ordered pairs, at most 0.25 or at least 0.75, the selector orders the same way."""
    decisive = (gold_tensor <= 0.25) | (gold_tensor >= 0.75)
    return float(((selector_tensor > 0.5) == (gold_tensor > 0.5))[decisive].mean())


def paired_gaps(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Best-of-Blackwell's robust welfare minus each baseline's, per welfare and group, summarised.

    Each welfare's Best-of-Blackwell method is the one whose target is that
    welfare's upper contour set. Baselines missing from ``results`` are skipped.
    """
    index_columns = [*group_columns, "question_id"]
    gap_frames = []
    for welfare, blackwell_method in WELFARE_METHODS.items():
        welfare_column = f"{welfare}_welfare"
        method_welfare = {
            method_name: method_rows.set_index(index_columns)[welfare_column]
            for method_name, method_rows in results.groupby("method")
        }
        gap_frames.extend(
            (method_welfare[blackwell_method] - method_welfare[baseline])
            .rename("value")
            .reset_index()
            .assign(welfare=welfare, baseline=baseline)
            for baseline in BASELINES
            if baseline in method_welfare
        )
    return bootstrap_summary(
        pd.concat(gap_frames, ignore_index=True),
        [*group_columns, "welfare", "baseline"],
        "value",
    )


def bootstrap_summary(
    frame: pd.DataFrame, group_columns: list[str], value_column: str
) -> pd.DataFrame:
    """Mean of ``value_column`` per group with a 95% bootstrap interval over questions.

    Rows of one question are averaged before resampling, so instances that
    share a question count once.
    """
    summary_rows = []
    for group_values, group_frame in frame.groupby(group_columns, sort=False):
        values: np.ndarray = (
            group_frame.groupby("question_id")[value_column].mean().to_numpy()  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType]
        )
        low, high = paired_bootstrap_interval(values)
        summary_rows.append(
            dict(zip(group_columns, group_values, strict=True))  # pyright: ignore[reportCallIssue, reportArgumentType]
            | {
                "value": values.mean(),
                "interval_low": low,
                "interval_high": high,
                "instances": len(values),
            }
        )
    return pd.DataFrame(summary_rows)
