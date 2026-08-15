"""Winner policies over candidate responses: Blackwell, Nash, and best-of-n."""

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog

from blackwell_ita.train_rms import bt_text, pairwise_text


def reward_scores(
    model,
    prompt: str,
    responses: list[str],
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Per-criterion reward scores for each response, shape (response, head)."""
    texts = [bt_text(prompt, response) for response in responses]
    with torch.no_grad():
        return (
            torch.cat(
                [
                    model.score(texts[start : start + batch_size], device).cpu()
                    for start in range(0, len(texts), batch_size)
                ]
            )
            .numpy()
        )


def preference_tensor(
    model,
    prompt: str,
    responses: list[str],
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Skew-symmetrised win probabilities, shape (head, response, response)."""
    count = len(responses)
    index_pairs = [(i, j) for i in range(count) for j in range(count) if i != j]
    texts = [
        pairwise_text(prompt, responses[i], responses[j]) for i, j in index_pairs
    ]
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            logits = model.score(texts[start : start + batch_size], device)
            probability_batches.append(torch.sigmoid(logits).cpu())
    probabilities = torch.cat(probability_batches).numpy()
    tensor = np.full((probabilities.shape[1], count, count), 0.5)
    for (i, j), row in zip(index_pairs, probabilities, strict=True):
        tensor[:, i, j] = row
    return (tensor + 1.0 - tensor.transpose(0, 2, 1)) / 2.0


def bt_preference_tensor(rewards: np.ndarray) -> np.ndarray:
    """BT-implied win probabilities, shape (head, response, response)."""
    gaps = rewards.T[:, :, None] - rewards.T[:, None, :]
    return 1.0 / (1.0 + np.exp(-gaps))


def blackwell_winner(preference_tensor: np.ndarray) -> np.ndarray:
    """Blackwell winner policy minimising the worst per-criterion shortfall."""
    # Data-oblivious target set S = {z : z >= 1/2}: minimise the worst clipped
    # shortfall max_{i,j} (1/2 - P_j(pi, e_i)) over pure opponents i and
    # criteria j. Constructing S from data replaces the fixed 1/2 in later work
    head_count, count, _ = preference_tensor.shape
    inequality_matrix = np.vstack(
        [
            np.hstack([-preference_tensor[head].T, -np.ones((count, 1))])
            for head in range(head_count)
        ]
    )
    result = linprog(
        c=[0.0] * count + [1.0],
        A_ub=inequality_matrix,
        b_ub=np.full(head_count * count, -0.5),
        A_eq=[[1.0] * count + [0.0]],
        b_eq=[1.0],
        bounds=[(0.0, None)] * count + [(0.0, None)],
    )
    if not result.success:
        raise RuntimeError(f"blackwell_winner linprog failed: {result.message}")
    return result.x[:count]


def best_of_nash(preference_matrix: np.ndarray) -> np.ndarray:
    """Von Neumann winner policy for a single preference matrix."""
    # The von Neumann winner is exactly the k=1 Blackwell winner: at tau = 1/2
    # the zero-shortfall policies are precisely the Nash equilibria
    return blackwell_winner(preference_matrix[None])


def bt_best_of_n(rewards: np.ndarray) -> np.ndarray:
    """One-hot policy on the highest-reward response."""
    policy = np.zeros(len(rewards))
    policy[int(np.argmax(rewards))] = 1.0
    return policy


def mean_win_rates(preference_tensor: np.ndarray) -> np.ndarray:
    """Mean win rate of each response against the rest, shape (head, response)."""
    # Subtracting the diagonal 1/2 removes each response's self-comparison;
    # the max guard keeps the degenerate single-candidate tensor finite
    count = preference_tensor.shape[1]
    return (preference_tensor.sum(axis=2) - 0.5) / max(count - 1, 1)


def mean_criterion_best_of_n(preference_tensor: np.ndarray) -> np.ndarray:
    """One-hot policy on the best mean-over-criteria win rate."""
    return bt_best_of_n(mean_win_rates(preference_tensor).mean(axis=0))


def worst_criterion_best_of_n(preference_tensor: np.ndarray) -> np.ndarray:
    """One-hot policy on the best worst-criterion win rate."""
    return bt_best_of_n(mean_win_rates(preference_tensor).min(axis=0))


def policy_support(
    policy: np.ndarray, threshold: float = 1e-6
) -> list[tuple[int, float]]:
    """Support atoms of a policy as (index, weight) pairs summing to one."""
    # LP solutions carry numerical dust: tiny negative entries and near-zero
    # atoms that would otherwise trigger spurious judge calls
    weights = np.clip(policy, 0.0, None)
    weights[weights < threshold] = 0.0
    weights = weights / weights.sum()
    return [
        (index, float(weight)) for index, weight in enumerate(weights) if weight > 0.0
    ]


def expected_scores(
    selections: pd.DataFrame, atom_scores: dict[tuple[str, str], float]
) -> pd.DataFrame:
    """Expected score per (prompt, method, n) policy from its atoms' scores."""
    scored = selections.assign(
        score=selections["weight"].to_numpy()
        * np.array(
            [
                atom_scores[(prompt, response)]
                for prompt, response in zip(
                    selections["prompt"], selections["response"], strict=True
                )
            ]
        )
    )
    grouped: pd.DataFrame = scored.groupby(["prompt", "method", "n"], as_index=False)[  # pyright: ignore[reportAssignmentType]
        ["weight", "score"]
    ].sum()
    grouped["score"] = grouped["score"] / grouped["weight"]
    return grouped.drop(columns=["weight"])
