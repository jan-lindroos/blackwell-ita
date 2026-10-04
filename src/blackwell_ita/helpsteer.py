"""HelpSteer's generated-pool experiment; GPU stages live in the notebook."""

import functools
import hashlib
import itertools

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from blackwell_ita.generation import generate_responses
from blackwell_ita.hub import ARTIFACTS_REPOSITORY
from blackwell_ita.judging import JUDGE_CRITERIA
from blackwell_ita.scoring import (
    bradley_terry_rewards,
    pairwise_preference_tensor,
    rewards_to_preferences,
    score_with_resume,
)
from blackwell_ita.selection import best_of_n, blackwell_winner, target_set_winner
from blackwell_ita.training import RewardModelConfiguration, load_trained

HELPSTEER_CRITERIA = JUDGE_CRITERIA
HELPSTEER_PREFIX = "helpsteer-claude-v1"
HELPSTEER_POOL_SIZE = 64
TUNING_PROMPTS = 50
EVALUATION_PROMPTS = 200
GENERATION_TEMPERATURE = 1.2
GENERATION_MAX_TOKENS = 1024
GENERATION_BACKBONES = ["mistralai/Mistral-7B-Instruct-v0.3", "RLHFlow/LLaMA3-SFT-v2"]
CRITERION_PAIRS = list(itertools.combinations(range(4), 2))
TUNING_REGULARIZATION = 0.01
TUNING_VERSION = "coarse-to-fine-eighths-v1"


def pool_prefix(backbone: str) -> str:
    """Separate generator pools under the experiment's versioned Hub prefix."""
    if backbone not in GENERATION_BACKBONES:
        raise ValueError(f"Unknown generation backbone: {backbone}")
    return f"{HELPSTEER_PREFIX}/{backbone.split('/')[-1].lower()}"


def helpsteer_prompts(pairs: pd.DataFrame, seed: int = 1810) -> pd.DataFrame:
    """Persist 50 tuning and 200 evaluation prompts, exclusively from RM test.

    The anchor is the preferred response of a decisive human-labelled pair.
    Both stages stay outside RM training and RM early-stopping validation.
    """
    if pairs.groupby("prompt").split.nunique().max() != 1:
        raise ValueError("A prompt crosses reward-model splits")
    eligible = pd.DataFrame(
        pairs[(pairs.split == "test") & pairs.overall.notna() & (pairs.overall != 0.5)]
    )
    eligible = eligible.sort_values("prompt").drop_duplicates("prompt")
    count = TUNING_PROMPTS + EVALUATION_PROMPTS
    if len(eligible) < count:
        raise ValueError(f"Need {count} decisive test prompts; found {len(eligible)}")
    chosen = eligible.sample(n=count, random_state=seed).reset_index(drop=True)
    return pd.DataFrame(
        {
            "prompt_id": [
                hashlib.sha256(p.encode()).hexdigest()[:16] for p in chosen.prompt
            ],
            "prompt": chosen.prompt,
            "anchor": np.where(
                chosen.overall > 0.5, chosen.response_a, chosen.response_b
            ),
            "split": ["tuning"] * TUNING_PROMPTS + ["evaluation"] * EVALUATION_PROMPTS,
        }
    )


def validate_prompts(prompts: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Reject stale manifests and any overlap with reward-model training/validation."""
    if prompts.groupby("split").size().to_dict() != {
        "evaluation": EVALUATION_PROMPTS,
        "tuning": TUNING_PROMPTS,
    }:
        raise ValueError("Expected 50 tuning and 200 evaluation prompts")
    if prompts.prompt.duplicated().any() or prompts.prompt_id.duplicated().any():
        raise ValueError("Prompts and IDs must be unique")
    expected = helpsteer_prompts(pairs)
    if (
        not prompts.sort_values("prompt_id")
        .reset_index(drop=True)
        .equals(expected.sort_values("prompt_id").reset_index(drop=True))
    ):
        raise ValueError(
            "Prompt manifest does not match the persisted reward-model split"
        )


def generate_helpsteer_candidates(
    backbone: str, prompts: pd.DataFrame, device: str
) -> pd.DataFrame:
    """Exactly 64 responses, temperature 1.2; no reservoir or post-hoc subsampling."""
    responses = generate_responses(
        backbone,
        prompts.prompt.tolist(),
        HELPSTEER_POOL_SIZE,
        device,
        max_new_tokens=GENERATION_MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE,
    )
    return (
        responses.merge(
            prompts.reset_index(drop=True)
            .rename_axis("prompt_index")
            .reset_index()[["prompt_index", "prompt_id"]],
            on="prompt_index",
            validate="many_to_one",
        )
        .drop(columns="prompt_index")
        .assign(
            backbone=backbone,
            temperature=GENERATION_TEMPERATURE,
            max_new_tokens=GENERATION_MAX_TOKENS,
            seed=1810,
        )
    )


def candidate_pools(
    prompts: pd.DataFrame, candidates: pd.DataFrame
) -> dict[str, list[str]]:
    """Validate complete candidate order; anchors are not part of candidate rows."""
    if set(candidates.prompt_id) != set(prompts.prompt_id):
        raise ValueError("Candidate prompts do not match the prompt manifest")
    for column, value in [
        ("temperature", GENERATION_TEMPERATURE),
        ("max_new_tokens", GENERATION_MAX_TOKENS),
        ("seed", 1810),
    ]:
        if column not in candidates or not (candidates[column] == value).all():
            raise ValueError(f"Candidate generation setting differs: {column}")
    if "backbone" not in candidates or candidates.backbone.nunique() != 1:
        raise ValueError(
            "Candidate artifact must contain exactly one generation backbone"
        )
    pools = {}
    for prompt_id, group in candidates.groupby("prompt_id"):
        ordered = group.sort_values("sample_index")
        if ordered.sample_index.tolist() != list(range(HELPSTEER_POOL_SIZE)):
            raise ValueError(
                f"Incomplete or duplicate candidate indices for {prompt_id}"
            )
        if not ordered.response.map(lambda value: isinstance(value, str)).all():
            raise ValueError("Missing candidate text")
        pools[str(prompt_id)] = ordered.response.tolist()
    return pools


def score_helpsteer_pools(
    configuration: RewardModelConfiguration,
    backbone: str,
    prompts: pd.DataFrame,
    candidates: pd.DataFrame,
    device: str,
) -> dict[str, np.ndarray]:
    """Resume scores on the Hub; BT stores raw rewards, pairwise stores tensors."""
    pools = candidate_pools(prompts, candidates)
    if not (candidates.backbone == backbone).all():
        raise ValueError(
            "Candidate generation backbone does not match the requested pool"
        )
    inputs = {
        row["prompt_id"]: (row["prompt"], pools[row["prompt_id"]] + [row["anchor"]])
        for row in prompts.to_dict("records")
    }
    load_model = functools.cache(
        lambda: load_trained(configuration, HELPSTEER_PREFIX, device)
    )

    def score_one(values: tuple[str, list[str]]) -> np.ndarray:
        model = load_model()
        if model.criteria != HELPSTEER_CRITERIA:
            raise ValueError(
                "Checkpoint criterion order does not match this experiment"
            )
        score = (
            bradley_terry_rewards
            if configuration.model_type == "bradley_terry"
            else pairwise_preference_tensor
        )
        return score(model, *values, device)

    scores = score_with_resume(
        ARTIFACTS_REPOSITORY,
        f"{configuration.name}_scores.npz",
        pool_prefix(backbone),
        inputs,
        score_one,
    )
    load_model.cache_clear()
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores


def selection_inputs(
    scores: np.ndarray, model_type: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Use only the 64 candidates for selection, never the trailing anchor."""
    size = HELPSTEER_POOL_SIZE + 1
    expected = (4, size) if model_type == "bradley_terry" else (4, size, size)
    if scores.shape != expected or not np.isfinite(scores).all():
        raise ValueError(f"Expected finite scores of shape {expected}")
    if model_type == "bradley_terry":
        return rewards_to_preferences(scores[:, :-1]), scores[:, :-1]
    if (
        (scores < 0).any()
        or (scores > 1).any()
        or not np.allclose(scores + scores.transpose(0, 2, 1), 1, atol=1e-6)
    ):
        raise ValueError("Pairwise scores must be complementary probabilities")
    return scores[:, :-1, :-1], None


def pairwise_normals(pair_weights: np.ndarray) -> np.ndarray:
    """One interpretable trade-off per unordered criterion pair."""
    normals = np.zeros((6, 4))
    for row, ((first, second), weight) in enumerate(
        zip(CRITERION_PAIRS, pair_weights, strict=True)
    ):
        normals[row, first], normals[row, second] = weight, 1 - weight
    return normals


def fit_target(games: list[np.ndarray], outcomes: list[np.ndarray]) -> dict:
    """Fit actual N=64 policy scores, not candidate/anchor two-response proxies.

    Two coarse coordinate sweeps, then eighth-step refinement until a full sweep
    makes no improvement. Both grids stay within [.25, .75]. A small fixed penalty
    favours equal pair weights and .5 thresholds. Only tuning labels enter.
    """
    if not games or len(games) != len(outcomes):
        raise ValueError("Need equally many tuning games and outcomes")
    for game, scores in zip(games, outcomes, strict=True):
        if scores.shape != (game.shape[1],) or not np.isfinite(scores).all():
            raise ValueError("Every tuning candidate needs a Claude overall score")
    weights, thresholds = np.full(6, 0.5), np.full(6, 0.5)

    @functools.cache
    def objective(parameters: tuple) -> float:
        w, b = np.array(parameters[:6]), np.array(parameters[6:])
        value = np.mean(
            [
                target_set_winner(game, pairwise_normals(w), b) @ scores
                for game, scores in zip(games, outcomes, strict=True)
            ]
        )
        penalty = np.mean((w - 0.5) ** 2) + np.mean((b - 0.5) ** 2)
        return float(value - TUNING_REGULARIZATION * penalty)

    best = objective(tuple(np.r_[weights, thresholds]))
    for step in [0.25, 0.125]:
        sweep = 0
        while True:
            improved = False
            for parameter in tqdm(range(12), desc=f"Target coordinates (step {step})"):
                for value in np.arange(0.25, 0.75 + step, step):
                    proposed = np.r_[weights, thresholds]
                    proposed[parameter] = value
                    score = objective(tuple(proposed))
                    if score > best + 1e-12:
                        weights, thresholds, best = proposed[:6], proposed[6:], score
                        improved = True
            sweep += 1
            if not improved or (step == 0.25 and sweep == 2):
                break
    return {
        "tuning_version": TUNING_VERSION,
        "pair_weights": weights.tolist(),
        "normals": pairwise_normals(weights).tolist(),
        "thresholds": thresholds.tolist(),
    }


def fit_scalar_weights(
    rewards: list[np.ndarray], outcomes: list[np.ndarray]
) -> list[float]:
    """Learn nonnegative weights on raw BT rewards using actual pool choices."""
    if not rewards or len(rewards) != len(outcomes):
        raise ValueError("Need equally many tuning rewards and outcomes")
    for values, scores in zip(rewards, outcomes, strict=True):
        if (
            values.shape[0] != 4
            or scores.shape != (values.shape[1],)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Every tuning candidate needs rewards and a Claude score")
    grid = (
        np.array([w for w in itertools.product(range(9), repeat=4) if sum(w) == 8]) / 8
    )
    penalties = np.mean((grid - 0.25) ** 2, axis=1)
    values = np.array(
        [
            np.mean(
                [best_of_n(r, w) @ y for r, y in zip(rewards, outcomes, strict=True)]
            )
            for w in grid
        ]
    )
    objective = values - TUNING_REGULARIZATION * penalties
    tied = np.flatnonzero(objective >= objective.max() - 1e-12)
    return grid[tied[np.argmin(penalties[tied])]].tolist()


def selection_policies(
    scores: np.ndarray, model_type: str, parameters: dict
) -> dict[str, np.ndarray]:
    """All Blackwells run on either scorer; scalar best-of-N is exclusively BT."""
    tensor, rewards = selection_inputs(scores, model_type)
    policies = {
        "blackwell_fixed": blackwell_winner(tensor),
        "blackwell_learned": target_set_winner(
            tensor, np.array(parameters["normals"]), np.array(parameters["thresholds"])
        ),
        "blackwell_overall": blackwell_winner(tensor[3:4]),
    }
    if rewards is not None:
        policies["best_of_n_overall"] = best_of_n(rewards[3:4])
        policies["best_of_n_weighted"] = best_of_n(
            rewards, np.array(parameters["scalar_weights"])
        )
    return policies
