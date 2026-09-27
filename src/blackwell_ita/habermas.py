import numpy as np
import pandas as pd

from blackwell_ita.models import RewardModel
from blackwell_ita.scoring import pairwise_preference_tensor
from blackwell_ita.selection import (
    SELECTION_METHODS,
    uniform_policy,
    welfare,
    win_rates,
)
from blackwell_ita.training import RewardModelConfiguration

COMPARISONS_URL = "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet"
COMPARISON_COLUMNS = {
    "question.id": "question_id",
    "question.text": "question",
    "round_id": "round_id",
    "iteration_index": "iteration_index",
    "metadata.participant_id": "participant_id",
    "own_opinion.text": "opinion",
    "candidates.text": "statements",
    "rankings.numerical_ranks": "ranks",
    "ratings.agreement": "agreements",
}
# MOCK marks a rating the interface filled in, so it counts as missing
AGREEMENT_SCALE = {
    "STRONGLY_DISAGREE": 1.0,
    "DISAGREE": 2.0,
    "SOMEWHAT_DISAGREE": 3.0,
    "NEUTRAL": 4.0,
    "SOMEWHAT_AGREE": 5.0,
    "AGREE": 6.0,
    "STRONGLY_AGREE": 7.0,
}
HABERMAS_CRITERIA = ["preference"]
HABERMAS_PREFIX = "habermas"
HABERMAS_CONFIGURATIONS = [
    RewardModelConfiguration(
        name="qwen3_4b_pairwise",
        model_type="pairwise",
        encoder_name="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=None,
        learning_rate=5e-6,
        evaluation_interval_steps=2500,
        max_tokens=2048,
    ),
    RewardModelConfiguration(
        name="qwen3_4b_bradley_terry",
        model_type="bradley_terry",
        encoder_name="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=None,
        learning_rate=5e-6,
        evaluation_interval_steps=2500,
        max_tokens=2048,
    ),
    RewardModelConfiguration(
        name="gemma4_12b_pairwise_lora",
        model_type="pairwise",
        encoder_name="google/gemma-4-12B-it",
        lora_rank=16,
        learning_rate=1e-4,
        evaluation_interval_steps=2500,
        max_tokens=2048,
    ),
]
GENERATION_BACKBONES = ["mistralai/Mistral-7B-Instruct-v0.3", "RLHFlow/LLaMA3-SFT-v2"]


def usable_rankings(comparisons: pd.DataFrame) -> pd.DataFrame:
    """One row per human participant and candidate set they fully ranked.

    Rank 0 is best and ties share a rank. Agreements map onto 1 to 7, NaN
    where missing.
    """
    completed = comparisons[
        (comparisons["metadata.provenance"] == "HUMAN_CITIZEN")
        & (comparisons["rankings.metadata.status"] == "COMPLETED")
    ]
    rankings = completed[list(COMPARISON_COLUMNS)].rename(columns=COMPARISON_COLUMNS)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
    fully_ranked = [
        len(ranks) >= 2 and len(ranks) == len(statements) and min(ranks) >= 0
        for ranks, statements in zip(
            rankings["ranks"], rankings["statements"], strict=True
        )
    ]
    rankings = rankings[fully_ranked]
    return rankings.assign(
        statements=[list(statements) for statements in rankings["statements"]],
        ranks=[[int(rank) for rank in ranks] for ranks in rankings["ranks"]],
        agreements=[
            [AGREEMENT_SCALE.get(agreement, np.nan) for agreement in agreements]
            for agreements in rankings["agreements"]
        ],
    ).reset_index(drop=True)


def load_habermas_rankings() -> pd.DataFrame:
    """Usable human rankings from the Habermas Machine candidate comparisons."""
    return usable_rankings(
        pd.read_parquet(
            COMPARISONS_URL,
            columns=[
                *COMPARISON_COLUMNS,
                "metadata.provenance",
                "rankings.metadata.status",
            ],
        )
    )


def rank_preference(first_rank: int, second_rank: int) -> float:
    """Probability target that the first statement is preferred, 0.5 on ties."""
    if first_rank < second_rank:
        return 1.0
    if first_rank > second_rank:
        return 0.0
    return 0.5


def habermas_prompt(question: str, opinion: str) -> str:
    """The context a pairwise model conditions on: question and one opinion."""
    return f"Question: {question}\n\nParticipant opinion: {opinion}"


def habermas_pairs(rankings: pd.DataFrame) -> pd.DataFrame:
    """Every unordered statement pair each participant ranked, with its target."""
    pair_rows = [
        {
            "question_id": ranking["question_id"],
            "split": ranking["split"],
            "prompt": habermas_prompt(ranking["question"], ranking["opinion"]),
            "response_a": ranking["statements"][first_index],
            "response_b": ranking["statements"][second_index],
            "preference": rank_preference(
                ranking["ranks"][first_index], ranking["ranks"][second_index]
            ),
        }
        for ranking in rankings.to_dict("records")
        for first_index in range(len(ranking["statements"]))
        for second_index in range(first_index + 1, len(ranking["statements"]))
    ]
    return pd.DataFrame(pair_rows)


def human_preference_tensor(participant_ranks: list[list[int]]) -> np.ndarray:
    """Observed preferences, shape (participant, statement, statement)."""
    ranks = np.asarray(participant_ranks)
    return (
        (ranks[:, :, None] < ranks[:, None, :])
        + 0.5 * (ranks[:, :, None] == ranks[:, None, :])
    ).astype(float)


def candidate_sets(rankings: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """One row per candidate set in ``split_name`` that two or more participants ranked."""
    split_rankings = rankings[rankings["split"] == split_name]
    set_rows = []
    for (round_id, iteration_index), set_rankings in split_rankings.groupby(  # pyright: ignore[reportGeneralTypeIssues]
        ["round_id", "iteration_index"], sort=True
    ):
        if len(set_rankings) < 2:
            continue
        set_rows.append(
            {
                "set_id": f"{round_id}_{iteration_index}",
                "question_id": set_rankings["question_id"].iloc[0],  # pyright: ignore[reportAttributeAccessIssue]
                "question": set_rankings["question"].iloc[0],  # pyright: ignore[reportAttributeAccessIssue]
                "statements": set_rankings["statements"].iloc[0],  # pyright: ignore[reportAttributeAccessIssue]
                "opinions": set_rankings["opinion"].tolist(),
                "participant_ranks": set_rankings["ranks"].tolist(),
                "participant_agreements": set_rankings["agreements"].tolist(),
            }
        )
    return pd.DataFrame(set_rows)


def deliberation_groups(
    rankings: pd.DataFrame,
    split_name: str,
    group_count: int = 100,
    participant_count: int = 4,
    seed: int = 1810,
) -> pd.DataFrame:
    """Four real participants per held-out question, with their group's top pick.

    Each question contributes its first opening candidate set that at least
    ``participant_count`` participants ranked. The anchor is that set's
    statement with the best mean rank among the chosen participants.
    """
    random_generator = np.random.default_rng(seed)
    opening_sets = candidate_sets(
        rankings[rankings["iteration_index"] == 0],  # pyright: ignore[reportArgumentType]
        split_name,
    )
    eligible_sets = (
        opening_sets[opening_sets["opinions"].map(len) >= participant_count]  # pyright: ignore[reportCallIssue]
        .drop_duplicates("question_id")
        .reset_index(drop=True)
    )
    chosen_positions = sorted(
        random_generator.choice(len(eligible_sets), size=group_count, replace=False)
    )
    group_rows = []
    for set_row in eligible_sets.iloc[chosen_positions].to_dict("records"):
        participant_positions = sorted(
            random_generator.choice(
                len(set_row["opinions"]),
                size=participant_count,
                replace=False,
            )
        )
        chosen_ranks = np.asarray(set_row["participant_ranks"])[participant_positions]
        group_rows.append(
            {
                "question_id": set_row["question_id"],
                "question": set_row["question"],
                "opinions": [
                    set_row["opinions"][position] for position in participant_positions
                ],
                "anchor": set_row["statements"][
                    int(chosen_ranks.mean(axis=0).argmin())
                ],
            }
        )
    return pd.DataFrame(group_rows)


def consensus_prompt(question: str, opinions: list[str]) -> str:
    """Generation instruction for a statement the whole group could endorse."""
    numbered_opinions = "\n\n".join(
        f"Participant {number}: {opinion}"
        for number, opinion in enumerate(opinions, start=1)
    )
    return (
        "A group of citizens is deliberating on the question below. Write a single"
        " consensus statement, in one paragraph, that the whole group could"
        " endorse. Reply with the statement only.\n\n"
        f"Question: {question}\n\n{numbered_opinions}"
    )


def participant_preference_tensor(
    model: RewardModel,
    question: str,
    opinions: list[str],
    statements: list[str],
    device: str,
) -> np.ndarray:
    """Predicted preferences, shape (participant, statement, statement)."""
    return np.stack(
        [
            pairwise_preference_tensor(
                model, habermas_prompt(question, opinion), statements, device
            )[0]
            for opinion in opinions
        ]
    )


def prefixed_welfare(
    prefix: str, per_participant_values: np.ndarray
) -> dict[str, float]:
    """Welfare metrics named ``prefix_<welfare>``."""
    return {
        f"{prefix}_{metric_name}": value
        for metric_name, value in welfare(per_participant_values).items()
    }


def human_track_results(
    sets: pd.DataFrame, predicted_tensors: dict[str, np.ndarray], selector_name: str
) -> pd.DataFrame:
    """Select on predicted preferences, score on the participants' own rankings.

    Win rates are against a uniformly random member of the candidate set,
    agreement is the expected 1 to 7 rating, over participants who rated.
    """
    result_rows = []
    for set_row in sets.to_dict("records"):
        human_tensor = human_preference_tensor(set_row["participant_ranks"])
        agreements = np.asarray(set_row["participant_agreements"], dtype=float)
        rated_participants = ~np.isnan(agreements).any(axis=1)
        for method_name, select in SELECTION_METHODS.items():
            policy = select(predicted_tensors[set_row["set_id"]])
            result_rows.append(
                {
                    "evidence": "human ground truth",
                    "selector": selector_name,
                    "method": method_name,
                    "set_id": set_row["set_id"],
                    "question_id": set_row["question_id"],
                    "candidate_count": len(policy),
                    "participant_count": len(human_tensor),
                }
                | prefixed_welfare(
                    "win_rate",
                    win_rates(policy, human_tensor, uniform_policy(len(policy))),
                )
                | (
                    prefixed_welfare(
                        "agreement", agreements[rated_participants] @ policy
                    )
                    if rated_participants.any()
                    else {}
                )
            )
    return pd.DataFrame(result_rows)


def model_track_results(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    grader_tensors: dict[str, np.ndarray],
    backbone_name: str,
    selector_name: str,
    grader_name: str,
) -> pd.DataFrame:
    """Select on one model's tensors over the generated pool, grade with another's.

    The anchor (the group's human top pick) sits at the last index, excluded
    from selection. Win rates are against the anchor and against a uniformly
    random pool member.
    """
    result_rows = []
    for group_row in groups.to_dict("records"):
        selector_tensor = selector_tensors[group_row["question_id"]]
        grader_tensor = grader_tensors[group_row["question_id"]]
        pool_size = selector_tensor.shape[1] - 1
        anchor_opponent = np.eye(pool_size + 1)[pool_size]
        pool_opponent = np.append(uniform_policy(pool_size), 0.0)
        for method_name, select in SELECTION_METHODS.items():
            policy = np.append(select(selector_tensor[:, :pool_size, :pool_size]), 0.0)
            result_rows.append(
                {
                    "evidence": f"model-based proxy, graded by {grader_name}",
                    "backbone": backbone_name,
                    "selector": selector_name,
                    "grader": grader_name,
                    "method": method_name,
                    "question_id": group_row["question_id"],
                }
                | prefixed_welfare(
                    "anchor", win_rates(policy, grader_tensor, anchor_opponent)
                )
                | prefixed_welfare(
                    "pool", win_rates(policy, grader_tensor, pool_opponent)
                )
            )
    return pd.DataFrame(result_rows)
