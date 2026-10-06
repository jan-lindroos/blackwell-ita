from itertools import combinations

import numpy as np
import pandas as pd

from blackwell_ita.generation import generate_responses
from blackwell_ita.models import RewardModel
from blackwell_ita.scoring import pairwise_preference_tensor, preferences_to_rewards
from blackwell_ita.selection import (
    SELECTION_METHODS,
    best_of_n,
    paired_bootstrap_interval,
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
HABERMAS_POOL_SIZE = 16
GENERATION_STRATEGIES = ["consensus", "diverse_v2"]


def usable_rankings(comparisons: pd.DataFrame) -> pd.DataFrame:
    """One row per human participant and candidate set they fully ranked.

    Rank 0 is best and ties share a rank.
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
        " consensus statement that the whole group could"
        " endorse. Reply with the statement only.\n\n"
        f"Question: {question}\n\n{numbered_opinions}"
    )


def diverse_candidate_prompts(groups: pd.DataFrame, seed: int = 1810) -> pd.DataFrame:
    """A fixed pool of 16 proposals, with stable participant IDs and varied order.

    Each block contains four consensus, four individual, six pair-focused,
    and two whole-group proposals. Targets use one-based IDs into group opinions.
    """
    rows = []
    for group in groups.to_dict("records"):
        opinions = list(group["opinions"])
        if len(opinions) != 4:
            raise ValueError("The diverse pool requires exactly four participants")
        # Rotate balanced orders; this balances positions across the pool,
        # not every candidate type or every possible order interaction.
        rng = np.random.default_rng(seed)
        slots = (
            [("consensus", ())] * 4
            + [("participant", (i,)) for i in range(1, 5)]
            + [("pair", pair) for pair in combinations(range(1, 5), 2)]
            + [("common_ground", ()), ("alternative_compromise", ())]
        )
        orders = np.array([[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]])
        labels = rng.permutation(4)
        for sample_index in range(HABERMAS_POOL_SIZE):
            kind, targets = slots[sample_index % 16]
            order = labels[orders[sample_index % 4]]
            targets = tuple(i + 1 for i in order if i + 1 in targets)
            numbered = "\n\n".join(f"Participant {i + 1}: {opinions[i]}" for i in order)
            if kind == "consensus":
                priority = (
                    "Seek a position that could attract broad support across the group. "
                    "Address shared concerns while acknowledging the main unresolved disagreement."
                )
            elif targets:
                priority = (
                    (
                        "Give equal weight to"
                        if kind == "pair"
                        else "Give particular weight to"
                    )
                    + " the stated priorities of participants "
                    + ", ".join(map(str, targets))
                    + ". Develop a proposal that preserves those priorities while making "
                    "a concrete accommodation for other participants' concerns. "
                    "Make any necessary trade-off explicit."
                )
            elif kind == "common_ground":
                priority = (
                    "Propose a limited, concrete step supported by concerns shared across "
                    "the group. Leave genuinely unresolved issues open."
                )
            else:
                priority = (
                    "Address the main disagreement through a concrete condition, safeguard, "
                    "exception, or staged implementation. Explain what that arrangement "
                    "accommodates and what disagreement remains."
                )
            prompt = (
                f"Question: {group['question']}\n\nParticipant opinions:\n{numbered}"
                "\n\nWrite one candidate proposal for this group to evaluate.\n"
                f"{priority}\n"
                "Give a concrete answer grounded in the supplied opinions. Explain the "
                "proposed policy, accommodation, or trade-off. Acknowledge relevant "
                "disagreement without inventing anyone's position. Do not claim that "
                "participants have agreed to your proposal. Do not describe this drafting "
                "task or mention participant numbers.\nReturn only the proposal."
            )
            rows.append(
                {
                    "question_id": group["question_id"],
                    "sample_index": sample_index,
                    "candidate_kind": kind,
                    "target_participants": list(targets),
                    "opinion_order": [int(i) + 1 for i in order],
                    "prompt": prompt,
                }
            )
    return pd.DataFrame(rows)


def generate_habermas_candidates(
    model_name: str,
    groups: pd.DataFrame,
    device: str,
    strategy: str = "diverse_v2",
) -> pd.DataFrame:
    """Generate a fixed 16-candidate pool on the notebook device."""
    if strategy not in GENERATION_STRATEGIES:
        raise ValueError(f"Unknown generation strategy: {strategy}")
    plan = diverse_candidate_prompts(groups)
    if strategy == "consensus":
        prompts = {
            group["question_id"]: consensus_prompt(
                group["question"], list(group["opinions"])
            )
            for group in groups.to_dict("records")
        }
        plan["prompt"] = plan["question_id"].map(
            lambda question_id: prompts[question_id]
        )
        plan["candidate_kind"] = "consensus"
        plan["target_participants"] = [[] for _ in range(len(plan))]
        plan["opinion_order"] = [list(range(1, 5)) for _ in range(len(plan))]
    plan["strategy"] = strategy
    responses = generate_responses(model_name, plan["prompt"].tolist(), 1, device)
    # Each prompt generates one response; preserve the group's sample index.
    return (
        plan.rename_axis("prompt_index")
        .reset_index()
        .merge(
            responses[["prompt_index", "response"]],
            on="prompt_index",
            validate="one_to_one",
        )
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


def selection_policies(
    selector_tensor: np.ndarray, selector_name: str
) -> dict[str, np.ndarray]:
    """Each method's mixture over the pool, with zero weight on the anchor.

    The anchor (the group's human top pick) sits at the last index. A
    Bradley-Terry selector also runs best-of-N on the rewards its tensor encodes.
    """
    selection_methods = SELECTION_METHODS | (
        {"best_of_n": lambda tensor: best_of_n(preferences_to_rewards(tensor))}
        if any(
            configuration.name == selector_name
            and configuration.model_type == "bradley_terry"
            for configuration in HABERMAS_CONFIGURATIONS
        )
        else {}
    )
    pool_size = selector_tensor.shape[1] - 1
    return {
        method_name: np.append(select(selector_tensor[:, :pool_size, :pool_size]), 0.0)
        for method_name, select in selection_methods.items()
    }


def model_track_results(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    grader_tensors: dict[str, np.ndarray],
    backbone_name: str,
    selector_name: str,
    grader_name: str,
) -> pd.DataFrame:
    """Select on one model's tensors over the generated pool, grade with another's.

    Win rates are against a uniformly random pool member, the anchor excluded.
    """
    result_rows = []
    for group_row in groups.to_dict("records"):
        selector_tensor = selector_tensors[group_row["question_id"]]
        grader_tensor = grader_tensors[group_row["question_id"]]
        pool_opponent = np.append(uniform_policy(selector_tensor.shape[1] - 1), 0.0)
        for method_name, policy in selection_policies(
            selector_tensor, selector_name
        ).items():
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
                    "pool", win_rates(policy, grader_tensor, pool_opponent)
                )
            )
    return pd.DataFrame(result_rows)


def bootstrap_summary(
    per_instance: pd.DataFrame, summary_columns: list[str]
) -> pd.DataFrame:
    """Mean ``value`` per group with a 95% bootstrap interval over instances."""
    summary_rows = []
    for group_values, group_results in per_instance.groupby(
        summary_columns, sort=False
    ):
        values = group_results["value"].to_numpy()
        low, high = paired_bootstrap_interval(values)
        summary_rows.append(
            dict(zip(summary_columns, group_values, strict=True))  # pyright: ignore[reportCallIssue, reportArgumentType]
            | {
                "value": values.mean(),
                "interval_low": low,
                "interval_high": high,
                "instances": len(values),
            }
        )
    return pd.DataFrame(summary_rows)


def pool_difference_summary(
    results: pd.DataFrame, welfare_names: list[str]
) -> pd.DataFrame:
    """Column method's pool win rate minus the row method's, pooled.

    Differences are taken within each selector-grader direction, then averaged
    within each backbone and question, which are the bootstrap instances.
    Best-of-N is only compared in the directions a Bradley-Terry model selects.
    """
    long_results = results.melt(
        id_vars=["backbone", "selector", "grader", "question_id", "method"],
        value_vars=[f"pool_{welfare_name}" for welfare_name in welfare_names],
        var_name="welfare",
    ).assign(welfare=lambda frame: frame["welfare"].str.removeprefix("pool_"))
    paired = long_results.merge(
        long_results,
        on=["backbone", "selector", "grader", "question_id", "welfare"],
        suffixes=("_row", "_column"),
    )
    per_instance = (
        paired.assign(
            value=paired["value_column"] - paired["value_row"],
            row_method=paired["method_row"],
            column_method=paired["method_column"],
        )
        .groupby(
            ["row_method", "column_method", "welfare", "backbone", "question_id"],
            sort=False,
        )["value"]
        .mean()
        .reset_index()  # pyright: ignore[reportAttributeAccessIssue]
    )
    return bootstrap_summary(per_instance, ["row_method", "column_method", "welfare"])
