from itertools import combinations

import numpy as np
import pandas as pd

from blackwell_ita.data import assign_prompt_splits
from blackwell_ita.frontier import averaged_frontier, pool_win_rates
from blackwell_ita.generation import generate_responses
from blackwell_ita.hub import ARTIFACTS_REPOSITORY, ensure_hub_dataframe
from blackwell_ita.models import RewardModel
from blackwell_ita.scoring import (
    pairwise_preference_tensor,
    preferences_to_rewards,
    win_rates_against,
)
from blackwell_ita.selection import (
    SELECTION_METHODS,
    best_of_n,
    paired_bootstrap_interval,
)
from blackwell_ita.training import RewardModelConfiguration

COMPARISONS_URL = "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet"
SURVEY_URL = "https://storage.googleapis.com/habermas_machine/datasets/hm_all_round_survey_responses.parquet"
# The data's stand-in for a participant who wrote nothing
PLACEHOLDER_OPINION = "No opinion was provided."
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
HABERMAS_PREFIX = "habermas-gold"
# The selecting proxies keep their checkpoints and runs apart from the gold's
SELECTOR_PREFIX = "habermas-selector"
# The gold model is trained on human rankings and stands in for the
# participants. The proxies select and only ever see gold labels
GOLD_CONFIGURATION = RewardModelConfiguration(
    name="gemma4_31b_pairwise_lora",
    model_type="pairwise",
    encoder_name="google/gemma-4-31B-it",
    lora_rank=16,
    learning_rate=1e-4,
    evaluation_interval_steps=2500,
    max_tokens=2048,
    batch_size=8,
)
PROXY_CONFIGURATIONS = [
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
]
GROUP_COUNTS = {"train": 300, "validation": 60, "test": 100}
DRAWS_PER_SLOT = {"train": 1, "validation": 1, "test": 2}
CANDIDATE_SLOTS = (
    [("consensus", ())] * 4
    + [("participant", (participant,)) for participant in range(1, 5)]
    + [("pair", pair) for pair in combinations(range(1, 5), 2)]
    + [("common_ground", ()), ("compromise", ())]
)
# Single-participant and pair slots only see their targets' opinions
SLOT_INSTRUCTIONS = {
    "consensus": "Write a statement the whole group could endorse.",
    "participant": "Write the statement as if the whole group shared this opinion.",
    "pair": "Write a statement both could endorse, giving their views equal weight.",
    "common_ground": "Keep to the points the participants share and leave their disagreements open.",
    "compromise": "Resolve the main disagreement with a concrete condition, exception or compromise.",
}
EXAMPLES_PER_PROMPT = 8
WHOLE_GROUP = [(0, 1, 2, 3)]
PARTICIPANT_PAIRS = list(combinations(range(4), 2))


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
    group_count: int,
    participant_count: int = 4,
    seed: int = 1810,
) -> pd.DataFrame:
    """Four real participants per question in ``split_name``.

    Each question contributes its first opening candidate set that at least
    ``participant_count`` participants ranked.
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
        group_rows.append(
            {
                "question_id": set_row["question_id"],
                "question": set_row["question"],
                "opinions": [
                    set_row["opinions"][position] for position in participant_positions
                ],
                "split": split_name,
            }
        )
    return pd.DataFrame(group_rows)


def ensure_rankings() -> pd.DataFrame:
    """Human rankings split by question, built and uploaded unless the hub has them."""
    return ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "rankings.parquet",
        HABERMAS_PREFIX,
        lambda: assign_prompt_splits(load_habermas_rankings(), "question_id"),
    )


def ensure_groups(rankings: pd.DataFrame) -> pd.DataFrame:
    """Deliberation groups for every split, built and uploaded unless the hub has them."""
    return ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "groups.parquet",
        HABERMAS_PREFIX,
        lambda: pd.concat(
            [
                deliberation_groups(rankings, split_name, group_count)
                for split_name, group_count in GROUP_COUNTS.items()
            ],
            ignore_index=True,
        ),
    )


def style_examples(
    rankings: pd.DataFrame, example_count: int = 400, seed: int = 1810
) -> pd.DataFrame:
    """Habermas Machine statements from train-split candidate sets, both rounds.

    Only statements of 30 to 220 words that open with a letter are kept,
    without the "Final consensus statement:" label some revisions carry.
    """
    statements = candidate_sets(rankings, "train")[
        ["question_id", "question", "statements"]
    ].explode(  # pyright: ignore[reportCallIssue]
        "statements"  # pyright: ignore[reportArgumentType]
    )
    word_counts = statements["statements"].str.split().str.len()  # pyright: ignore[reportAttributeAccessIssue]
    eligible = statements[
        word_counts.between(30, 220)
        & statements["statements"].str[0].str.isalpha()  # pyright: ignore[reportAttributeAccessIssue]
        & ~statements["statements"].str.startswith("Final consensus")  # pyright: ignore[reportAttributeAccessIssue]
    ]
    return (
        eligible.rename(columns={"statements": "statement"})  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        .sample(n=example_count, random_state=seed)
        .reset_index(drop=True)
    )


def statement_prompt(
    question: str,
    shown_opinions: list[str],
    instruction: str,
    example_text: str,
    target_words: int,
) -> str:
    """The generation prompt for one candidate statement."""
    opinion_text = "\n\n".join(f"Opinion: {opinion}" for opinion in shown_opinions)
    return (
        "Groups of ordinary UK citizens discuss a policy question, and a statement"
        " of the group's view is written for them in plain everyday language."
        f" Here are statements written for other groups.\n\n{example_text}\n\n"
        f"Write the statement for a new group.\n\nQuestion: {question}\n\n"
        f"{opinion_text}\n\n{instruction} Match the example statements closely"
        " in wording, phrasing habits and tone: plain everyday language, written"
        " as the group's view, and reusing the opinions' own words and phrases"
        " wherever possible, as the examples do. State the position plainly and"
        " confidently rather than hedging, and vary the opening as the examples"
        " do. Aim for about"
        f" {target_words} words in one paragraph, with no headings, lists or"
        " titles and no mention of participants or opinions. Reply with the"
        " statement only."
    )


def candidate_prompts(
    groups: pd.DataFrame, examples: pd.DataFrame, seed: int = 1810
) -> pd.DataFrame:
    """The generation prompts for every group's pool.

    Each draw covers the 16 slots once (four consensus, four single
    participant, six pair, one common ground and one compromise), so a
    test pool's first half holds one statement per slot and so does its
    second. The order opinions are shown in rotates. Every prompt shows
    three statements from other questions as style examples.
    """
    random_generator = np.random.default_rng(seed)
    orders = np.array([[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]])
    rows = []
    for group in groups.to_dict("records"):
        opinions = list(group["opinions"])
        if len(opinions) != 4:
            raise ValueError("The candidate pool requires exactly four participants")
        other_examples = examples[examples["question_id"] != group["question_id"]]
        example_lengths = examples["statement"].str.split().str.len().to_numpy()
        labels = random_generator.permutation(4)
        for draw in range(DRAWS_PER_SLOT[group["split"]]):
            for slot_index, (kind, targets) in enumerate(CANDIDATE_SLOTS):
                order = [
                    int(position) + 1
                    for position in labels[orders[(slot_index + draw) % 4]]
                    if not targets or int(position) + 1 in targets
                ]
                shown_examples = other_examples.iloc[
                    random_generator.choice(
                        len(other_examples), size=EXAMPLES_PER_PROMPT, replace=False
                    )
                ]
                example_text = "\n\n".join(
                    f"Question: {example['question']}\nStatement: {example['statement']}"
                    for example in shown_examples.to_dict("records")
                )
                rows.append(
                    {
                        "question_id": group["question_id"],
                        "split": group["split"],
                        "sample_index": draw * len(CANDIDATE_SLOTS) + slot_index,
                        "candidate_kind": kind,
                        "target_participants": list(targets),
                        "opinion_order": order,
                        "prompt": statement_prompt(
                            group["question"],
                            [opinions[participant - 1] for participant in order],
                            SLOT_INSTRUCTIONS[kind],
                            example_text,
                            # Lengths follow the Habermas statements
                            int(round(random_generator.choice(example_lengths), -1)),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def generate_candidates(prompts: pd.DataFrame) -> pd.DataFrame:
    """One generated statement per prompt row."""
    return prompts.assign(response=generate_responses(prompts["prompt"].tolist()))


def ensure_candidates(rankings: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Every group's generated pool, generated and uploaded unless the hub has it."""
    return ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "candidates.parquet",
        HABERMAS_PREFIX,
        lambda: generate_candidates(
            candidate_prompts(groups, style_examples(rankings))
        ),
    )


def pool_statements(candidates: pd.DataFrame, question_id: str) -> list[str]:
    """A group's generated statements in sample order."""
    return (
        candidates[candidates["question_id"] == question_id]
        .sort_values("sample_index")["response"]  # pyright: ignore[reportCallIssue]
        .tolist()
    )


def complete_groups(groups: pd.DataFrame) -> pd.DataFrame:
    """Groups in which every participant wrote an opinion."""
    return groups[  # pyright: ignore[reportReturnType]
        [PLACEHOLDER_OPINION not in list(opinions) for opinions in groups["opinions"]]
    ]


def load_final_statements() -> pd.DataFrame:
    """The statement each human participant saw as their round's outcome."""
    survey = pd.read_parquet(
        SURVEY_URL,
        columns=[
            "question.id",
            "metadata.participant_id",
            "metadata.provenance",
            "candidate.text",
        ],
    )
    return survey[survey["metadata.provenance"] == "HUMAN_CITIZEN"].rename(  # pyright: ignore[reportCallIssue, reportReturnType]
        columns={
            "question.id": "question_id",
            "metadata.participant_id": "participant_id",
            "candidate.text": "winner",
        }
    )[["question_id", "participant_id", "winner"]]  # pyright: ignore[reportReturnType]


def round_winners(
    groups: pd.DataFrame, rankings: pd.DataFrame, final_statements: pd.DataFrame
) -> pd.DataFrame:
    """Each group's round and the statement that round finally chose.

    The round is the one whose opening rankings hold all of the group's
    opinions. Its winner is the final statement its participants were shown.
    Groups whose round has no recorded winner are left out.
    """
    opening = rankings[rankings["iteration_index"] == 0]
    winners_by_round = (
        rankings[["round_id", "question_id", "participant_id"]]
        .drop_duplicates()
        .merge(final_statements, on=["question_id", "participant_id"])
        .groupby("round_id")["winner"]
        .agg(lambda statements: statements.mode().iloc[0])
    )
    winner_rows = []
    for group in groups.to_dict("records"):
        question_rankings = opening[opening["question_id"] == group["question_id"]]
        round_ids = [
            round_id
            for round_id, round_rankings in question_rankings.groupby("round_id")  # pyright: ignore[reportAttributeAccessIssue]
            if set(group["opinions"]) <= set(round_rankings["opinion"])
        ]
        if round_ids and round_ids[0] in winners_by_round.index:
            winner_rows.append(
                {
                    "question_id": group["question_id"],
                    "round_id": round_ids[0],
                    "winner": winners_by_round[round_ids[0]],
                }
            )
    return pd.DataFrame(winner_rows, columns=["question_id", "round_id", "winner"])


def ensure_winners(groups: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Test groups' round winners, built and uploaded unless the hub has them."""
    return ensure_hub_dataframe(
        ARTIFACTS_REPOSITORY,
        "winners.parquet",
        HABERMAS_PREFIX,
        lambda: round_winners(
            groups[groups["split"] == "test"],  # pyright: ignore[reportArgumentType]
            rankings,
            load_final_statements(),
        ),
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


def participant_winner_rates(
    model: RewardModel,
    question: str,
    opinions: list[str],
    statements: list[str],
    winner: str,
    device: str,
) -> np.ndarray:
    """Each statement's predicted win rate against the winner, shape (participant, statement)."""
    return np.stack(
        [
            win_rates_against(
                model, habermas_prompt(question, opinion), statements, winner, device
            )[0]
            for opinion in opinions
        ]
    )


def gold_pairs(
    groups: pd.DataFrame, candidates: pd.DataFrame, gold_tensors: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Every unordered pool pair per participant, targeted at the gold preference."""
    pair_rows = []
    for group in groups.to_dict("records"):
        statements = pool_statements(candidates, group["question_id"])
        gold_tensor = gold_tensors[group["question_id"]]
        for participant, opinion in enumerate(group["opinions"]):
            prompt = habermas_prompt(group["question"], opinion)
            for first_index, second_index in combinations(range(len(statements)), 2):
                pair_rows.append(
                    {
                        "question_id": group["question_id"],
                        "split": group["split"],
                        "prompt": prompt,
                        "response_a": statements[first_index],
                        "response_b": statements[second_index],
                        "preference": float(
                            gold_tensor[participant, first_index, second_index]
                        ),
                    }
                )
    return pd.DataFrame(pair_rows)


def selection_policies(
    selector_tensor: np.ndarray, selector_name: str
) -> dict[str, np.ndarray]:
    """Each method's mixture over the pool.

    A Bradley-Terry selector also runs best-of-N on the rewards its tensor
    encodes, summed over participants.
    """
    selection_methods = SELECTION_METHODS | (
        {"best_of_n": lambda tensor: best_of_n(preferences_to_rewards(tensor))}
        if any(
            configuration.name == selector_name
            and configuration.model_type == "bradley_terry"
            for configuration in PROXY_CONFIGURATIONS
        )
        else {}
    )
    return {
        method_name: select(selector_tensor)
        for method_name, select in selection_methods.items()
    }


def subgroup_label(subgroup: tuple[int, ...]) -> str:
    """One-based participant positions, such as "1,3"."""
    return ",".join(str(position + 1) for position in subgroup)


def evaluation_results(
    groups: pd.DataFrame,
    selector_tensors: dict[str, np.ndarray],
    gold_tensors: dict[str, np.ndarray],
    selector_name: str,
    subgroups: list[tuple[int, ...]],
    winner_rates: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Gold-graded results per participant, selecting on the proxy's tensors.

    Selection and grading only see each subgroup's participants. The win
    rate is against a uniformly drawn pool member. A candidate counts as
    beaten when the participant prefers the method's mixture to it with
    probability above 1/2, and as half beaten on an exact tie. With
    ``winner_rates`` the win rate against the round's winner is added, NaN
    where the round has none.
    """
    winner_rates = winner_rates or {}
    result_rows = []
    for group in groups.to_dict("records"):
        for subgroup in subgroups:
            selector_tensor = selector_tensors[group["question_id"]][list(subgroup)]
            gold_tensor = gold_tensors[group["question_id"]][list(subgroup)]
            for method_name, policy in selection_policies(
                selector_tensor, selector_name
            ).items():
                against_candidates = np.einsum("i,kij->kj", policy, gold_tensor)
                candidates_beaten = (against_candidates > 0.5 + 1e-9).sum(
                    axis=1
                ) + 0.5 * (np.abs(against_candidates - 0.5) <= 1e-9).sum(axis=1)
                for position, participant in enumerate(subgroup):
                    result_rows.append(
                        {
                            "selector": selector_name,
                            "subgroup": subgroup_label(subgroup),
                            "group_size": len(subgroup),
                            "question_id": group["question_id"],
                            "method": method_name,
                            "participant": participant + 1,
                            "win_rate": float(against_candidates[position].mean()),
                            "candidates_beaten": float(candidates_beaten[position]),
                            "winner_win_rate": float(
                                policy @ winner_rates[group["question_id"]][participant]
                            )
                            if group["question_id"] in winner_rates
                            else np.nan,
                        }
                    )
    return pd.DataFrame(result_rows)


def instance_welfare(results: pd.DataFrame) -> pd.DataFrame:
    """Average, best-served, worst-served and Nash welfare of each instance.

    The same average and worst-served values against the round's winner come
    as ``winner_average`` and ``winner_worst``.
    """
    return (
        results.groupby(
            ["selector", "group_size", "subgroup", "question_id", "method"],
            sort=False,
        )
        .agg(
            average=("win_rate", "mean"),
            best=("win_rate", "max"),
            worst=("win_rate", "min"),
            nash=(
                "win_rate",
                lambda win_rates: float(np.exp(np.log(win_rates).mean())),
            ),
            winner_average=("winner_win_rate", "mean"),
            winner_worst=("winner_win_rate", "min"),
        )
        .reset_index()
    )


def gold_frontier(
    groups: pd.DataFrame,
    gold_tensors: dict[str, np.ndarray],
    subgroups: list[tuple[int, ...]],
    axis: str,
) -> pd.DataFrame:
    """The achievable frontier under the gold model, averaged over instances."""
    return averaged_frontier(
        [
            pool_win_rates(gold_tensors[question_id][list(subgroup)])
            for question_id in groups["question_id"]
            for subgroup in subgroups
        ],
        axis,
    )


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


if __name__ == "__main__":
    # Pool generation calls the Claude API, so it runs locally, not on the GPU
    habermas_rankings = ensure_rankings()
    ensure_candidates(habermas_rankings, ensure_groups(habermas_rankings))
